import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import segmentation_models_pytorch as smp
from tars.base.model import Model
from tars.base.dataset import DatasetType
from tars.config.base.dataset_config import DatasetConfig
from torchvision import transforms
from tars.datasets.segmentation_dataset import SegmentationDataset
import pytorch_lightning.metrics as metrics


class SegmentationModel(Model):
    def __init__(self, encoder_name: str='resnet34', model_load_path:str=None):
        super(SegmentationModel, self).__init__()
        self.num_classes = len(DatasetConfig.objects_list)
        self.encoder_name = encoder_name
        self.model = smp.Unet(
            encoder_name=self.encoder_name,
            classes=self.num_classes
        )
        self.min_size = (320, 320) # will change based on model used

        self.model.to(self.conf.main.device)
        if model_load_path is not None:
            self.model.load_state_dict(torch.load(model_load_path, map_location=self.conf.main.device))
        self.loss = nn.CrossEntropyLoss()
        self.iou_metric = metrics.classification.IoU(num_classes=self.num_classes)

        self.datasets = {}

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, gt = batch
        pred = self(img)
        loss = self.loss(pred, gt)
        self.log('train_loss', loss.item())
        return loss

    def configure_optimizers(self):
        return self.conf.get_optim(self.parameters())

    def validation_step(self, batch, batch_idx, dataloader_idx):
        img, gt = batch
        pred = self(img)

        self.log_dict({
            'val_loss': self.loss(pred, gt).item(),
            'val_iou': self.iou_metric(pred.softmax(1), gt)
        })

    # data stuff
    def get_dataset(self, type):
        return SegmentationDataset(type, self.get_img_transforms())

    def setup(self, stage):
        for t in [DatasetType.TRAIN, DatasetType.VALID_SEEN, DatasetType.VALID_UNSEEN]:
            self.datasets[t] = self.get_dataset(t)

    def train_dataloader(self):
        return self.shared_dataloader(DatasetType.TRAIN)

    def val_dataloader(self):
        return [
            self.shared_dataloader(DatasetType.VALID_SEEN),
            self.shared_dataloader(DatasetType.VALID_UNSEEN),
        ]

    def shared_dataloader(self, type):
        return DataLoader(
            self.datasets[type], batch_size=self.conf.batch_size, pin_memory=True,
            num_workers=self.conf.main.num_threads - 1
        )

    def get_img_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.min_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
