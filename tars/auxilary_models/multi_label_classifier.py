import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tars.base.model import Model
from tars.config.base.dataset_config import DatasetConfig
from tars.datasets.multi_label_dataset import MultiLabelDataset


class MultiLabelClassifier(Model):
    def __init__(self):
        self.num_classes = len(DatasetConfig.objects_list)
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = self.loss(pred, batch[1])
        self.log_dict({
            'loss': loss.item(),
            'train_acc': (pred.sigmoid().round() == batch[1]).sum() / pred.shape[0]
        })
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        pred = self(batch[0])
        loss = self.loss(pred, batch[1])
        self.log_dict({
            'val_loss': loss.item(),
            'val_acc': (pred.sigmoid().round() == batch[1]).sum() / pred.shape[0]
        })

    def shared_dataloader(self, type):
        dataset = MultiLabelDataset(type, self.get_img_transforms())
        return DataLoader(
                dataset, batch_size=self.conf.batch_size, pin_memory=True,
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
