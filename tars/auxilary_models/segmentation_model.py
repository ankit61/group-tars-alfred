import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from tars.base.model import Model
from tars.config.base.dataset_config import DatasetConfig


class SegmentationModel(Model):
    def __init__(self, device, model_load_path=None):
        super(SegmentationModel, self).__init__()
        self.num_classes = len(DatasetConfig.objects_list)
        self.model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            classes=self.num_classes
        )

        self.model.to(device)
        if model_load_path is not None:
            self.model.load_state_dict(torch.load(model_load_path, map_location=device))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, gt = batch
        pred = self(img)
        return self.loss(pred, gt)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
