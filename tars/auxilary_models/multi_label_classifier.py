import torch.nn as nn
from torchvision import models, transforms
from tars.base.model import Model
from tars.config.base.dataset_config import DatasetConfig


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
        return {
            'loss': self.loss(pred, batch[1]),
            'acc': 0, # compute acc here - probably cant use default acc fuctions because this is multi-class classification
        }

    def validation_step(self, batch, batch_idx, dataloader_idx):
        pred = self(batch[0])
        loss = self.loss(pred, batch[1])
        return {
            'loss': loss.item(),
            'acc': 0 # # compute acc here - probably cant use default acc fuctions because this is multi-class classification
        }

    # data stuff
    def shared_dataloader(self, type):
        raise NotImplementedError

    def get_img_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.min_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
