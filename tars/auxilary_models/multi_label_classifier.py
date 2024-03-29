import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tars.base.model import Model
from tars.config.base.dataset_config import DatasetConfig
from tars.datasets.multi_label_dataset import MultiLabelDataset


class MultiLabelClassifier(Model):
    def __init__(self, model_load_path=None):
        super(MultiLabelClassifier, self).__init__()
        self.num_classes = len(DatasetConfig.objects_list)
        self.model_load_path = model_load_path

        self.model = getattr(models, self.conf.model_name)(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.loss = nn.BCEWithLogitsLoss()

        if self.model_load_path is not None:
            self.load_from_checkpoint(
                self.model_load_path,
                map_location=self.conf.main.device
            )

    def forward(self, img):
        return self.model(img)

    def configure_optimizers(self):
        return self.conf.get_optim(self.parameters())

    def shared_step(self, batch, metric_prefix):
        pred = self(batch[0])
        loss = self.loss(pred, batch[1])
        metrics = self.get_metrics(pred, batch[1])
        assert 'loss' not in metrics
        metrics['loss'] = loss
        metrics = {f'{metric_prefix}_{k}': metrics[k] for k in metrics}
        return metrics

    def get_metrics(self, pred, gt):
        class_pred = self.predict_classes(pred)
        class_pred = class_pred.reshape(-1).float()
        gt = gt.reshape(-1)
        
        true_pos = torch.dot(class_pred, gt)
        precision = true_pos / class_pred.sum()
        recall = true_pos / gt.sum()
        f_score = 2 * ((precision * recall) / (precision + recall))
        
        return {
            'precision': precision,
            'recall': recall,
            'f_score': f_score
        }


    def predict_classes(self, pred):
        return (pred.sigmoid() > self.conf.pred_threshold).int()
        # try to just predict top 10 and see validation metrics
        # implemented in vision module

    def training_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, metric_prefix='train')
        metrics['loss'] = metrics.pop('train_loss')
        self.log_dict(metrics)
        return metrics['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx):
        metrics = self.shared_step(batch, metric_prefix='val')
        self.log_dict(metrics)

    def shared_dataloader(self, type):
        dataset = MultiLabelDataset(type, self.get_img_transforms())
        return DataLoader(
                dataset, batch_size=self.conf.batch_size, pin_memory=True,
                num_workers=self.conf.main.num_threads - 1
            )

    @classmethod
    def get_img_transforms(cls):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
