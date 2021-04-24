import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from tars.base.model import Model
from tars.config.base.dataset_config import DatasetConfig


class MaskRCNN(Model):
    def __init__(self, model_load_path):
        self.model = maskrcnn_resnet50_fpn(num_classes=len(DatasetConfig.objects_list))
        self.model.load_state_dict(torch.load(self.conf.mask_rcnn_path))

    def configure_optimizers(self):
        raise NotImplementedError

    def forward(self, img):
        '''
            Args:
                img: [N, C, H, W] tensor
        '''
        raise NotImplementedError

    def training_step(self):
        raise NotImplementedError
