import torch.optim as optim
from tars.config.base.model_config import ModelConfig


class SegmentationModelConfig(ModelConfig):
    batch_size = 24
    accumulate_grad_batches = 4
    smp_model = 'Unet'  # should exactly match a model in segmentation_models_pytorch

    def get_optim(self, parameters):
        return optim.SGD(parameters, lr=1e-3, momentum=0.9)
