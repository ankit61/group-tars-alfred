import torch.optim as optim
from tars.config.base.model_config import ModelConfig


class SegmentationModelConfig(ModelConfig):
    batch_size = 16

    def get_optim(self, parameters):
        return optim.Adam(parameters, lr=1e-3)
