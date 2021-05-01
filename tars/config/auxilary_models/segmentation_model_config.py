import torch.optim as optim
from tars.base.config import Config


class SegmentationModelConfig(Config):
    batch_size = 32

    def get_optim(self, parameters):
        return optim.Adam(parameters, lr=1e-3)
