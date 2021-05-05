import torch
from tars.config.base.model_config import ModelConfig


class EmbedAndReadoutConfig(ModelConfig):
    batch_size = 8
    patience = 10
    num_workers = 4

    def get_optim(self, parameters):
        return torch.optim.Adam(parameters, lr=1e-3)


class PretrainingDecoderConfig(ModelConfig):
    pass