from torch import optim
from tars.config.base.model_config import ModelConfig


class MultiLabelClassifierConfig(ModelConfig):
    batch_size = 64

    def get_optim(self, parameters):
        return optim.Adam(parameters)
