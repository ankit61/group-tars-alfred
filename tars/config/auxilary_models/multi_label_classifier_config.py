from torch import optim
from tars.base.config import Config


class MultiLabelClassifierConfig(Config):
    batch_size = 32

    def get_optim(self, parameters):
        return optim.Adam(parameters)
