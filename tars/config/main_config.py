import torch
from tars.base.config import Config


class MainConfig(Config):
    def __init__(self):
        super(MainConfig, self).__init__()

        # gpu use
        self.use_gpu = torch.cuda.is_available()
        self.gpu_id = 0
        self.device = torch.device(f'cuda:{self.gpu_id}' if self.use_gpu else 'cpu')
