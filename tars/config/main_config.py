import torch
import os
from pathlib import Path
from tars.base.config import Config


class MainConfig(Config):

    # gpu use
    use_gpu = torch.cuda.is_available()
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}' if use_gpu else 'cpu')

    # basic dirs
    alfred_dir = os.path.join(Path(__file__).parents[1], 'alfred/')
