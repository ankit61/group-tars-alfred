import os
from pathlib import Path
from tars.base.config import Config
from tars.config.main_config import MainConfig


class MocaPolicyConfig(Config):
    moca_dir = os.path.join(Path(__file__).parents[2], 'moca/')
    saved_model_path = os.path.join(moca_dir, 'exp/pretrained/pretrained.pth')
