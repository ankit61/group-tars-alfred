import os
from tars.base.config import Config
from tars.config.main_config import MainConfig


class BaselinePolicyConfig(Config):
    saved_model_path = os.path.join(MainConfig.alfred_dir, 'exp/model:seq2seq_im_mask,name:base30_pm010_sg010_01/best_seen.pth')

    stop_literal = '<<stop>>' # different from AlfredEnvConfig.stop_action
    goal_literal = '<<goal>>'
