import os
from tars.config.base.model_config import ModelConfig


class Seq2SeqPolicyConfig(ModelConfig):
    stop_literal = '<<stop>>' # different from AlfredEnvConfig.stop_action
    goal_literal = '<<goal>>'
