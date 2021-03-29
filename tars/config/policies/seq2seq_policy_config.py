import os
from tars.base.config import Config


class Seq2SeqPolicyConfig(Config):
    stop_literal = '<<stop>>' # different from AlfredEnvConfig.stop_action
    goal_literal = '<<goal>>'
