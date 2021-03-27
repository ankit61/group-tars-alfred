from argparse import Namespace
from tars.base.policy import Policy
from tars.alfred.models.model.seq2seq_im_mask import Module as BaselineModel
from tars.alfred.models.nn.resnet import Resnet


class BaselinePolicy(Policy):
    def __init__(self, model_load_path):
        super().__init__()

    def forward(self, img, goal_inst, low_insts):
        pass
