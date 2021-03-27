import torch
from torch.nn import Module
import torch.nn.functional as F
from torchvision import transforms
from tars.base.configurable import Configurable
from tars.envs.alfred_env import AlfredEnv
from tars.config.envs.alfred_env_config import AlfredEnvConfig
from tars.alfred.gen import constants


class Policy(Configurable, Module):
    def __init__(self):
        Configurable.__init__(self)
        Module.__init__(self)
        self.num_actions = len(AlfredEnvConfig.actions)
        self.int_mask_size = [constants.DETECTION_SCREEN_HEIGHT, constants.DETECTION_SCREEN_WIDTH]

    def clean_preds(self, action, int_mask):
        return action.argmax(1), F.interpolate(int_mask.round(), self.int_mask_size).squeeze(1)

    def forward(self, img, goal_inst, low_insts):
        '''
            Args:
                img: [N, C, H, W] tensor (transformed by get_img_transforms)
                goal_inst: [N, M] (transformed by get_text_transforms)
                low_insts: [N, M, ML] (transformed by get_text_transforms)
            Returns:
                action: [N, A]
                interaction_mask: [N, 1, H, W]

            Legend:
                N: batch size
                C, H, W: channels, height, width
                M: max sentence length
                ML: max number of low level instructions
                V: vocabulary size
                A: number of actions
        '''
        raise NotImplementedError

    @staticmethod
    def get_img_transforms():
        '''
            Returns the transforms needed to be applied on raw RGB images
            from simulator
        '''
        raise NotImplementedError

    @staticmethod
    def get_text_transforms():
        '''
            Returns a function that takes in a list of sentences and returns
            a list where each element of the list is a another list of
            size [sentence length]. Each element in the inner list is an integer
            denoting word number in a vocab
        '''
        raise NotImplementedError
