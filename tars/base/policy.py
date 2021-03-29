import torch
from torch.nn import Module
import torch.nn.functional as F
from torchvision import transforms
from tars.base.configurable import Configurable
from tars.config.envs.alfred_env_config import AlfredEnvConfig
from tars.alfred.gen import constants


class Policy(Configurable, Module):
    def __init__(self):
        Configurable.__init__(self)
        Module.__init__(self)
        self.num_actions = len(AlfredEnvConfig.actions)
        self.int_mask_size = [constants.DETECTION_SCREEN_HEIGHT, constants.DETECTION_SCREEN_WIDTH]

    def clean_preds(self, action, int_mask):
        return action.argmax(1).cpu().numpy(), \
            F.interpolate(int_mask.round(), self.int_mask_size).round().squeeze(1).bool().cpu().numpy()

    @classmethod
    def get_action_str(cls, clean_action):
        '''
            Args:
                clean_action: tensor of shape [N] containing action indexes
        '''
        return AlfredEnvConfig.actions.index2word(list(map(lambda x: x.item(), clean_action)))

    def reset(self):
        '''
            Clear any stateful info
        '''
        raise NotImplementedError

    def forward(self, img, goal_inst, low_insts):
        '''
            Args:
                img: [N, C, H, W] tensor (transformed by get_img_transforms)
                goal_inst: list of shape [N, 1, M] (transformed by get_text_transforms)
                low_insts: list of shape [N, M, ML] (transformed by get_text_transforms)
            Returns:
                action: [N, A]
                interaction_mask: [N, 1, H, W]

            Legend:
                N: batch size
                C, H, W: channels, height, width
                M: sentence lengths (varies per sentence, but for ease of notation displayed as one variable)
                ML: number of low level instructions (varies per env, but for ease of notation displayed as one variable)
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

    def get_text_transforms(self):
        '''
            Returns a function that takes in a list of sentences and returns
            a list where each element of the list is a another list of
            size [sentence length]. Each element in the inner list is an integer
            denoting word number in a vocab
        '''
        raise NotImplementedError
