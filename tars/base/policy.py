from typing import Union, List
import torch.nn.functional as F
from tars.base.model import Model
from tars.config.envs.alfred_env_config import AlfredEnvConfig
from tars.alfred.gen import constants


class Policy(Model):
    def __init__(self):
        Model.__init__(self)
        self.num_actions = len(AlfredEnvConfig.actions)
        self.int_mask_size = [constants.DETECTION_SCREEN_HEIGHT, constants.DETECTION_SCREEN_WIDTH]

    def clean_preds(self, action, int_mask, int_object):
        clean_action = action.argmax(1).cpu().numpy()
        clean_mask = None if int_mask is None else F.interpolate(int_mask.round(), self.int_mask_size).round().squeeze(1).bool().cpu().numpy()
        clean_object = None if int_object is None else int_object.argmax(1).cpu().numpy()
        return clean_action, clean_mask, clean_object

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
                interaction_object: [N, O] (optional)

            Legend:
                N: batch size
                C, H, W: channels, height, width
                M: sentence lengths (varies per sentence, but for ease of notation displayed as one variable)
                ML: number of low level instructions (varies per env, but for ease of notation displayed as one variable)
                A: number of actions
                O: number of objects
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
        return self.text_transform

    def text_transform(self, langs: Union[List[List[str]], List[str]], is_goal: bool) -> List[List[List]]:
        '''
            Args:
                langs: list of sentences
                is_goal: boolean denoting whether passed sentences are goals
                            or low level descs
            Returns:
                out: langs converted to integer indices as per
                        model's vocab
        '''
        raise NotImplementedError
