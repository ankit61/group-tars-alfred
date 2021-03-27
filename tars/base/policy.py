from torch.nn import Module
from tars.base.configurable import Configurable


class Policy(Configurable, Module):
    def __init__(self):
        Configurable.__init__(self)
        Module.__init__(self)

    def forward(self, img, goal_inst, low_insts):
        '''
            Args:
                img: [N, C, H, W] tensor
                goal_inst: [N, M, V]
                low_insts: [N, M, ML, V]
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
