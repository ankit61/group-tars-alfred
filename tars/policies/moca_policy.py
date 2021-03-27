from tars.base.policy import Policy


class MocaPolicy(Policy):
    def __init__(self):
        super().__init__()

    def forward(self, img, goal_inst, low_insts):
        pass
