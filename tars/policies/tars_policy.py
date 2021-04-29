import itertools
import torch
from tars.auxilary_models.vision_module import VisionModule
from tars.auxilary_models.context_module import ContextModule
from tars.auxilary_models.action_module import ActionModule
from tars.config.base.dataset_config import DatasetConfig
from tars.base.policy import Policy


class TarsPolicy(Policy):
    def __init__(self):
        super(TarsPolicy, self).__init__()

        self.num_objects = len(DatasetConfig.objects_list)
        self.object_na_idx = DatasetConfig.object_na_idx

        self.context_module = ContextModule(
                                self.num_actions, self.num_objects,
                                self.object_na_idx, self.conf
                            )
        self.vision_module = VisionModule(self.context_module.object_emb, self.object_na_idx, self.conf)
        self.action_module = ActionModule(
                                self.context_module.action_emb,
                                self.context_module.object_emb, self.conf
                            )

        self.reset() # initializes all past trajectory info

    def reset(self):
        self.past_actions = torch.full(
                            (self.conf.batch_size, self.conf.past_actions_len),
                            self.num_actions
                        )

        self.past_objects = torch.full(
                            (self.conf.batch_size, self.conf.past_objects_len),
                            self.object_na_idx
                        )

        self.actions_itr = itertools.cycle(range(self.past_actions.shape[1]))
        self.objects_itr = [itertools.cycle(range(self.past_objects.shape[1]))] * self.conf.batch_size

        self.inst_lstm_hidden = torch.zeros(self.conf.batch_size, self.conf.inst_hidden_size)
        self.inst_lstm_cell = torch.zeros(self.conf.batch_size, self.conf.inst_hidden_size)

        self.goal_lstm_hidden = torch.zeros(self.conf.batch_size, self.conf.goal_hidden_size)
        self.goal_lstm_cell = torch.zeros(self.conf.batch_size, self.conf.goal_hidden_size)

    def forward(self, img, goal_inst, low_insts):
        '''
            All elements in one batch must be from different envs
            Args:
                img: tensor of shape [N, C, H, W]
                goal_inst: [S, N]
                low_inst: [S', N]
        '''
        context = self.context_module(
                    self.past_actions,
                    self.past_objects,
                    self.inst_lstm_cell,
                    self.goal_lstm_cell
                )

        vision_features = self.vision_module(img)

        action, int_object, inst_hidden_cell, goal_hidden_cell =\
            self.action_module(
                goal_inst,
                low_insts,
                vision_features,
                context
            )

        self.inst_lstm_hidden, self.inst_lstm_cell = inst_hidden_cell
        self.goal_lstm_hidden, self.goal_lstm_cell = goal_hidden_cell

        # update history
        self.past_actions[:, next(self.actions_itr)] = action.argmax(1)
        int_obj_idxs = int_object.argmax(1)
        for i in range(int_obj_idxs.shape[0]):
            if int_obj_idxs[i] != self.object_na_idx:
                self.past_objects[i, next(self.objects_itr[i])] = int_obj_idxs[i]

        int_mask = self.find_instance_mask(img, int_object) if self.conf.use_mask else None

        return action, int_mask, int_object

    def training_step(self, batch, batch_idx):
        pass

    def find_instance_mask(self, img, int_object):
        # run instance segmentation
        # argmax over channels
        # keep only mask over int_object
        # identify which instance to use using MOCA's style
        # return img with seg map over predicted instance only
        pass
