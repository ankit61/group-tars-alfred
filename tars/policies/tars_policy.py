import itertools
from numpy.core.fromnumeric import sort
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tars.datasets.imitation_dataset import ImitationDataset
from tars.auxilary_models import VisionModule
from tars.auxilary_models import ContextModule
from tars.auxilary_models import ActionModule
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

        self.datasets = {}

        self.action_loss = nn.CrossEntropyLoss()
        self.object_loss = nn.CrossEntropyLoss()

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

    def shared_step(self, batch):
        # FIXME: Features needed:
        # gradient clipping, truncated BPTT, teacher forcing
        self.reset()

        ac_loss, obj_loss = 0, 0
        for mini_batch, batch_size in ImitationDataset.mini_batches(batch):
            self.trim_history(batch_size)
            action, _, int_object = self(
                                        mini_batch['images'],
                                        mini_batch['goal_inst'],
                                        mini_batch['low_insts']
                                    )
            ac_loss += self.action_loss(action, mini_batch['expert_actions'])
            obj_loss += self.object_loss(int_object, mini_batch['expert_int_objects'])

        return {
            'loss': ac_loss + obj_loss,
            'action_loss': ac_loss.item(),
            'object_loss': obj_loss.item()
        }

    def configure_optimizers(self):
        return self.conf.get_optim(self.parameters())

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        metrics = self.shared_step(batch)
        metrics = {f'val_{k}': metrics[k] for k in metrics}

        self.log_dict(metrics)

    def trim_history(self, batch_size):
        self.past_actions = self.past_actions[:batch_size]
        self.past_objects = self.past_objects[:batch_size]

        self.inst_lstm_hidden = self.inst_lstm_hidden[:batch_size]
        self.inst_lstm_cell = self.inst_lstm_cell[:batch_size]

        self.goal_lstm_hidden = self.goal_lstm_hidden[:batch_size]
        self.goal_lstm_cell = self.goal_lstm_cell[:batch_size]

    # data stuff

    def setup(self, stage):
        for type in ['train', 'valid_seen', 'valid_unseen']:
            self.datasets[type] = ImitationDataset(
                                    type=type,
                                    img_transforms=self.get_img_transforms(),
                                    text_transforms=self.text_transform,
                                    text_collate=self.action_module.context_emb_model.text_collate
                                )

    def get_img_transforms(self):
        return self.vision_module.get_img_transforms()

    def text_transform(self, sents, is_goal):
        return self.action_module.context_emb_model.text_transforms(sents, is_goal)

    def train_dataloader(self):
        return self.shared_dataloader('train')

    def val_dataloader(self):
        return [
            self.shared_dataloader('valid_seen'),
            self.shared_dataloader('valid_unseen')
        ]

    def shared_dataloader(self, type):
        return DataLoader(
                self.datasets[type], batch_size=self.conf.batch_size,
                collate_fn=self.datasets[type].collate, pin_memory=True,
                num_workers=self.conf.main.num_threads
            )

    def find_instance_mask(self, img, int_object):
        # run instance segmentation
        # argmax over channels
        # keep only mask over int_object
        # identify which instance to use using MOCA's style
        # return img with seg map over predicted instance only
        pass
