import itertools
from numpy.core.fromnumeric import sort
import torch
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn.utils.rnn as rnn_utils
from tars.base.dataset import DatasetType
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

    def get_img_transforms(self):
        # forward to vision module
        return transforms.Compose([
            transforms.Resize(self.vision_module.detection_model.min_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def text_transform(self, sents, is_goal):
        return self.action_module.context_emb_model.text_transforms(sents, is_goal)

    def setup(self, stage):
        for type in ['train', 'valid_seen', 'valid_unseen']:
            self.datasets[type] = ImitationDataset(
                                    type=type,
                                    img_transforms=self.get_img_transforms(),
                                    text_transforms=self.text_transform
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
        dataset = self.datasets[type]

        def collate_fn(batch):
            out = {}
            sort_desc = (lambda x: sorted(x, key=lambda e: -e.shape[0]))
            out['expert_actions'] = rnn_utils.pack_sequence(
                                        sort_desc([b['expert_actions'] for b in batch])
                                    )
            out['expert_int_objects'] = rnn_utils.pack_sequence(
                                        sort_desc([b['expert_int_objects'] for b in batch])
                                    )

            out['images'] = rnn_utils.pack_sequence(
                                sort_desc([torch.stack(b['images']) for b in batch])
                            )

            out['goal_inst'] = \
                self.action_module.context_emb_model.text_collate(
                    [b['goal_inst'] for b in batch]
                )

            out['low_insts'] = \
                self.action_module.context_emb_model.text_collate(
                    [b['low_insts'] for b in batch]
                )

            return out

        return DataLoader(
                dataset, batch_size=self.conf.batch_size,
                collate_fn=collate_fn, pin_memory=True
            )

    def training_step(self, batch, batch_idx):
        self.reset()

    def find_instance_mask(self, img, int_object):
        # run instance segmentation
        # argmax over channels
        # keep only mask over int_object
        # identify which instance to use using MOCA's style
        # return img with seg map over predicted instance only
        pass
