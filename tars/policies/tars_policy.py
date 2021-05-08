import itertools
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
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

        if self.conf.remove_context:
            assert self.conf.remove_goal_lstm, 'if there is no context, goal lstm cannot exist'

        self.context_module = ContextModule(
                                self.num_actions, self.num_objects,
                                self.object_na_idx, self.conf
                            )
        self.vision_module = VisionModule(self.num_objects, self.object_na_idx, self.conf)
        self.action_module = ActionModule(
                                self.context_module.action_embed_and_readout.embed,
                                self.context_module.int_object_embed_and_readout.embed,
                                self.conf
                            )

        if self.conf.use_mask:
            self.maskrcnn = maskrcnn_resnet50_fpn(num_classes=self.num_objects)
            self.maskrcnn.eval()
            self.maskrcnn.load_state_dict(torch.load(self.conf.mask_rcnn_path, map_location=self.conf.main.device))
            self.maskrcnn = self.maskrcnn.to(self.conf.main.device)

        self.datasets = {}

        self.action_loss = nn.CrossEntropyLoss(reduction='sum') # meaned at end
        self.object_loss = nn.CrossEntropyLoss(reduction='sum') # meaned at end

        self.reset() # initializes all past trajectory info

    def reset(self):
        self.past_actions = torch.full(
                            (self.conf.batch_size, self.conf.past_actions_len),
                            self.num_actions, device=self.conf.main.device
                        )

        self.past_objects = torch.full(
                            (self.conf.batch_size, self.conf.past_objects_len),
                            self.object_na_idx, device=self.conf.main.device

                        )

        self.actions_itr = itertools.cycle(range(self.past_actions.shape[1]))
        self.objects_itr = [itertools.cycle(range(self.past_objects.shape[1]))] * self.conf.batch_size

        self.inst_lstm_hidden = torch.zeros(self.conf.batch_size, self.conf.inst_hidden_size, device=self.conf.main.device)
        self.inst_lstm_cell = torch.zeros(self.conf.batch_size, self.conf.inst_hidden_size, device=self.conf.main.device)

        self.goal_lstm_hidden = torch.zeros(self.conf.batch_size, self.conf.goal_hidden_size, device=self.conf.main.device)
        self.goal_lstm_cell = torch.zeros(self.conf.batch_size, self.conf.goal_hidden_size, device=self.conf.main.device)

    def forward(self, img, goal_inst, low_insts):
        '''
            All elements in one batch must be from different envs
            Args:
                img: tensor of shape [N, C, H, W]
                goal_inst: [S, N]
                low_inst: [S', N]
        '''
        context = None
        if not self.conf.remove_context:
            context = self.context_module(
                        self.past_actions.clone(),
                        self.past_objects.clone(),
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

        self.update_history(action, int_object)

        int_mask = self.find_instance_mask(img, int_object) if self.conf.use_mask else None

        return action, int_mask, int_object

    def update_history(self, action, int_object):
        self.past_actions[:, next(self.actions_itr)] = action.argmax(1)
        int_obj_idxs = int_object.argmax(1)
        for i in range(int_obj_idxs.shape[0]):
            if int_obj_idxs[i] != self.object_na_idx:
                self.past_objects[i, next(self.objects_itr[i])] = int_obj_idxs[i]


    def shared_step(self, batch):
        # FIXME: Features needed:
        # gradient clipping, truncated BPTT, teacher forcing
        self.reset()

        ac_loss, obj_loss = 0, 0
        seq_len = 0
        pred_actions, pred_objects = [], []
        for mini_batch, batch_size in ImitationDataset.mini_batches(batch):
            self.trim_history(batch_size)
            action, _, int_object = self(
                                        mini_batch['images'],
                                        mini_batch['goal_inst'],
                                        mini_batch['low_insts']
                                    )
            pred_actions.append(action.argmax(1).float().mean())
            pred_objects.append(int_object.argmax(1).float().mean())
            ac_loss += self.action_loss(action, mini_batch['expert_actions'])
            expert_objs = mini_batch['expert_int_objects']
            object_mask = (expert_objs != self.object_na_idx)
            obj_loss += self.object_loss(int_object[object_mask], expert_objs[object_mask])
            seq_len += batch_size

        return {
            'loss': (ac_loss + obj_loss) / seq_len,
            'action_loss': ac_loss.item() / seq_len,
            'object_loss': obj_loss.item() / seq_len,
            'pred_action_std': torch.tensor(pred_actions).std(),
            'pred_object_std': torch.tensor(pred_objects).std()
        }

    def configure_optimizers(self):
        optim = self.conf.get_optim(self.parameters())
        scheduler = self.conf.get_lr_scheduler(optim)
        return (optim if scheduler is None else [optim], [scheduler])

    def training_step(self, batch, batch_idx):
        metrics = self.shared_step(batch)
        metrics = {f'train_{k}': metrics[k] for k in metrics}
        self.log_dict(metrics)
        metrics['loss'] = metrics.pop('train_loss')
        return metrics['loss']

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
        for type in [DatasetType.TRAIN, DatasetType.VALID_SEEN, DatasetType.VALID_UNSEEN]:
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

    def shared_dataloader(self, type):
        return DataLoader(
                self.datasets[type], batch_size=self.conf.batch_size,
                collate_fn=self.datasets[type].collate, pin_memory=True,
                num_workers=self.conf.main.num_threads, shuffle=(type == DatasetType.TRAIN)
            )

    def find_instance_mask(self, img, int_object):
        '''
            Args:
                img: [N, C, H, W]
                int_object:
                    Each element in the tensor of shape [N] is an index
                    of the class
        '''
        # use self.maskrcnn
        # run instance segmentation
        # argmax over channels
        # keep only mask over int_object
        # identify which instance to use using MOCA's style
        # return img with seg map over predicted instance only
        pass
