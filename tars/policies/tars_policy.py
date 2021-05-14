import itertools
import random
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
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
import tars.alfred.gen.constants as constants


class TarsPolicy(Policy):
    def __init__(self):
        super(TarsPolicy, self).__init__()

        self.num_objects = len(DatasetConfig.objects_list)
        self.object_na_idx = DatasetConfig.object_na_idx
        self.teacher_prob = self.conf.teacher_forcing_init

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

        self.inst_lstm_hiddens_cells = [
                                (torch.zeros(
                                    self.conf.batch_size,
                                    self.conf.inst_hidden_size,
                                    device=self.conf.main.device
                                ), ) * 2
                                for _ in range(self.conf.num_inst_lstm_layers)
                            ]

        if not self.conf.remove_goal_lstm:
            self.goal_lstm_hiddens_cells = [
                                    (torch.zeros(
                                        self.conf.batch_size,
                                        self.conf.goal_hidden_size,
                                        device=self.conf.main.device
                                    ), ) * 2
                                    for _ in range(self.conf.num_goal_lstm_layers)
                                ]
        else:
            self.goal_lstm_hiddens_cells = None

    def forward(self, img, goal_inst, low_insts, gt_action=None, gt_int_object=None):
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
                        self.inst_lstm_hiddens_cells[-1][1],
                        self.goal_lstm_hiddens_cells[-1][1]
                    )

        vision_features = self.vision_module(img)

        action, int_object, self.inst_lstm_hiddens_cells, self.goal_lstm_hiddens_cells =\
            self.action_module(
                goal_inst,
                low_insts,
                vision_features,
                context,
                self.inst_lstm_hiddens_cells,
                self.goal_lstm_hiddens_cells,
            )

        self.update_history(action, int_object, gt_action, gt_int_object)

        int_mask = self.find_instance_mask(img, int_object) if self.conf.use_mask else None

        return action, int_mask, int_object

    def update_history(self, action, int_object, gt_action, gt_int_object):
        if random.random() < self.teacher_prob and gt_action is not None and\
            gt_int_object is not None:

            self.past_actions[:, next(self.actions_itr)] = gt_action
            for i in range(gt_int_object.shape[0]):
                if gt_int_object[i] != self.object_na_idx:
                    self.past_objects[i, next(self.objects_itr[i])] = gt_int_object[i]
        else:
            self.past_actions[:, next(self.actions_itr)] = action.argmax(1)
            int_obj_idxs = int_object.argmax(1)
            for i in range(int_obj_idxs.shape[0]):
                if int_obj_idxs[i] != self.object_na_idx:
                    self.past_objects[i, next(self.objects_itr[i])] = int_obj_idxs[i]


    def shared_step(self, batch, test_time):
        # FIXME: Features needed:
        # gradient clipping, truncated BPTT, teacher forcing
        self.reset()

        ac_loss, obj_loss = 0, 0
        ac_seq_len, obj_seq_len = 0, 0
        pred_actions, pred_objects = [], []

        if self.global_step % self.conf.teacher_forcing_step == 0:
            self.teacher_prob *= self.conf.teacher_forcing_curriculum

        for mini_batch, batch_size in ImitationDataset.mini_batches(batch):
            self.trim_history(batch_size)
            # self.past_actions = mini_batch['expert_actions'].repeat_interleave(self.past_actions.shape[1]).reshape(self.past_actions.shape)
            # self.past_objects = mini_batch['expert_int_objects'].repeat_interleave(self.past_objects.shape[1]).reshape(self.past_objects.shape)
            if test_time:
                action, _, int_object = self(
                                        mini_batch['images'],
                                        mini_batch['goal_inst'],
                                        mini_batch['low_insts']
                                    )
            else:
                action, _, int_object = self(
                                        mini_batch['images'],
                                        mini_batch['goal_inst'],
                                        mini_batch['low_insts'],
                                        mini_batch['expert_actions'],
                                        mini_batch['expert_int_objects']
                                    )
            pred_actions.append(action.argmax(1).float().mean().item())
            pred_objects.append(int_object.argmax(1).float().mean().item())
            ac_loss += self.action_loss(action, mini_batch['expert_actions'])
            expert_objs = mini_batch['expert_int_objects']
            object_mask = (expert_objs != self.object_na_idx)
            obj_loss += self.object_loss(int_object[object_mask], expert_objs[object_mask])
            ac_seq_len += batch_size
            obj_seq_len += object_mask.sum()

        # print('Action: ', torch.tensor(pred_actions).unique(), torch.tensor(pred_actions).std())
        # print('Object: ', torch.tensor(pred_objects).unique(), torch.tensor(pred_objects).std())
        ac_loss = ac_loss / max(1e-5, ac_seq_len)
        obj_loss = obj_loss / max(1e-5, obj_seq_len)

        return {
            'loss': ac_loss + obj_loss,
            'action_loss': ac_loss.item(),
            'object_loss': obj_loss.item(),
            'pred_action_std': torch.tensor(pred_actions).std(),
            'pred_object_std': torch.tensor(pred_objects).std()
        }

    def configure_optimizers(self):
        optim = self.conf.get_optim(self.parameters())
        scheduler = self.conf.get_lr_scheduler(optim)
        return (optim if scheduler is None else ([optim], [scheduler]))

    def training_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, test_time=False)
        metrics = {f'train_{k}': metrics[k] for k in metrics}
        self.log_dict(metrics)
        metrics['loss'] = metrics.pop('train_loss')
        return metrics['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx):
        metrics = self.shared_step(batch, test_time=True)
        metrics = {f'val_{k}': metrics[k] for k in metrics}

        self.log_dict(metrics)

    def trim_history(self, batch_size):
        self.past_actions = self.past_actions[:batch_size]
        self.past_objects = self.past_objects[:batch_size]

        self.inst_lstm_hiddens_cells = [
            (
                self.inst_lstm_hiddens_cells[i][0][:batch_size],
                self.inst_lstm_hiddens_cells[i][1][:batch_size]
            )
            for i in range(len(self.inst_lstm_hiddens_cells))
        ]

        if not self.conf.remove_goal_lstm:
            self.goal_lstm_hiddens_cells = [
            (
                self.goal_lstm_hiddens_cells[i][0][:batch_size],
                self.goal_lstm_hiddens_cells[i][1][:batch_size]
            )
            for i in range(len(self.goal_lstm_hiddens_cells))
        ]

    ### data stuff

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


    def get_trainer_kwargs(self):
        trainer_kwargs = self.conf.main.default_trainer_args
        trainer_kwargs['accumulate_grad_batches'] = 1
        return trainer_kwargs

    def find_instance_mask(self, imgs, int_objects):
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
        prev_class = 0
        prev_center = torch.zeros(2)

        predicted_masks = [] # size N list containing mask

        for img, int_object in zip(imgs, int_objects):

            pred_class = int_object

            # mask generation
            with torch.no_grad():
                out = self.maskrcnn([to_tensor(img).to('cuda' if self.conf.main.use_gpu else 'cpu')])[0]
                for k in out:
                    out[k] = out[k].detach().cpu()

            if sum(out['labels'] == pred_class) == 0:
                mask = np.zeros((constants.SCREEN_WIDTH,constants.SCREEN_HEIGHT))

            else:
                masks = out['masks'][out['labels'] == pred_class].detach().cpu()
                scores = out['scores'][out['labels'] == pred_class].detach().cpu()

                # Instance selection based on the minimum distance between the prev. and cur. instance of a same class.
                if prev_class != pred_class:
                    scores, indices = scores.sort(descending=True)
                    masks = masks[indices]
                    prev_class = pred_class
                    prev_center = masks[0].squeeze(dim=0).nonzero().double().mean(dim=0)
                else:
                    cur_centers = torch.stack([m.nonzero().double().mean(dim=0) for m in masks.squeeze(dim=1)])
                    distances = ((cur_centers - prev_center) ** 2).sum(dim=1)
                    distances, indices = distances.sort()
                    masks = masks[indices]
                    prev_center = cur_centers[0]

                mask = np.squeeze(masks[0].numpy(), axis=0)
            predicted_masks.append(mask)

        return predicted_masks
