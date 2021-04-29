import itertools
import torch
import torch.nn as nn
from torchvision.models import resnet18
from tars.base.model import Model
from tars.auxilary_models.mask_rcnn import MaskRCNN
from tars.auxilary_models.segmentation_model import SegmentationModel
from tars.auxilary_models.readout_transformer import ReadoutTransformer
from tars.config.base.dataset_config import DatasetConfig
from tars.base.policy import Policy


class ContextModule(Model):
    def __init__(self, num_actions, num_objects, object_na_idx, conf):
        super(ContextModule, self).__init__()
        self.action_emb = nn.Embedding(
                            num_actions + 1, # add one padding action
                            conf.action_emb_dim,
                            padding_idx=num_actions
                        )

        # FIXME: may want to use pretrained word embeddings for objects
        self.object_emb = nn.Embedding(
                            num_objects, conf.object_emb_dim,
                            padding_idx=object_na_idx
                        )

        self.context_mixer = nn.Linear(
            conf.action_hist_emb_dim + conf.int_hist_emb_dim + conf.inst_hidden_size + conf.goal_hidden_size,
            conf.context_size
        )

        self.action_transformer = ReadoutTransformer(
                                    in_features=conf.action_emb_dim,
                                    out_features=conf.action_hist_emb_dim,
                                    nhead=conf.transformer_num_heads,
                                    num_layers=conf.transformer_num_layers,
                                    max_len=conf.past_actions_len
                                )
        self.past_object_transformer = ReadoutTransformer(
                                        in_features=conf.object_emb_dim,
                                        out_features=conf.int_hist_emb_dim,
                                        nhead=conf.transformer_num_heads,
                                        num_layers=conf.transformer_num_layers,
                                        max_len=conf.past_objects_len
                                    )

    def forward(self, past_actions, past_objects, inst_lstm_cell, goal_lstm_cell):
        actions = self.action_emb(past_actions).permute(1, 0, 2)
        objects = self.object_emb(past_objects).permute(1, 0, 2)

        action_readout = self.action_transformer(actions)
        objects_readout = self.past_object_transformer(objects)

        explicit_context = torch.cat((action_readout, objects_readout), dim=1)
        implicit_context = torch.cat((inst_lstm_cell, goal_lstm_cell), dim=1)

        concated = torch.cat((explicit_context, implicit_context), 1)
        context = self.context_mixer(concated)

        return context


class VisionModule(Model):
    def __init__(self, obj_dim, conf):
        super(VisionModule, self).__init__()
        self.max_img_objects = conf.max_img_objects
        self.raw_vision_features_size = conf.raw_vision_features_size

        self.vision_cnn = resnet18(pretrained=True)
        assert self.vision_cnn.fc.in_features == self.raw_vision_features_size
        self.vision_cnn.fc = nn.Sequential()

        self.objects_transformer = ReadoutTransformer(
                                    in_features=conf.object_emb_dim,
                                    out_features=conf.vision_object_emb_dim,
                                    nhead=conf.transformer_num_heads,
                                    num_layers=conf.transformer_num_layers,
                                    max_len=conf.max_img_objects,
                                    use_pe=False
                                )

        self.object_embedding = obj_dim

        self.use_instance_seg = conf.use_instance_seg
        if self.use_instance_seg:
            self.detection_model = MaskRCNN(model_load_path=conf.detection_model_path)
        else:
            self.detection_model = SegmentationModel(
                                    device=conf.main.device,
                                    model_load_path=conf.detection_model_path
                                )

        self.vision_mixer = nn.Linear(
                            conf.raw_vision_features_size + conf.vision_object_emb_dim,
                            conf.vision_features_size
                        )

    def forward(self, img):
        # can run this on all images and cache on disk to save training/eval time

        raw_vision_features = self.vision_cnn(img)
        assert raw_vision_features.shape == \
            torch.Size([img.shape[0], self.raw_vision_features_size])

        with torch.no_grad():
            if self.use_instance_seg:
                raise NotImplementedError
            else:
                seg_img = self.detection_model(img)
                seg_img = seg_img.argmax(1)
                objects = []

                for i in range(seg_img.shape[0]):  # FIXME: can we avoid loop?
                    img_objects, counts = seg_img[i].unique(return_counts=True)
                    # pick self.max_img_objects largest objects
                    img_objects = img_objects[counts.sort(descending=True)[1]][:self.max_img_objects]
                    padding = torch.tensor(
                                [DatasetConfig.object_na_idx] * \
                                    (self.max_img_objects - len(img_objects))
                            )

                    objects.append(torch.cat((img_objects, padding)))

                objects = torch.stack(objects)

        objects = self.object_embedding(objects.int()).permute(1, 0, 2)
        objects_readout = self.objects_transformer(objects)

        out = self.vision_mixer(
                torch.cat((objects_readout, raw_vision_features), dim=1)
            )

        return out


class ActionModule(Model):
    def __init__(self, action_emb, obj_emb, conf):
        super(ActionModule, self).__init__()
        self.num_actions = action_emb.num_embeddings
        self.num_objects = obj_emb.num_embeddings

        self.action_emb = action_emb
        self.obj_emb = obj_emb

        # FIXME: this should be replaced by context embedding model
        self.context_emb_model = nn.Embedding(100, conf.word_emb_dim)

        self.multi_attn_insts = nn.MultiheadAttention(
                                    embed_dim=conf.context_size + conf.vision_features_size,
                                    num_heads=conf.action_attn_heads,
                                    kdim=conf.word_emb_dim,
                                    vdim=conf.word_emb_dim
                                )

        self.multi_attn_goal = nn.MultiheadAttention(
                                embed_dim=conf.context_size,
                                num_heads=conf.action_attn_heads,
                                kdim=conf.word_emb_dim,
                                vdim=conf.word_emb_dim
                            )

        self.inst_lstm = nn.LSTMCell(
                            2 * (conf.context_size + conf.vision_features_size),
                            conf.inst_hidden_size
                        )

        self.goal_lstm = nn.LSTMCell(
                            conf.context_size + conf.action_emb_dim + conf.object_emb_dim,
                            conf.goal_hidden_size
                        )

        self.predictor_fc = nn.Linear(
                                conf.inst_hidden_size,
                                self.num_actions + self.num_objects
                            )

    def forward(
        self, goal_inst, low_insts, vision_features,
        context, inst_hidden_cell=None, goal_hidden_cell=None
    ):
        goal_embs = self.context_emb_model(goal_inst)
        insts_embs = self.context_emb_model(low_insts)

        # inst LSTM
        context_vision = torch.cat((context, vision_features), dim=1)
        insts_attended, inst_attn_wts = self.multi_attn_insts(
                                query=context_vision.unsqueeze(0), key=insts_embs,
                                value=insts_embs, need_weights=False
                            )
        insts_attended = insts_attended.squeeze(0)

        inst_lstm_in = torch.cat((insts_attended, context_vision), dim=1)
        inst_hidden_cell = self.inst_lstm(inst_lstm_in, inst_hidden_cell)

        action_obj = self.predictor_fc(inst_hidden_cell[0])

        # goal LSTM
        goal_attended, goal_attn_wts = self.multi_attn_goal(
                    query=context.unsqueeze(0), key=goal_embs,
                    value=goal_embs, need_weights=False
                )
        goal_attended = goal_attended.squeeze(0)

        action = action_obj[:, :self.num_actions]
        obj = action_obj[:, self.num_actions:]

        action_emb = self.action_emb(action.argmax(1))
        obj_emb = self.obj_emb(obj.argmax(1))

        goal_lstm_in = torch.cat((goal_attended, action_emb, obj_emb), dim=1)
        goal_hidden_cell = self.goal_lstm(goal_lstm_in, goal_hidden_cell)

        return action, obj, inst_hidden_cell, goal_hidden_cell


class TarsPolicy(Policy):
    def __init__(self):
        super(TarsPolicy, self).__init__()

        self.num_objects = len(DatasetConfig.objects_list)
        self.na_object_idx = DatasetConfig.object_na_idx

        self.context_module = ContextModule(
                                self.num_actions, self.num_objects,
                                self.na_object_idx, self.conf
                            )
        self.vision_module = VisionModule(self.context_module.object_emb, self.conf)
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
                            self.na_object_idx
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
            if int_obj_idxs[i] != self.na_object_idx:
                self.past_objects[i, next(self.objects_itr[i])] = int_obj_idxs[i]

        int_mask = self.find_instance_mask(img, int_object) if self.conf.use_mask else None

        return action, int_mask, int_object

    def find_instance_mask(self, img, int_object):
        # run instance segmentation
        # argmax over channels
        # keep only mask over int_object
        # identify which instance to use using MOCA's style
        # return img with seg map over predicted instance only
        pass
