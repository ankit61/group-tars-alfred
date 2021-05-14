import torch
import torch.nn as nn
from tars.base.model import Model
from tars.auxilary_models.context_emb_model import ContextEmbeddingModel
from tars.config.envs.alfred_env_config import AlfredEnvConfig


class ActionModule(Model):
    def __init__(self, action_emb, obj_emb, conf):
        super(ActionModule, self).__init__()
        self.num_actions = len(AlfredEnvConfig.actions)
        self.num_objects = obj_emb.num_embeddings
        self.remove_goal_lstm = conf.remove_goal_lstm
        self.remove_context = conf.remove_context
        self.activation = getattr(nn, conf.activation)()

        self.action_emb = action_emb
        self.obj_emb = obj_emb

        self.context_emb_model = ContextEmbeddingModel(conf.context_emb_model_name_or_path)

        context_vision_features = (0 if self.remove_context else conf.context_size) + conf.vision_features_size
        self.multi_attn_insts = nn.MultiheadAttention(
                                    embed_dim=context_vision_features,
                                    num_heads=conf.action_attn_heads,
                                    kdim=self.context_emb_model.hidden_size,
                                    vdim=self.context_emb_model.hidden_size
                                )

        if not self.remove_goal_lstm:
            self.multi_attn_goal = nn.MultiheadAttention(
                                    embed_dim=conf.context_size,
                                    num_heads=conf.action_attn_heads,
                                    kdim=self.context_emb_model.hidden_size,
                                    vdim=self.context_emb_model.hidden_size
                                )

            self.goal_lstm = nn.LSTMCell(
                        conf.context_size,
                        conf.goal_hidden_size
                    )

        self.inst_lstm_dropout = nn.Dropout(conf.inst_lstm_dropout)

        self.inst_lstm = nn.LSTMCell(
                            2 * context_vision_features +\
                            (0 if self.remove_goal_lstm else conf.goal_hidden_size),
                            conf.inst_hidden_size
                        )

        self.predictor_fc = nn.Linear(
                                conf.inst_hidden_size,
                                self.num_actions + self.num_objects
                            )
        nn.init.xavier_uniform_(self.predictor_fc.weight)

        self.inst_attn_ln = nn.LayerNorm([context_vision_features])
        self.goal_attn_ln = nn.LayerNorm([conf.context_size])
        self.inst_lstm_ln = nn.LayerNorm([conf.inst_hidden_size])

    def forward(
        self, goal_inst, low_insts, vision_features,
        context, inst_hidden_cell=None, goal_hidden_cell=None
    ):
        with torch.no_grad():
            if not self.remove_goal_lstm:
                goal_embs = self.context_emb_model(goal_inst)
            insts_embs = self.context_emb_model(low_insts)

        # inst LSTM
        if self.remove_context:
            context_vision = vision_features
        else:
            context_vision = torch.cat((context, vision_features), dim=1)

        insts_attended, _ = self.multi_attn_insts(
                            query=context_vision.unsqueeze(0), key=insts_embs,
                            value=insts_embs, need_weights=False
                        )
        insts_attended = self.activation(
                            self.inst_attn_ln(insts_attended.squeeze(0))
                        )

        if self.remove_goal_lstm:
            goal_cell = torch.zeros(context_vision.shape[0], 0, device=context_vision.device)
        elif goal_hidden_cell is None:
            goal_cell = torch.zeros(context_vision.shape[0], self.goal_lstm.hidden_size, device=context_vision.device)
        else:
            goal_cell = goal_hidden_cell[1]

        inst_lstm_in = torch.cat((insts_attended, context_vision, goal_cell), dim=1)
        inst_hidden_cell = self.inst_lstm(inst_lstm_in, inst_hidden_cell)

        inst_lstm_out = self.inst_lstm_ln(inst_hidden_cell[0])
        action_obj = self.predictor_fc(self.inst_lstm_dropout(inst_lstm_out))

        action = action_obj[:, :self.num_actions]
        obj = action_obj[:, self.num_actions:]

        # goal LSTM
        if not self.remove_goal_lstm:
            goal_attended, _ = self.multi_attn_goal(
                                query=context.unsqueeze(0),
                                key=goal_embs, value=goal_embs,
                                need_weights=False
                            )
            goal_attended = self.activation(
                                self.goal_attn_ln(
                                    goal_attended.squeeze(0)
                                )
                            )

            goal_hidden_cell = self.goal_lstm(goal_attended, goal_hidden_cell)
        else:
            goal_hidden_cell = None, None

        return action, obj, inst_hidden_cell, goal_hidden_cell
