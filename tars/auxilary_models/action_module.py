import torch
import torch.nn as nn
from tars.base.model import Model
from tars.auxilary_models import StackedLSTMCell
from tars.config.envs.alfred_env_config import AlfredEnvConfig


class ActionModule(Model):
    def __init__(self, action_emb, obj_emb, context_emb_size, conf):
        super(ActionModule, self).__init__()
        self.num_actions = len(AlfredEnvConfig.actions)
        self.num_objects = obj_emb.num_embeddings
        self.remove_goal_lstm = conf.remove_goal_lstm
        self.remove_context = conf.remove_context
        self.activation = getattr(nn, conf.activation)()

        self.action_emb = action_emb
        self.obj_emb = obj_emb

        context_vision_features = (0 if self.remove_context else conf.context_size) + conf.vision_features_size
        self.multi_attn_insts = nn.MultiheadAttention(
                                    embed_dim=context_vision_features,
                                    num_heads=conf.action_attn_heads,
                                    kdim=context_emb_size,
                                    vdim=context_emb_size
                                )
        conf.initialize_weights(self.multi_attn_insts)

        if not self.remove_goal_lstm:
            self.multi_attn_goal = nn.MultiheadAttention(
                                    embed_dim=conf.context_size,
                                    num_heads=conf.action_attn_heads,
                                    kdim=context_emb_size,
                                    vdim=context_emb_size
                                )
            conf.initialize_weights(self.multi_attn_goal)

            self.goal_lstm = StackedLSTMCell(
                        conf.context_size + self.action_emb.embedding_dim,
                        conf.goal_hidden_size, num_layers=conf.num_goal_lstm_layers
                    )
            conf.initialize_weights(self.goal_lstm)

        self.inst_lstm_dropout = nn.Dropout(conf.inst_lstm_dropout)

        self.action_lstm = StackedLSTMCell(
                            2 * context_vision_features +\
                            (0 if self.remove_context else conf.action_hist_emb_dim) +\
                            (0 if self.remove_goal_lstm else conf.goal_hidden_size),
                            conf.inst_hidden_size,
                            num_layers=conf.num_inst_lstm_layers
                        )
        conf.initialize_weights(self.action_lstm)

        self.obj_lstm = StackedLSTMCell(
                            self.action_lstm.input_size + self.action_emb.embedding_dim,
                            self.action_lstm.hidden_size,
                            num_layers=conf.num_inst_lstm_layers
                        )
        conf.initialize_weights(self.obj_lstm)

        self.action_predictor = nn.Linear(
                                conf.inst_hidden_size * conf.num_inst_lstm_layers,
                                self.num_actions
                            )
        conf.initialize_weights(self.action_predictor)

        self.obj_predictor = nn.Linear(
                                conf.inst_hidden_size * conf.num_inst_lstm_layers,
                                self.num_objects
                            )
        conf.initialize_weights(self.obj_predictor)

        self.inst_attn_ln = nn.LayerNorm([context_vision_features])
        self.goal_attn_ln = nn.LayerNorm([conf.context_size])
        self.action_lstm_ln = nn.LayerNorm([self.action_predictor.in_features])
        self.obj_lstm_ln = nn.LayerNorm([self.obj_predictor.in_features])

    def forward(
        self, goal_embs, insts_embs, vision_features,
        context, action_readout, last_action_hidden_cell, last_obj_hidden_cell,
        last_goal_hidden_cell
    ):
        # inst LSTM
        if self.remove_context:
            context_vision = vision_features
            action_readout = torch.zeros(vision_features.shape[0], 0)
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
            last_goal_cell = torch.zeros(context_vision.shape[0], 0, device=context_vision.device)
        else:
            last_goal_cell = last_goal_hidden_cell[-1][1]

        action_lstm_in = torch.cat((insts_attended, context_vision, action_readout, last_goal_cell), dim=1)

        action, action_hidden_cell = self.prediction_lstm_forward(
                                        lstm_in=action_lstm_in,
                                        lstm_cell=self.action_lstm,
                                        lstm_ln=self.action_lstm_ln,
                                        predictor=self.action_predictor,
                                        last_hidden_cell=last_action_hidden_cell
                                    )

        action_emb = self.action_emb(action.argmax(1))
        obj_lstm_in = torch.cat([action_lstm_in, action_emb], dim=1)
        obj, obj_hidden_cell = self.prediction_lstm_forward(
                                    lstm_in=obj_lstm_in,
                                    lstm_cell=self.obj_lstm,
                                    lstm_ln=self.obj_lstm_ln,
                                    predictor=self.obj_predictor,
                                    last_hidden_cell=last_obj_hidden_cell
                                )

        # goal LSTM
        if not self.remove_goal_lstm:
            goal_hidden_cell = self.goal_lstm_forward(
                                goal_embs,
                                context,
                                last_goal_hidden_cell,
                                action_emb
                            )
        else:
            goal_hidden_cell = None

        return action, obj, action_hidden_cell, obj_hidden_cell, goal_hidden_cell

    def prediction_lstm_forward(self, lstm_in, lstm_cell, lstm_ln, predictor, last_hidden_cell):
        hidden_cell = lstm_cell(lstm_in, last_hidden_cell)

        lstms_outs = torch.cat(
                        [hidden_cell[i][0] for i in range(len(hidden_cell))],
                        dim=1
                    )

        pred_in = self.inst_lstm_dropout(lstm_ln(lstms_outs))
        return predictor(pred_in), hidden_cell

    def goal_lstm_forward(self, goal_embs, context, action_emb, last_goal_hidden_cell):
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

        goal_lstm_in = torch.cat([goal_attended, action_emb], dim=1)
        goal_hidden_cell = self.goal_lstm(goal_lstm_in, last_goal_hidden_cell)

        return goal_hidden_cell
