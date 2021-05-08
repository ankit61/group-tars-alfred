import torch
import torch.nn as nn
from tars.base.model import Model
from tars.auxilary_models.embed_and_readout import EmbedAndReadout
from tars.datasets.history_dataset import HistoryType


class ContextModule(Model):
    def __init__(self, num_actions, num_objects, object_na_idx, policy_conf):
        super(ContextModule, self).__init__()

        self.padding_action_idx = num_actions

        self.action_embed_and_readout = EmbedAndReadout.load_from_checkpoint(
            policy_conf.action_readout_path,
            dict_size=num_actions + 1, # +1 for padding
            embed_dim=policy_conf.action_emb_dim,
            out_dim=policy_conf.action_hist_emb_dim,
            padding_idx=self.padding_action_idx,
            history_max_len=policy_conf.past_actions_len,
            policy_conf=policy_conf
        )

        self.int_object_embed_and_readout = EmbedAndReadout.load_from_checkpoint(
            policy_conf.int_object_readout_path,
            dict_size=num_objects, # object_na already included in num_objects
            embed_dim=policy_conf.object_emb_dim,
            out_dim=policy_conf.int_hist_emb_dim,
            padding_idx=object_na_idx,
            history_max_len=policy_conf.past_objects_len,
            policy_conf=policy_conf
        )

        self.context_mixer = nn.Linear(
            policy_conf.action_hist_emb_dim + policy_conf.int_hist_emb_dim + policy_conf.inst_hidden_size + policy_conf.goal_hidden_size,
            policy_conf.context_size
        )


    def forward(self, past_actions, past_objects, inst_lstm_cell, goal_lstm_cell):
        action_readout = self.action_embed_and_readout.forward(past_actions)
        int_objects_readout = self.int_object_embed_and_readout.forward(past_objects)

        explicit_context = torch.cat((action_readout, int_objects_readout), dim=1)
        implicit_context = torch.cat((inst_lstm_cell, goal_lstm_cell), dim=1)

        concated = torch.cat((explicit_context, implicit_context), 1)
        context = self.context_mixer(concated)

        return context
