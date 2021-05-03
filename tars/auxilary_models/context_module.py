import torch
import torch.nn as nn
from tars.base.model import Model
from tars.auxilary_models.embed_and_readout import EmbedAndReadout


class ContextModule(Model):
    def __init__(self, num_actions, num_objects, object_na_idx, conf):
        super(ContextModule, self).__init__()

        self.padding_action_idx = num_actions

        self.action_embed_and_readout = EmbedAndReadout(
            dict_size=num_actions + 1, # +1 for padding
            embed_dim=conf.action_emb_dim,
            out_dim=conf.action_hist_emb_dim,
            padding_idx=self.padding_action_idx,
            max_len=conf.past_actions_len,
            conf=conf
        )

        self.int_object_embed_and_readout = EmbedAndReadout(
            dict_size=num_objects, # object_na already included in num_objects
            embed_dim=conf.object_emb_dim,
            out_dim=conf.int_hist_emb_dim,
            padding_idx=object_na_idx,
            max_len=conf.past_objects_len,
            conf=conf
        )

        self.context_mixer = nn.Linear(
            conf.action_hist_emb_dim + conf.int_hist_emb_dim + conf.inst_hidden_size + conf.goal_hidden_size,
            conf.context_size
        )


    def forward(self, past_actions, past_objects, inst_lstm_cell, goal_lstm_cell):
        action_readout = self.action_embed_and_readout.forward(past_actions)
        int_objects_readout = self.int_object_embed_and_readout.forward(past_objects)

        explicit_context = torch.cat((action_readout, int_objects_readout), dim=1)
        implicit_context = torch.cat((inst_lstm_cell, goal_lstm_cell), dim=1)

        concated = torch.cat((explicit_context, implicit_context), 1)
        context = self.context_mixer(concated)

        return context
