import torch
import torch.nn as nn
from tars.base.model import Model
from tars.auxilary_models.readout_transformer import ReadoutTransformer


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
