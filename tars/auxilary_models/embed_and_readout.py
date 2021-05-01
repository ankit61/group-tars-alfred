import torch
import torch.nn as nn
from tars.base.model import Model
from tars.auxilary_models.readout_transformer import ReadoutTransformer


class EmbedAndReadout(Model):
    def __init__(self, dict_size, embed_dim, out_dim, padding_idx, max_len, conf):
        super(EmbedAndReadout, self).__init__()

        self.embed = nn.Embedding(
                        dict_size,
                        conf.embed_dim,
                        padding_idx=padding_idx
                    )


        self.readout_transformer = ReadoutTransformer(
                                    in_features=embed_dim,
                                    out_features=out_dim,
                                    nhead=conf.transformer_num_heads,
                                    num_layers=conf.transformer_num_layers,
                                    max_len=max_len
                                )
    

    def forward(self, past_items):
        items = self.embed(past_items).permute(1, 0, 2)
        readout = self.readout_transformer(items)
        return readout
