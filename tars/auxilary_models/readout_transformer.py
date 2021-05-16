import math
import torch
import torch.nn as nn
from tars.base.model import Model


class PositionalEncoding(Model):
    # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, max_len, dropout=0.1):
        # FIXME: Try embeddings
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        x = x + self.pe.weight[:x.size(0), :].unsqueeze(1)
        return self.dropout(x)


class ReadoutTransformer(Model):
    def __init__(self, in_features, out_features, nhead, num_layers, max_len, use_pe=True):
        super(ReadoutTransformer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nhead = nhead
        self.max_len = max_len
        self.use_pe = use_pe

        self.position_encoding = PositionalEncoding(
                                    in_features, max_len=max_len + 1 # +1 for readout
                                ) if self.use_pe else nn.Sequential()

        self.readout_token = nn.Parameter(torch.rand(in_features), requires_grad=True)
        transformer_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=nhead)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        '''
            Args:
                x: [sequence length, batch size, feature dim] tensor
        '''
        r_token = self.readout_token.repeat(x.shape[1]).reshape(x.shape[1:])
        readable_x = torch.cat((r_token.unsqueeze(0), x), dim=0)

        readable_x = self.position_encoding(readable_x)

        out = self.transformer(readable_x)

        return self.linear(out[0]) # return output corresponding to readout only
