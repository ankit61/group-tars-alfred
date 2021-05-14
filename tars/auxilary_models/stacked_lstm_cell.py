import torch.nn as nn
from tars.base.model import Model


class StackedLSTMCell(Model):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StackedLSTMCell, self).__init__()
        self.num_layers = num_layers
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(
                input_size=(input_size if i == 0 else hidden_size),
                hidden_size=hidden_size
            ) for i in range(num_layers)
        ])

    def forward(self, x, past_hiddens_cells=None):
        # FIXME: add skip connections
        if past_hiddens_cells is None:
            past_hiddens_cells = [None] * self.num_layers
        hiddens_cells = []
        cur_in = x
        for i in range(len(self.lstm_cells)):
            out = self.lstm_cells[i](cur_in, past_hiddens_cells[i])
            cur_in = out[0]
            hiddens_cells.append(out)

        return hiddens_cells
