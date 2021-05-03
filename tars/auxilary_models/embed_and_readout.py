import torch
import torch.nn as nn
import pdb
from pytorch_lightning.callbacks import ModelCheckpoint
from tars.base.model import Model
from tars.auxilary_models.readout_transformer import ReadoutTransformer
from tars.datasets.history_dataset import HistoryType, HistoryDataset
from torch.utils.data.dataloader import DataLoader
from tars.config.main_config import MainConfig
from typing import Union


class EmbedAndReadout(Model):
    def __init__(self, dict_size, embed_dim, out_dim, padding_idx, history_max_len, conf, use_pe=True, pretrain_type: Union[str, HistoryType] = None):
        super(EmbedAndReadout, self).__init__()

        self.history_max_len = history_max_len
        self.pretrain_type = pretrain_type if isinstance(type, HistoryType) else HistoryType(pretrain_type)
        self.conf = conf

        self.embed = nn.Embedding(
            dict_size + 1, # +1 for SOS token for pretraining
            embed_dim,
            padding_idx=padding_idx,
        )

        self.readout_transformer = ReadoutTransformer(
            in_features=embed_dim,
            out_features=out_dim,
            nhead=conf.transformer_num_heads,
            num_layers=conf.transformer_num_layers,
            max_len=history_max_len,
            use_pe=use_pe
        )

        if self.pretrain_type:
            self.decoder = PretrainingDecoder(self.embed, out_dim)
            self.sos_token = dict_size
    

    def forward(self, items):
        items_embed = self.embed(items).permute(1, 0, 2)
        readout = self.readout_transformer(items_embed)
        return readout


# ======================== PRETRAINING CODE ========================

    def update_item_history(self, item_history, new_items):
        batch_size = new_items.shape[0]
        if item_history == None:
            new_item_history = new_items.reshape(-1, 1)
        elif item_history.shape[1] < self.history_max_len:
            new_item_history = torch.cat((item_history[:batch_size], new_items.reshape(-1, 1)), dim=1)
        else:
            new_item_history = torch.cat((item_history[:batch_size, 1:], new_items.reshape(-1, 1)), dim=1)

        return new_item_history


    def shared_step(self, batch):
        loss_fct = nn.CrossEntropyLoss()
        item_history = None
        mini_steps = 0
        loss = 0
        for mini_batch, batch_size in HistoryDataset.mini_batches(batch):
            item_history = self.update_item_history(item_history, mini_batch)
            decoder_hidden = self.forward(item_history)
            decoder_cell = torch.zeros_like(decoder_hidden)
            decoder_input = torch.tensor([self.sos_token]).repeat(batch_size)
            
            seq_len = item_history.shape[1]
            mini_batch_loss = 0

            for di in range(seq_len):
                decoder_logits, decoder_hidden, decoder_cell = self.decoder.forward(decoder_input, decoder_hidden, decoder_cell)
                _, top_item = decoder_logits.topk(1)
                decoder_input = top_item.reshape(-1).detach()
                mini_batch_loss += loss_fct(decoder_logits, item_history[:, di])
            
            loss += (mini_batch_loss / seq_len)
            mini_steps += 1
        
        if mini_steps > 1:
            loss = loss / mini_steps
        return loss


    def training_step(self, train_batch, batch_idx):
        '''
            Args:
                train_batch: tensor of shape [N, E] where N = batch size and E = longest item sequence in batch
        '''
        loss = self.shared_step(train_batch)
        self.log_dict({'train_loss': loss})
        return loss


    def validation_step(self, val_batch, batch_idx, dataloader_idx):
        loss = self.shared_step(val_batch)
        if dataloader_idx == 0:
            loss_key = 'val_seen_loss'
        else:
            loss_key = 'val_unseen_loss'

        self.log_dict({loss_key: loss})


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    
    def configure_callbacks(self):
        return [
            ModelCheckpoint(monitor='val_seen_loss')
        ]


    def shared_dataloader(self, type):
        dataset = HistoryDataset(type, self.pretrain_type)
        return DataLoader(
            dataset,
            batch_size=4,
            collate_fn=dataset.collate,
            num_workers=1
        )


    def train_dataloader(self):
        return self.shared_dataloader('train')


    def val_dataloader(self):
        return [
            self.shared_dataloader('valid_seen'),
            self.shared_dataloader('valid_unseen')
        ]

        

class PretrainingDecoder(nn.Module):
    def __init__(self, embed, hidden_dim):
        super(PretrainingDecoder, self).__init__()
        self.embed = embed
        self.lstm = nn.LSTMCell(embed.embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, embed.num_embeddings)


    def forward(self, input_idx, hidden, cell):
        input_emb = self.embed(input_idx)
        next_hidden, next_cell = self.lstm.forward(input_emb, (hidden, cell))
        logits = self.out.forward(next_hidden)
        return logits, next_hidden, next_cell

