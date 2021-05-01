import torch
import torch.nn as nn
from tars.base.model import Model
from tars.auxilary_models.readout_transformer import ReadoutTransformer


class EmbedAndReadout(Model):
    def __init__(self, dict_size, embed_dim, out_dim, padding_idx, past_item_len, conf, pretraining=False):
        super(EmbedAndReadout, self).__init__()

        self.seq_len = past_item_len
        self.padding_idx = padding_idx
        self.pretraining = pretraining
        self.conf = conf

        self.embed = nn.Embedding(
                        dict_size,
                        embed_dim,
                        padding_idx=padding_idx
                    )

        self.readout_transformer = ReadoutTransformer(
                                    in_features=embed_dim,
                                    out_features=out_dim,
                                    nhead=conf.transformer_num_heads,
                                    num_layers=conf.transformer_num_layers,
                                    max_len=past_item_len
                                )

        if self.pretraining:
            self.decoder = PreTrainingDecoder(self.embed, out_dim)
    

    def forward(self, past_items):
        items = self.embed(past_items).permute(1, 0, 2)
        readout = self.readout_transformer(items)
        return readout


    def shared_step(self, batch, seq_lens):
        unfold = nn.Unfold(kernel_size=(1, self.past_item_len))
        loss_fct = nn.CrossEntropyLoss()
        batch_size = train_batch.shape
        batch_loss = 0
        for i in range(batch_size):
            full_item_seq = train_batch[i]
            item_indices = torch.arange(seq_lens[i]).reshape(1, 1, 1, seq_lens[i]).float()
            item_indices_batch = unfold(item_indices).transpose(1, 2).long()
            past_item_seq_batch = full_item_seq[item_indices_batch][0]

            decoder_batch_size = past_item_seq_batch.shape[0]
            decoder_hidden = self.forward(past_k_items)
            decoder_cell = torch.zeros_like(decoder_hidden)
            decoder_input = torch.tensor([[self.conf.sos_token]]).repeat(decoder_batch_size)
            loss = 0
            for di in range(self.past_item_len):
                decoder_logits, decoder_hidden, decoder_cell = self.decoder.forward(decoder_input, decoder_hidden, decoder_cell)
                _, top_item = decoder_logits.topk(1)
                decoder_input = top_item.detach()
                loss += loss_fct(decoder_logits, past_k_items[:, di])
            
            batch_loss += (loss / self.past_item_len)

        return (batch_loss / batch_size)


    def training_step(self, train_batch, seq_lens):
        '''
            Args:
                train_batch: tensor of shape [N, E] where N = batch size and E = longest item sequence in batch
        '''
        return self.shared_step(train_batch, seq_lens)


    def validation_step(self, val_batch, seq_lens):
        return self.shared_step(val_batch, seq_lens)



class PreTrainingDecoder(nn.Module):
    def __init__(self, embed, hidden_dim):
        super(PreTrainingDecoder, self).__init__()
        self.embed = embed
        self.lstm = nn.LSTMCell(embed.embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, embed.num_embeddings)

    def forward(self, input_idx, hidden, cell):
        input_emb = self.embed(input_idx)
        next_hidden, next_cell = self.lstm.forward(input_emb, (hidden, cell))
        logits = self.out.forward(next_hidden)
        return logits, next_hidden, next_cell

