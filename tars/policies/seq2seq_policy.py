from typing import List, Union
from argparse import Namespace
import numpy as np
import revtok
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchvision import transforms
from tars.base.policy import Policy
from tars.moca.models.nn.resnet import Resnet
from tars.config.envs.alfred_env_config import AlfredEnvConfig
from tars.alfred.gen.utils.py_util import remove_spaces_and_lower


class Seq2SeqPolicy(Policy):
    def __init__(self, seq2seq_model):
        super().__init__()

        self.model = seq2seq_model
        self.model.share_memory()
        self.model.eval()
        self.model.test_mode = True

        args = Namespace()
        args.gpu = self.conf.main.use_gpu
        args.visual_model = 'resnet18'
        self.resnet = Resnet(args, eval=True, share_memory=True, use_conv_feat=True)

        self.action_remapping = {}
        old_vocab = list(map(lambda x: x.split('_')[0], self.model.vocab['action_low'].to_dict()['index2word']))
        actions = set()
        for i in range(len(AlfredEnvConfig.actions)):
            new_action = AlfredEnvConfig.actions.index2word(i)
            if new_action in old_vocab:
                self.action_remapping[old_vocab.index(new_action)] = i
                actions.add(new_action)

        assert len(actions) == self.num_actions

    def reset(self):
        self.model.reset()

    def remap_actions(self, old_pred):
        out = torch.zeros(*(list(old_pred.shape[:-1]) + [self.num_actions]))

        for old_idx, new_idx in self.action_remapping.items():
            out[..., new_idx] = old_pred[..., old_idx]

        return out

    def featurize_img(self, img):
        return self.resnet.resnet_model.extract(img).unsqueeze(0)

    def pack_lang(self, lang):
        pad_seq = pad_sequence(lang, batch_first=True, padding_value=self.model.pad)
        seq_lengths = np.array(list(map(len, lang)))
        embed_seq = self.model.emb_word(pad_seq)
        return pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)

    def forward(self, img, goal_inst, low_insts):
        raise NotImplementedError

    @staticmethod
    def get_img_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def get_text_transforms(self):
        return self.text_transform

    def text_transform(self, langs: Union[List[List[str]], List[str]], is_goal: bool) -> List[List[List]]:
        out = []
        for lang in langs:
            if not isinstance(lang, list):
                lang = [lang]  # if string is passed, treat it as one sentence

            tokenized = [revtok.tokenize(remove_spaces_and_lower(x)) for x in lang]
            if is_goal:
                assert len(tokenized) == 1
                tokenized[0].append(self.conf.goal_literal)
            else:
                tokenized += [[self.conf.stop_literal]]

            out.append([])
            for s in tokenized:
                out[-1].append([self.model.vocab['word'].word2index(w.strip(), train=False) for w in s])
        return out
