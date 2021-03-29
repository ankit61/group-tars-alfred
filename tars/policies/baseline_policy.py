from collections import defaultdict
from typing import List, Union
import itertools
from argparse import Namespace
import numpy as np
import revtok
import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tars.base.policy import Policy
from tars.alfred.models.model.seq2seq_im_mask import Module as BaselineModel
from tars.alfred.models.nn.resnet import Resnet
from tars.config.envs.alfred_env_config import AlfredEnvConfig
from tars.alfred.gen.utils.py_util import remove_spaces_and_lower


class BaselinePolicy(Policy):
    def __init__(self, model_load_path=None):
        super().__init__()
        model_load_path = self.conf.saved_model_path if model_load_path is None else model_load_path

        self.model, _ = BaselineModel.load(model_load_path, self.conf.main.device)
        self.model.share_memory()
        self.model.eval()
        self.model.test_mode = True # only to be used for testing

        args = Namespace()
        args.gpu = self.conf.main.use_gpu
        args.visual_model = 'resnet18'
        self.resnet = Resnet(args, eval=True, share_memory=True, use_conv_feat=True)

    def reset(self):
        self.model.reset()

    def forward(self, img, goal_inst, low_insts):
        # use batch size 1 for now because higher num needs testing
        assert img.shape[0] == 1 and len(goal_inst) == 1 and len(low_insts)

        def merge_insts(goal_inst, low_insts):
            assert len(goal_inst) == len(low_insts)
            out = []
            for i in range(len(goal_inst)):
                cur_low_insts = list(itertools.chain(*low_insts[i]))
                out.append(torch.tensor(goal_inst[i][0] + cur_low_insts))

            return out

        def remap_actions(old_pred):
            out = torch.zeros(*(list(old_pred.shape[:-1]) + [self.num_actions]))
            old_vocab = list(map(lambda x: x.split('_')[0], self.model.vocab['action_low'].to_dict()['index2word']))

            actions = set()
            for i in range(len(AlfredEnvConfig.actions)):
                new_action = AlfredEnvConfig.actions.index2word(i)
                if new_action in old_vocab:
                    out[..., i] = old_pred[..., old_vocab.index(new_action)]
                actions.add(new_action)

            assert len(actions) == self.num_actions
            return out

        with torch.no_grad():
            # get img features
            img_features = self.resnet.resnet_model.extract(img)

            # get language embeddings
            merged_lang = merge_insts(goal_inst, low_insts)
            pad_seq = pad_sequence(merged_lang, batch_first=True, padding_value=self.model.pad)
            seq_lengths = np.array(list(map(len, merged_lang)))
            embed_seq = self.model.emb_word(pad_seq)
            packed_lang = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)

            # add everything as feat
            feat = defaultdict(list)
            feat['frames'] = img_features.unsqueeze(0)
            feat['lang_goal_instr'] = packed_lang

            # run model
            m_out = self.model.step(feat)
            action, mask = m_out['out_action_low'], m_out['out_action_low_mask']

            # FIXME: depends on above assert
            action, mask = remap_actions(action.squeeze()), torch.sigmoid(mask.squeeze())

            # FIXME: depends on above assert
            return action.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0)

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
        def transform(langs: Union[List[List[str]], List[str]], is_goal: bool) -> List[List[List]]:
            '''
                Args:
                    langs: list of sentences
                    is_goal: boolean denoting whether passed sentences are goals
                             or low level descs
                Returns:
                    out: langs converted to integer indices as per
                         self.model.vocab['word']
            '''

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

        return transform
