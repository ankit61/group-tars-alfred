from collections import defaultdict
import itertools
import numpy as np
import torch
from tars.policies import Seq2SeqPolicy
from tars.alfred.models.model.seq2seq_im_mask import Module as BaselineModel


class BaselinePolicy(Seq2SeqPolicy):
    def __init__(self, model_load_path=None):
        model_load_path = self.conf.saved_model_path if model_load_path is None else model_load_path
        print("Loading model from {}".format(model_load_path))
        model, _ = BaselineModel.load(model_load_path, self.conf.main.device)
        super().__init__(model)

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

        with torch.no_grad():
            feat = defaultdict(list)

            # get img features
            feat['frames'] = self.featurize_img(img)

            # get language embeddings
            merged_lang = merge_insts(goal_inst, low_insts)
            feat['lang_goal_instr'] = self.pack_lang(merged_lang)

            # run model
            m_out = self.model.step(feat)
            action, mask = m_out['out_action_low'], m_out['out_action_low_mask']

            # FIXME: depends on above assert
            action, mask = self.remap_actions(action.squeeze()), torch.sigmoid(mask.squeeze())

            # FIXME: depends on above assert
            return action.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0)
