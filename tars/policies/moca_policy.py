from itertools import chain
import numpy as np
import revtok
import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from tars.policies.seq2seq_policy import Seq2SeqPolicy
from tars.moca.models.model.seq2seq_im_mask import Module as MOCA
from tars.moca.gen.utils.py_util import remove_spaces_and_lower


class MocaPolicy(Seq2SeqPolicy):
    def __init__(self, model_load_path=None):
        model_load_path = self.conf.saved_model_path if model_load_path is None else model_load_path

        model, _ = MOCA.load(model_load_path, self.conf.main.device)
        super(MocaPolicy, self).__init__(model)

        self.maskrcnn = maskrcnn_resnet50_fpn(num_classes=self.conf.mask_rcnn_classes)
        self.maskrcnn.eval()
        self.maskrcnn.load_state_dict(torch.load(self.conf.mask_rcnn_path, map_location=self.conf.main.device))
        self.maskrcnn = self.maskrcnn.to(self.conf.main.device)

        self.prev_image = None
        self.prev_action = None

        self.prev_class = 0
        self.prev_center = torch.zeros(2)

    def reset(self):
        self.model.reset()
        self.prev_image = None
        self.prev_action = None

        self.prev_class = 0
        self.prev_center = torch.zeros(2)

    @staticmethod
    def get_img_transforms():
        return transforms.ToTensor()

    def resnet_transform(self, img):
        trfs = Seq2SeqPolicy.get_img_transforms()
        to_pil = transforms.ToPILImage()
        out = [trfs(to_pil(img[i])) for i in range(img.shape[0])]
        return torch.stack(out, 0)

    def forward(self, img, goal_inst, low_insts):
        assert img.shape[0] == 1 # only batch size = 1 supported
        with torch.no_grad():
            normalized_img = self.resnet_transform(img)

            feat = {}
            feat['frames'] = self.featurize_img(normalized_img)

            feat['lang_goal'] = self.pack_lang([torch.tensor(g[0]) for g in goal_inst])
            feat['lang_instr'] = self.pack_lang([torch.tensor(list(chain(*l))) for l in low_insts])

            m_out = self.model.step(feat)
            m_pred = self.model.extract_preds(m_out, None, feat, clean_special_tokens=False)
            m_pred = list(m_pred.values())[0]

            # action prediction
            action = m_pred['action_low']
            if self.prev_image is not None:
                if (self.prev_image == img).all() and self.prev_action == action and action == 'MoveAhead_25':
                    dist_action = m_out['out_action_low'][0][0].detach().cpu()
                    idx_rotateR = self.model.vocab['action_low'].word2index('RotateRight_90')
                    idx_rotateL = self.model.vocab['action_low'].word2index('RotateLeft_90')
                    action = 'RotateLeft_90' if dist_action[idx_rotateL] > dist_action[idx_rotateR] else 'RotateRight_90'

            mask = torch.zeros(*self.int_mask_size)
            if self.model.has_interaction(action):
                class_dist = m_pred['action_low_mask'][0]
                pred_class = np.argmax(class_dist)

                # mask generation
                with torch.no_grad():
                    out = self.maskrcnn([img.to(self.conf.main.device).squeeze(0)])[0] # assumes above assert
                    for k in out:
                        out[k] = out[k].detach().cpu()

                if sum(out['labels'] == pred_class) != 0:
                    masks = out['masks'][out['labels'] == pred_class].detach().cpu()
                    scores = out['scores'][out['labels'] == pred_class].detach().cpu()

                    # Instance selection based on the minimum distance between the prev. and cur. instance of a same class.
                    if self.prev_class != pred_class:
                        scores, indices = scores.sort(descending=True)
                        masks = masks[indices]
                        self.prev_class = pred_class
                        self.prev_center = masks[0].squeeze(dim=0).nonzero().double().mean(dim=0)
                    else:
                        cur_centers = torch.stack([m.nonzero().double().mean(dim=0) for m in masks.squeeze(dim=1)])
                        distances = ((cur_centers - self.prev_center)**2).sum(dim=1)
                        distances, indices = distances.sort()
                        masks = masks[indices]
                        self.prev_center = cur_centers[0]

                    mask = masks[0].squeeze(0)

            self.prev_image = img
            self.prev_action = action

            new_action_idx = self.action_remapping[self.model.vocab['action_low'].word2index(action)]
            action = torch.zeros(1, self.num_actions) # assumes above assert
            action[..., new_action_idx] = 1

            return action, mask.unsqueeze(0).unsqueeze(0)
