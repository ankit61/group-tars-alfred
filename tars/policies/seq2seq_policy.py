from argparse import Namespace
from collections import defaultdict
import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from tars.base.policy import Policy
from tars.moca.models.model.seq2seq_im_mask import Module as MOCA
from tars.moca.models.nn.resnet import Resnet
from tars.config.envs.alfred_env_config import AlfredEnvConfig


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

    def reset(self):
        self.model.reset()

    def remap_actions(self, old_pred):
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

    def get_img_features(self, img):
        return self.resnet.resnet_model.extract(img).unsqueeze(0)

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
        raise NotImplementedError
