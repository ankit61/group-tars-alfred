import os
from pathlib import Path
from tars.config.policies.seq2seq_policy_config import Seq2SeqPolicyConfig


class MocaPolicyConfig(Seq2SeqPolicyConfig):
    moca_dir = os.path.join(Path(__file__).parents[2], 'moca/')
    saved_model_path = os.path.join(moca_dir, 'exp/pretrained/pretrained.pth')
    mask_rcnn_path = '/data/best_models/weight_maskrcnn.pt' if os.path.exists('/data/best_models/weight_maskrcnn.pt') else os.path.join(moca_dir, 'weight_maskrcnn.pt')
    mask_rcnn_classes = 119
