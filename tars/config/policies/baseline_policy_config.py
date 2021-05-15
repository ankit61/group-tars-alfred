import os
from tars.config.policies.seq2seq_policy_config import Seq2SeqPolicyConfig
from tars.config.base.dataset_config import DatasetConfig


class BaselinePolicyConfig(Seq2SeqPolicyConfig):
    saved_model_path = os.path.join(DatasetConfig.alfred_dir, 'exp/model:seq2seq_im_mask,name:base30_pm010_sg010_01/best_seen.pth')
