from tars.base.config import Config
from tars.config.base.dataset_config import DatasetConfig


class TarsPolicyConfig(Config):
    use_mask = False
    batch_size = 32

    # feature sizes
    context_size = 512
    vision_features_size = 512
    raw_vision_features_size = 512

    # history
    past_actions_len = 10
    past_objects_len = 10

    # embeddings
    action_emb_dim = 64
    object_emb_dim = 64
    action_hist_emb_dim = context_size // 2
    int_hist_emb_dim = context_size // 2
    word_emb_dim = 128
    vision_object_emb_dim = 128

    # LSTMs
    inst_hidden_size = 256
    goal_hidden_size = 256

    # vision module
    use_instance_seg = False
    detection_model_path = None
    max_img_objects = 10

    # readout transformer
    transformer_num_heads = 8
    transformer_num_layers = 2

    # action module
    action_attn_heads = 4

    # contextual embedding model
    context_emb_model_name_or_path = "albert-base-v2"
