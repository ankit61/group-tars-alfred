import os
from tars.base.config import Config
from tars.config.main_config import MainConfig


class DatasetConfig(Config):
    data_base_dir = os.path.join(MainConfig.alfred_dir, 'data/json_2.1.0')
    splits_file = os.path.join(MainConfig.alfred_dir, 'data/splits/oct21.json')

    traj_file = 'traj_data.json'
    aug_traj_file = 'augmented_traj_data.json'
    high_res_img_dir = 'high_res_images'
    instance_mask_dir = 'instance_masks'
    depth_img_dir = 'depth_images'

    start_idx = 0 # start dataset from this index, inclusive
    end_idx = 0 # end dataset at this index, exclusive
