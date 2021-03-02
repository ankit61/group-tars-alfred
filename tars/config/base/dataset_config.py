from pathlib import Path
import os
from tars.base.config import Config


class DatasetConfig(Config):
    def __init__(self):
        super(DatasetConfig, self).__init__()
        self.data_base_dir = os.path.join(Path(__file__).parents[2], 'alfred/data/json_2.1.0')
        self.splits_file = os.path.join(Path(__file__).parents[2], 'alfred/data/splits/oct21.json')

        self.traj_file = 'traj_data.json'
        self.aug_traj_file = 'augmented_traj_data.json'
        self.high_res_img_dir = 'high_res_images'
        self.instance_mask_dir = 'instance_masks'
        self.depth_img_dir = 'depth_images'
