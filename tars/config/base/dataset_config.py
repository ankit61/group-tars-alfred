import os
from pathlib import Path
from tars.base.config import Config
from tars.alfred.gen.constants import OBJECTS
from vocab import Vocab


class DatasetConfig(Config):
    alfred_dir = os.path.join(Path(__file__).parents[2], 'alfred/')
    data_base_dir = '/data/json_2.1.0' if os.path.exists('/data/json_2.1.0') else os.path.join(alfred_dir, 'data/json_2.1.0')
    splits_file = os.path.join(alfred_dir, 'data/splits/oct21.json')

    object_na = 0
    objects_list = [object_na] + OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']
    objects_vocab = Vocab(objects_list)
    object_na_idx = objects_vocab.word2index(object_na)

    traj_file = 'traj_data.json'
    aug_traj_file = 'augmented_traj_data.json'
    high_res_img_dir = 'high_res_images'
    instance_mask_dir = 'instance_masks'
    instance_target_dir = 'instance_targets'
    depth_img_dir = 'depth_images'

    start_idx = 0 # start dataset from this index, inclusive
    end_idx = None # end dataset at this index, exclusive
