import os
import argparse
from tars.policies import *
from tars.config.base.dataset_config import DatasetConfig
from tars.evaluators.metrics_evaluator import MetricsEvaluator
from tars.base.evaluator import Evaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='valid_seen/pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-13/trial_T20190909_115736_122556/')
    parser.add_argument('--lang-idx', type=int, default=0)
    parser.add_argument('--policy', type=str, default='BaselinePolicy')
    parser.add_argument('--evaluator', type=str, default='MetricsEvaluator')

    args = parser.parse_args()
    args.task = os.path.join(DatasetConfig().data_base_dir, args.task)
    args.json_file = os.path.join(args.task, 'traj_data.json')

    return args

def main():
    args = get_args()
    policy = globals()[args.policy]()
    evaluator = globals()[args.evaluator](policy)
    evaluator.evaluate(args.json_file, args.lang_idx)


if __name__ == '__main__':
    main()
