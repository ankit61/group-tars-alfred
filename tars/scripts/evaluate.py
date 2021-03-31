import os
import argparse
from tars.policies import *
from tars.config.base.dataset_config import DatasetConfig
from tars.evaluators.metrics_evaluator import MetricsEvaluator
from tars.base.evaluator import Evaluator
from tars.base.dataset import Dataset, DatasetType


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--split', type=str, default='valid_seen', choices=list(map(lambda x: x.value, DatasetType.__members__.values())))
    parser.add_argument('--lang-idx', type=int, default=0)
    parser.add_argument('--policy', type=str, default='BaselinePolicy')
    parser.add_argument('--evaluator', type=str, default='MetricsEvaluator')
    
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    # get policy and evaluator
    policy = globals()[args.policy]()
    evaluator = globals()[args.evaluator](policy)
    if args.task:
        # evaluate a single task
        json_file = os.path.join(DatasetConfig().data_base_dir, args.split, args.task, DatasetConfig().traj_file)
        evaluator.evaluate(json_file, args.lang_idx)
    else:
        # evaluate an entire split (or part of it depending on the config)
        evaluator.evaluate_split(DatasetType(args.split))


if __name__ == '__main__':
    main()
