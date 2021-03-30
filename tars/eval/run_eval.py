import argparse
import torch
import os
import json
import git
from pprint import pprint
from tars.base import evaluator
from tars.alfred.env.thor_env import ThorEnv
from importlib import import_module


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--data', type=str, default="alfred/data/json_feat_2.1.0")
    parser.add_argument('--eval_split', type=str, default="valid_seen")
    parser.add_argument('--policy_name', type=str, default="baseline_policy")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--gpu', dest='gpu', action='store_true')

    # eval params
    parser.add_argument('--max_steps', type=int, default=1000, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')

    # debug
    parser.add_argument('--fast_epoch', dest='fast_epoch', action='store_true')

    # parse arguments
    args = parser.parse_args()

    # load splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint({k: len(v) for k, v in splits.items()})

    files = splits[args.eval_split]

    # debugging: fast epoch
    if args.fast_epoch:
        files = files[:2]

    # set up evaluator
    policy_module = import_module("tars.policies." + args.policy_name)
    if args.policy_name == "baseline_policy":
        policy = policy_module.BaselinePolicy(args.model_path)

    evaluator = evaluator.Evaluator(args, policy)

    thor_env = ThorEnv()

    # run evaluation
    for (i, task) in enumerate(files):
        print("Evaluating: {} ({})".format(task['task'], task['repeat_idx']))
        print("Number of trajectories left: {}".format(len(files) - i - 1))
        json_file = "/".join([args.data, args.eval_split, task['task'], "traj_data.json"])
        evaluator.evaluate(thor_env, json_file, task['repeat_idx'])

    # save results
    evaluator.save_results()
