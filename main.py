import argparse
import json
from trainers import Trainer
from evaluators import Evaluator
from visualizers import Visualizer
import utils
import os
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
    parser.add_argument('--cfg_path', default='configs/train.json', type=str,
                        help='path to config file')
    parser.add_argument('--data_root', default='../data', type=str,
                        help='path to dataset')
    parser.add_argument('--alg', default='clean', type=str,
                        help='Algorithm to train | Clean / Adv')
    parser.add_argument('--save_path', default='results', type=str,
                        help='path to save file')
    parser.add_argument('--attack_steps', default=10, type=int,
                        help='number of attack iterations (PGD-n)')
    parser.add_argument('--mode', default=None,
                        help='mode to use| Can be train, eval, vis')
    parser.add_argument('--restore', default=None,
                        help='path to restore')
    args = parser.parse_args()
    return args


def main(args):

    # Read configs
    with open(args.cfg_path, "r") as fp:
        configs = json.load(fp)

    # Update the configs based on command line args
    arg_dict = vars(args)
    for key in arg_dict:
        if key in configs:
            if arg_dict[key] is not None:
                configs[key] = arg_dict[key]
    configs = utils.ConfigMapper(configs)

    configs.attack_eps = float(configs.attack_eps) / 255
    configs.attack_lr = float(configs.attack_lr) / 255

    configs.save_path = os.path.join(configs.save_path, configs.mode, configs.alg)
    pathlib.Path(configs.save_path).mkdir(parents=True, exist_ok=True)

    if configs.mode == 'train':
        trainer = Trainer(configs)
        trainer.train()
    elif configs.mode == 'eval':
        evaluator = Evaluator(configs)
        evaluator.eval()
    elif configs.mode == 'vis':
        visualizer = Visualizer(configs)
        visualizer.visualize()
    else:
        raise ValueError('mode should be train, eval or vis')


if __name__ == '__main__':
    args = parse_args()
    main(args)

