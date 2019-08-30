import argparse
import json
from trainers import Trainer
import utils
import os


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
    parser.add_argument('--cfg_path', default='config.json', type=str,
                        help='path to config file')
    parser.add_argument('--data_root', default='../data', type=str,
                        help='path to dataset')
    parser.add_argument('--alg', default='clean', type=str,
                        help='Algorithm to train | Clean / Adv')
    parser.add_argument('--save_path', default='results', type=str,
                        help='path to save file')
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
            configs[key] = arg_dict[key]
    configs = utils.ConfigMapper(configs)

    configs.attack_eps = float(configs.attack_eps) / 255
    configs.attack_lr = float(configs.attack_lr) / 255

    if not os.path.exists(configs.save_path):
        os.mkdir(configs.save_path)
    configs.save_path = os.path.join(configs.save_path, configs.alg)
    if not os.path.exists(configs.save_path):
        os.mkdir(configs.save_path)

    # Create a trainer
    trainer = Trainer(configs)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)

