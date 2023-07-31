#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import argparse
from pathlib import Path
LINK_PROJECT = Path(os.path.abspath(__file__)).parent.parent.parent


def args_parser():
    parser = argparse.ArgumentParser()
    # link project
    parser.add_argument(
        '--link_project', default=f'{LINK_PROJECT}', help='link project')

    parser.add_argument(
        '--save_folder', default=f'test', help='folder to save results')
    # simulation parameters
    parser.add_argument('--method', default='NAFA', help='scheduling method')
    parser.add_argument('--lambda_r', type=int, default=30,
                        help='request arrival rate')
    parser.add_argument('--panel_size', default=0.5, help='solar panel size')
    parser.add_argument('--tradeoff', type=float,
                        default=3.0, help='tradeoff parameter')
    # NAFA parameters
    parser.add_argument('--learning_rate', default=5e-4,
                        help='NAFA learning rate')
    parser.add_argument('--update_tar_interval', default=5000,
                        help='target network update periodicity')
    parser.add_argument('--batch_size', default=64, help='mini-batch size')
    parser.add_argument('--num_actions', default=64, help='number of actions')
    parser.add_argument('--max_buff', default=1e6, help='replay memory size')
    parser.add_argument('--epsilon', default=0.5, help='initial epsilon')
    parser.add_argument('--epsilon_min', default=0.01, help='final epsilon')
    parser.add_argument('--eps_decay', default=30000,
                        help='decay rate of epsilon')
    parser.add_argument('--discount', type=float,
                        default=0.995, help='rewards discount')
    parser.add_argument('--print_interval', default=2000,
                        help='print interval')
    parser.add_argument('--trial', default=1, help='trial number')
    args = parser.parse_args()
    return args
