import os
import argparse
from pathlib import Path
import os
import argparse
from pathlib import Path
LINK_PROJECT = Path(os.path.abspath(__file__)).parent


def args_parser():
    parser = argparse.ArgumentParser()
    # link project
    parser.add_argument(
        '--link_project', default=f'{LINK_PROJECT}', help='Link project')
    parser.add_argument(
        '--algorithm', default=f'AODAI', help='Algorithm to run')
    parser.add_argument(
        '--folder_name', default=f'test', help='Folder to save results')
    # simulation parameters
    parser.add_argument('--num_vec', type=int, default=5, help='Number of vehicular edge servers')
    parser.add_argument('--cloud', default='True', help='Including cloud server or not')
    parser.add_argument('--num_times_to_run', type=int, default=1, help='Number of times running the an algorithm, all are independent')
    parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
    parser.add_argument('--target_model_update', type=float, default=0.995, help='cCopy rate of new model')
    parser.add_argument('--expectation_update_krate', type=float, default=0.9999, help='Control the rate of approximating volatile attributes in AODAI')
    parser.add_argument('--dur', type=int, default=1, help='Number of time steps in a pseudo-episode')
    parser.add_argument('--typeTransfer', default='all', help='Type of Neural Network Crafting')
    parser.add_argument('--pre_numVSs', type=int, default=5, help='Number of VSs in previous')
    args = parser.parse_args()
    return args