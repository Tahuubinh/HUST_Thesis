import os
from pathlib import Path
from args_parser import args_parser
import sys

args = args_parser()

LINK_PROJECT = Path(os.path.abspath(__file__))
LINK_PROJECT = LINK_PROJECT.parent.parent
# print(LINK_PROJECT)
# data 3 is the main one
# 800 5, 1400 10, 2600 20, 3800 30, 5000 40, 6200 50
if args.num_vec == 5:
    NUM_VEHICLE = 5
    NUM_TASKS_PER_TIME_SLOT = 800
elif args.num_vec == 10:
    NUM_VEHICLE = 10
    NUM_TASKS_PER_TIME_SLOT = 1400
elif args.num_vec == 15:
    NUM_VEHICLE = 15
    NUM_TASKS_PER_TIME_SLOT = 2000
elif args.num_vec == 20:
    NUM_VEHICLE = 20
    NUM_TASKS_PER_TIME_SLOT = 2600
elif args.num_vec == 25:
    NUM_VEHICLE = 25
    NUM_TASKS_PER_TIME_SLOT = 3200
elif args.num_vec == 30:
    NUM_VEHICLE = 30
    NUM_TASKS_PER_TIME_SLOT = 3800
else:
    print(f'No data of {args.num_vec} VEC! Choose 5, 10, 15, 20, 25, or 30 instead')
    sys.exit(0)
# DATA_LOCATION = "data_task/data3/"
DATA_LOCATION = "data_task/data" + str(NUM_TASKS_PER_TIME_SLOT) + "/"
DATA_DIR = os.path.join(LINK_PROJECT, "data")
RESULT_DIR = os.path.join(LINK_PROJECT, "result/result3/")
DATA_TASK = os.path.join(LINK_PROJECT, DATA_LOCATION)
COMPUTATIONAL_CAPACITY_900 = 2  # Ghz
COMPUTATIONAL_CAPACITY_901 = 2  # Ghz
COMPUTATIONAL_CAPACITY_902 = 2  # Ghz
COMPUTATIONAL_CAPACITY_903 = 2
COMPUTATIONAL_CAPACITY_904 = 2
COMPUTATIONAL_CAPACITY_905 = 2
COMPUTATIONAL_CAPACITY_906 = 2
COMPUTATIONAL_CAPACITY_907 = 2
COMPUTATIONAL_CAPACITY_LOCAL = 4  # Ghz
COMPUTATIONAL_CAPACITY_CLOUD = 4  # Ghz
TRANS_RATE_EDGE_TO_CLOUD = 1  # Mbps
CHANNEL_BANDWIDTH = 20  # MHz
# List_COMPUTATION = [COMPUTATIONAL_CAPACITY_900, COMPUTATIONAL_CAPACITY_901, COMPUTATIONAL_CAPACITY_902,
#                     COMPUTATIONAL_CAPACITY_903, COMPUTATIONAL_CAPACITY_904, COMPUTATIONAL_CAPACITY_905,
#                     COMPUTATIONAL_CAPACITY_906, COMPUTATIONAL_CAPACITY_907, COMPUTATIONAL_CAPACITY_LOCAL]
# Pr = 46 # dBm
Pr = 46
P = 39.810  # mW
SIGMASquare = 100  # dBm   background noise power
PATH_LOSS_EXPONENT = 4  # alpha
NUM_EDGE_SERVER = NUM_VEHICLE + 1  # a server + vehicles
List_COMPUTATION = [COMPUTATIONAL_CAPACITY_900] * \
    NUM_VEHICLE + [COMPUTATIONAL_CAPACITY_LOCAL]
NUM_STATE = NUM_EDGE_SERVER * 2 + 2
if args.cloud == 'True':
    NUM_ACTION = NUM_EDGE_SERVER + 1  # including cloud, must + 1
else:
    NUM_ACTION = NUM_EDGE_SERVER
# NUM_ACTION = NUM_EDGE_SERVER # no BS or no cloud
# NUM_ACTION = NUM_EDGE_SERVER
SCAILING_CO_EFFICIENT = 1
FILE_NAME = "Not change"
NB_STEPS = NUM_TASKS_PER_TIME_SLOT * 121
# NB_STEPS = 405000


class Config:
    Pr = 46
    Pr2 = 24
    Wm = 10
    length_hidden_layer = 4
    n_unit_in_layer = [16, 32, 32, 8]
