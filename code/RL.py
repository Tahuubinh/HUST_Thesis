from main import Run_ExpectedTaskDDQN
from BinhRL.util.options import args_parser
from BinhRL.value_based_rl.NAFA import NAFA_Agent
from mix_state_env import MixStateEnv
import torch
from config import *

if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device("cpu")
    env = MixStateEnv()
    env.seed(123)
    args.epsilon_min = 0
    args.epsilon = 0.1
    args.learning_rate = 0.001
    args.batch_size = 32
    args.num_actions = NUM_ACTION
    agent = NAFA_Agent(args, env, num_series=1,
                       max_episode=121, ep_long=800)

    agent.train()
