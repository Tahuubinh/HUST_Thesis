from copy import deepcopy
import torch
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from util.data_util import read_energy_data
from schedule.value_based_rl.buffer.prioritized_experience_replay import PrioritizedExperienceReplay
from schedule.value_based_rl.DeepRLModel import DeepRLModel

# network architecture


class DQN(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, actions_dim)
        )

    def forward(self, x):
        return self.nn(x)


class NAFAPER_Agent(DeepRLModel):
    def __init__(self, *args, **kwargs):
        super(NAFAPER_Agent, self).__init__(*args, **kwargs)
        print(self.args)
        self.is_training = True
        self.buffer = PrioritizedExperienceReplay(int(self.args.max_buff))
        self.action_dim = self.env.action_space.n
        self.model = DQN(self.env.observation_space.shape[0], self.action_dim).to(
            self.args.device)
        self.target_model = DQN(
            self.env.observation_space.shape[0], self.action_dim).to(self.args.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = Adam(self.model.parameters(),
                                lr=self.args.learning_rate)

        # non-Linear epsilon decay
        epsilon_final = self.args.epsilon_min
        epsilon_start = self.args.epsilon
        epsilon_decay = self.args.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

    # back-propagation
    def learning(self, fr):
        tree_idx, experiences, weights = self.buffer.sample(
            self.args.batch_size)
        s0, a, r, s1, done = zip(*experiences)
        s0 = np.asarray(s0)
        s1 = np.asarray(s1)
        r = torch.tensor(r, dtype=torch.float).to(self.args.device)
        s0_input = self.uniformState(s0)
        s1_input = self.uniformState(s1)
        a = torch.tensor(a, dtype=torch.long).to(self.args.device)
        # done = torch.tensor(done, dtype=torch.float).to(self.args.device)
        q_values = self.model(s0_input)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(s1_input)
        max_q_action = self.findMaxAction(s1, next_q_values)
        max_q_action = torch.tensor(max_q_action).to(self.args.device)

        next_q_state_values = self.target_model(s1_input)

        next_q_value = next_q_state_values.gather(
            1, max_q_action.unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.args.discount * next_q_value
        # Notice that we need to detach the expected_q_value
        loss = (torch.FloatTensor(weights) *
                F.mse_loss(q_value, expected_q_value.detach())).mean()
        #loss = (q_value - expected_q_value.detach()).pow(2).mean()
        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if fr % self.args.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        absolute_errors = np.abs(
            expected_q_value.detach() - q_value.detach()).tolist()
        # print(absolute_errors)
        # assert 2 == 3
        self.buffer.batch_update(tree_idx, absolute_errors)
        # for i in range(self.args.batch_size):
        #     absolute_errors = np.abs(expected_q_value[i].detach() - q_value[i].detach()).item()
        #     # Update priority
        #     self.buffer.batch_update(tree_idx[i], absolute_errors)
        return loss.item()

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    # NAFA's training stage
    # def train(self):
    #     losses = []
    #     all_rewards = []
    #     counters = []
    #     fr = 0
    #     episode_reward = 0
    #     GHI_Data = read_energy_data(is_train=True)
    #     epsilon = None
    #     for series in range(self.num_series):
    #         done = False
    #         ep_num = 0
    #         self.env.replay(is_train=True, simulation_start=ep_num * self.ep_long,
    #                                simulation_end=(ep_num + 1) * self.ep_long, GHI_Data=GHI_Data)
    #         print(f'SERIES {series}')
    #         while ep_num < self.max_episode:
    #             state = self.env.reset(is_train=True, simulation_start=ep_num * self.ep_long,
    #                                    simulation_end=(ep_num + 1) * self.ep_long, GHI_Data=GHI_Data)
    #             done = False
    #             while not done:
    #                 fr += 1
    #                 epsilon = self.epsilon_by_frame(fr)
    #                 action = self.act(state, epsilon)
    #                 next_state, reward, done = self.env.step(action)
    #                 self.buffer.append(
    #                     [state, action, reward, next_state, False])
    #                 episode_reward += reward
    #                 all_rewards.append(reward)
    #                 counters.append(self.env.counter)
    #                 if self.buffer.size() > self.args.batch_size:
    #                     loss = self.learning(fr)
    #                     losses.append(loss)
    #                 else:
    #                     losses.append(0)

    #                 # print(self.args)
    #                 # print(self.args.print_interval)
    #                 # print(done)
    #                 # break
    #                 # if fr % self.args.print_interval == 0:
    #                 #     print("PERIOD: %d\nrewards: %f, losses: %f, episode: %d" % (
    #                 #         self.env.counter / 24, episode_reward / ((self.env.counter - self.env.simulation_start) / 24),
    #                 #         np.sum(losses[-100:]) / 100, ep_num))
    #                 #     print("epsilon = {}".format(epsilon))
    #                 state = next_state

    #             # print('Episode: {}\nrewards: {}  epsilon: {} losses: {}'.format(ep_num, episode_reward, epsilon,
    #             #                                                                 np.sum(losses[-100:]) / 100))
    #             self.saveResults()
    #             episode_reward = 0
    #             ep_num += 1
    #             # self.env.event_queue.print_queue()
