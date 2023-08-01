import numpy as np
import torch
import gym
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from environment import *
from mix_state_env import MixStateEnv
from config import *
import copy
from MyGlobal import MyGlobals
from itertools import count
from torch.distributions import Categorical
import keras
from keras.models import load_model

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

state_size = NUM_STATE
action_size = NUM_ACTION
lr = 0.001


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train(actor, critic, num_iters, num_episodes, duration, gamma, env):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    exploit_rate_files = open(
        RESULT_DIR + MyGlobals.folder_name + "exploit_rate.csv", "w")
    exploit_rate_files.write('1')
    for i in range(2, NUM_ACTION + 1):
        exploit_rate_files.write(',' + str(i))
    exploit_rate_files.write('\n')

    for iter in range(num_iters):
        env.replay()
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            count_exploit = [0] * NUM_ACTION

            while not done:
                log_probs = []
                values = []
                rewards = []
                masks = []
                for i in count():
                    state = torch.FloatTensor(state).to(device)
                    dist, value = actor(state), critic(state)

                    action = dist.sample()
                    prob_dist = dist.probs
                    # print(type(prob_dist))
                    # print(prob_dist)
                    # print(action)
                    # print(torch.topk(prob_dist.flatten(), NUM_ACTION).indices.tolist())
                    # print(torch.topk(prob_dist.flatten(), NUM_ACTION).indices.tolist().index(action))
                    # print(torch.topk(
                    #     prob_dist.flatten(), NUM_ACTION))
                    # assert 2 == 3
                    count_exploit[torch.topk(
                        prob_dist.flatten(), NUM_ACTION).indices.tolist().index(action)] += 1
                    # if action == dist.probs.argmax():
                    #     count_exploit += 1
                    next_state, reward, done, _ = env.step(
                        action.cpu().numpy())

                    log_prob = dist.log_prob(action).unsqueeze(0)

                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(torch.tensor(
                        [reward], dtype=torch.float, device=device))
                    masks.append(torch.tensor(
                        [1-done], dtype=torch.float, device=device))

                    state = next_state

                    if done:
                        if (env.old_avg_reward < -1500):
                            return
                        print('Episode: {}, Score: {}'.format(
                            episode, env.old_avg_reward))

                        # print(dist.probs)
                        #print('Iteration: {}, Score: {}'.format(episode, i))
                        break

                    if (i > duration):
                        break

                next_state = torch.FloatTensor(next_state).to(device)
                next_value = critic(next_state)
                returns = compute_returns(
                    next_value, rewards, masks, gamma=gamma)

                log_probs_cat = torch.cat(log_probs)
                returns_cat = torch.cat(returns).detach()
                returns_cat = (returns_cat - torch.mean(returns_cat)) / (torch.std(returns_cat) + 0.0001)
                values_cat = torch.cat(values)

                advantage = returns_cat - values_cat

                actor_loss = -(log_probs_cat * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()

                optimizerA.zero_grad()
                optimizerC.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                optimizerA.step()
                optimizerC.step()

            tempstr = ','.join([str(elem) for elem in count_exploit])
            exploit_rate_files.write(tempstr+"\n")
            # exploit_rate_files.write('{}\n'.format(count_exploit))
            print(tempstr)

    exploit_rate_files.close()


def test(actor, critic, num_episodes, env):
    if (env.old_avg_reward < -1500):
        return
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.probs.argmax()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            state = next_state

        print('Test Episode: {}, Score: {}'.format(episode, env.old_avg_reward))

# MyGlobals.folder_name = "Actor_Critic/dur10_g_0_99/" + '1' +'/'


def runAC(i, dur, gamma):
    # MyGlobals.folder_name = "Actor_Critic_800_30s/dur" + str(dur) + "/" + str(i) +'/'
    MyGlobals.folder_name = f"test/gamma{gamma}/dur{dur}/{i}/"
    env = MixStateEnv()
    env.seed(123)
    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size, action_size).to(device)
    keras_model = load_model(f'{RESULT_DIR}ExpectedTaskD3QN_many_VEC/{NUM_TASKS_PER_TIME_SLOT}_{NUM_VEHICLE}/model.h5')
    # for i, layer in enumerate(keras_model.layers):
    #     if isinstance(layer, keras.layers.Dense):
    #         weights, biases = layer.get_weights()
    #         print(i, weights.shape)
    #         pytorch_model_state = actor.state_dict()
    #         pytorch_model_state[f'linear{i-1}.weight'] = torch.from_numpy(weights.transpose())
    #         pytorch_model_state[f'linear{i-1}.bias'] = torch.from_numpy(biases)
    #         actor.load_state_dict(pytorch_model_state)
    #         break
    # for param in actor.parameters():
    #     param.requires_grad = False
    # actor.linear2 = nn.Linear(128, 256)
    # actor.linear3 = nn.Linear(256, NUM_ACTION)
    # for i, layer in enumerate(keras_model.layers):
    #     if isinstance(layer, keras.layers.Dense):
    #         weights, biases = layer.get_weights()
    #         print(i, weights.shape)
    #         pytorch_model_state = critic.state_dict()
    #         pytorch_model_state[f'linear{i-1}.weight'] = torch.from_numpy(weights.transpose())
    #         pytorch_model_state[f'linear{i-1}.bias'] = torch.from_numpy(biases)
    #         if i == 2:
    #             break
    #         critic.load_state_dict(pytorch_model_state)
    # for param in critic.parameters():
    #     param.requires_grad = False
    # critic.linear2 = nn.Linear(128, 256)
    # critic.linear3 = nn.Linear(256, 1)
    # critic.linear3.requires_grad = True
    train(actor, critic, num_iters=9, num_episodes=121,
          duration=dur, gamma=gamma, env=env)
    test(actor, critic, num_episodes=31, env=env)


runAC(1, 1000, 0.995)
# for i in range(1, 51):
#     runAC(i, dur = 10, gamma = 0.9)
#     runAC(i, dur = 15, gamma = 0.9)
#     runAC(i, dur = 20, gamma = 0.9)
#     runAC(i, dur = 25, gamma = 0.9)
#     runAC(i, dur = 30, gamma = 0.9)
#     runAC(i, dur = 10, gamma = 0.99)
#     runAC(i, dur = 15, gamma = 0.99)
#     runAC(i, dur = 20, gamma = 0.99)
#     runAC(i, dur = 25, gamma = 0.99)
#     runAC(i, dur = 30, gamma = 0.99)

# for i in range(1, 31):
#     runAC(i, dur = 10, gamma = 0.99)

# for i in range(1, 31):
#     runAC(i, dur = 15, gamma = 0.99)

# for i in range(1, 31):
#     runAC(i, dur = 20, gamma = 0.99)

# for i in range(1, 31):
#    runAC(i, dur = 25, gamma = 0.99)

# for i in range(1, 31):
#    runAC(i, dur = 30, gamma = 0.99)

# for i in range(1, 31):
#    runAC(i, dur = 10, gamma = 0.9)

# for i in range(1, 31):
#    runAC(i, dur = 15, gamma = 0.9)

# for i in range(1, 31):
#    runAC(i, dur = 20, gamma = 0.9)

# for i in range(1, 31):
#    runAC(i, dur = 25, gamma = 0.9)

# for i in range(1, 31):
#    runAC(i, dur = 30, gamma = 0.9)

# for i in range(1, 31):
#     runAC(i, dur = 35, gamma = 0.9)

# for i in range(1, 31):
#     runAC(i, dur = 40, gamma = 0.9)

# for i in range(1, 31):
#     runAC(i, dur = 35, gamma = 0.99)

# for i in range(1, 31):
#     runAC(i, dur = 40, gamma = 0.99)

# runAC(1 , dur = 30, gamma = 0.99)
