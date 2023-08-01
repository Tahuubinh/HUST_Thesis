import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import copy
from environment import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0").unwrapped
#env = BusEnv("Random")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0001

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = self.linear4(output)
        #print(output)
        #print(F.softmax(output, dim=-1))
        distribution = Categorical(F.softmax(output, dim=-1))
        #print(distribution)
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        value = self.linear4(output)
        return value
    
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in count():
            if i > 300:
                print('Iteration: {}, Score: {}'.format(iter, i))
                return
                break
            log_probs = []
            values = []
            #env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            
            next_state, reward, done, _, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            next_state = torch.FloatTensor(next_state).to(device)
            next_value = critic(next_state)
            returns = compute_returns(next_value, rewards, masks)

            log_probs_cat = torch.cat(log_probs)
            returns_cat = torch.cat(returns).detach()
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


            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break

    env.close()

actor = Actor(state_size, action_size).to(device)
critic = Critic(state_size, action_size).to(device)
trainIters(actor, critic, n_iters=500)

def testIters(actor, n_iters):
    for iter in range(n_iters):
        state = env.reset()
        env.reset()

        for i in count():
            if i > 1000:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break
                
            state = torch.FloatTensor(state).to(device)
            dist = actor(state)
            action = dist.probs.argmax()
            
            next_state, reward, done, _, _ = env.step(action.cpu().numpy())

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break

testIters(actor, 100)