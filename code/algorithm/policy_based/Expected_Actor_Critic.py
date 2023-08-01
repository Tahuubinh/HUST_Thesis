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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train(actor, critic, num_iters, num_episodes, duration):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(num_iters):
        env.replay()
        for episode in range(num_episodes):
            state = env.reset()
            done = False
    
            while not done:
                log_probs = []
                values = []
                rewards = []
                masks = []
                for i in count():
                    state = torch.FloatTensor(state).to(device)
                    dist, value = actor(state), critic(state)
        
                    action = dist.sample()
                    next_state, reward, done, _ = env.step(action.cpu().numpy())
        
                    log_prob = dist.log_prob(action).unsqueeze(0)
        
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
                    masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
        
                    state = next_state
                    
                    if done:
                        print('Episode: {}, Score: {}'.format(episode, env.old_avg_reward))
                        #print('Iteration: {}, Score: {}'.format(episode, i))
                        break
                    
                    if (i > duration):
                        break
        
        
        
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
                
def test(actor, critic, num_episodes):
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
        
MyGlobals.folder_name = "Actor_Critic/dur10/" + '1' + '/'
env = MixStateEnv()
env.seed(123)
actor = Actor(state_size, action_size).to(device)
critic = Critic(state_size, action_size).to(device)
train(actor, critic, num_iters = 1, num_episodes=61, duration = 10)
test(actor, critic, num_episodes=31)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    