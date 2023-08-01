import numpy as np
import torch
import gym
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from environment import *
from environment.mix_state_env import MixStateEnv
from config.config import *
import copy
from config.MyGlobal import MyGlobals

#A The loss function expects an array of action probabilities for the actions that were taken and the discounted rewards.
#B It computes the log of the probabilities, multiplies by the discounted rewards, sums them all and flips the sign.

l1 = 32
l2 = 64
l3 = 32
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.state_size = NUM_STATE
#         self.action_size = NUM_ACTION
#         self.linear1 = nn.Linear(self.state_size, l1)
#         #torch.nn.init.kaiming_uniform_(self.linear1.weight)
#         self.linear2 = nn.Linear(l1, l2)
#         #torch.nn.init.kaiming_uniform_(self.linear2.weight)
#         self.linear3 = nn.Linear(l2, l3)
#         #torch.nn.init.kaiming_uniform_(self.linear3.weight)
#         self.linear4 = nn.Linear(l3, self.action_size)
#         #torch.nn.init.kaiming_uniform_(self.linear4.weight)
 
#     def forward(self, x):
#         output = F.relu(self.linear1(x))
#         output = F.relu(self.linear2(output))
#         output = F.relu(self.linear3(output))
#         output = self.linear4(output)
#         #print(output)
#         output = F.softmax(output, dim = 1)
#         #print("Softmax: " + str(output))
#         #x = torch.nn.Softmax(dim=0)(x)
#         return output

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.state_size = NUM_STATE
        self.action_size = NUM_ACTION
        self.linear1 = nn.Linear(self.state_size, 128)
        #torch.nn.init.kaiming_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(128, 256)
        #torch.nn.init.kaiming_uniform_(self.linear2.weight)
        self.linear4 = nn.Linear(256, self.action_size)
        #torch.nn.init.kaiming_uniform_(self.linear4.weight)
 
    def forward(self, x):
        output = F.relu(self.linear1(x))
        output = F.relu(self.linear2(output))
        output = self.linear4(output)
        #print(output)
        output = F.softmax(output, dim = 1)
        #print("Softmax: " + str(output))
        #x = torch.nn.Softmax(dim=0)(x)
        return output

MAX_DUR = 500
MAX_EPISODES = 400
score = [] #A

class RFAgent:
    def __init__(self, model, env = None):
        self.env = env
        self.model = model
        learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train(self, env, num_iters, num_episodes, max_dur, gamma, duration):
        duration -= 1
        
        for iter in range(num_iters):
            self.env.replay()
            for episode in range(num_episodes):
                score = []
                curr_state = self.env.reset()
                done = False
                while not done:
                    transitions = [] #B
                    
                    for t in range(max_dur): 
                        state_float_torch = torch.from_numpy(curr_state.reshape([1, -1])).float()
                        act_prob = self.model(state_float_torch)
                        action = np.random.choice(np.array(range(NUM_ACTION)), p=act_prob.data.numpy().flatten())
                        prev_state = copy.copy(curr_state)
                        curr_state, reward, done, info = env.step(action)
                        score.append(reward)
                        transitions.append((prev_state, action, reward))
                        if (t > duration):
                            break
                        if done:
                            break
                        
                    reward_batch = torch.Tensor([r for (s,a,r) in transitions])
                    
                    #print(reward_batch, gamma)
                    disc_returns = self.discount_rewards(reward_batch, gamma = gamma)
                    #print(disc_returns)
                    state_batch = torch.Tensor([s for (s,a,r) in transitions])
                    action_batch = torch.Tensor([a for (s,a,r) in transitions])
                    pred_batch = self.model(state_batch) 
                    
                    prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze() #O
                    #print(prob_batch, disc_returns)
                    loss = self.loss_fn(prob_batch, disc_returns)
                    #print(loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    #print("check model predict after backward: ", model(torch.from_numpy(curr_state.reshape([1, -1])).float()))
                  
                if (env.old_avg_reward < -1500):
                    return
                print('Episode: {}, Score: {}'.format(episode, env.old_avg_reward))
    
    def test(self, env, num_episodes):
        if (env.old_avg_reward < -1500):
            return
        for episode in range(num_episodes):
            curr_state = self.env.reset()
            done = False
            while not done:
                state_float_torch = torch.from_numpy(curr_state.reshape([1, -1])).float()
                act_prob = self.model(state_float_torch)
                # action = np.random.choice(np.array(range(NUM_ACTION)), p=act_prob.data.numpy().flatten())
                action = np.argmax(act_prob.data.numpy().flatten())
                curr_state, reward, done, info = env.step(action)
                score.append(reward)
            print('Test Episode: {}, Score: {}'.format(episode, env.old_avg_reward))
    
    def discount_rewards(self, rewards, gamma=0.99):
        lenr = len(rewards)
        disc_return = torch.pow(gamma,torch.arange(lenr).float()) * rewards #A
        #disc_return /= disc_return.max() #B
        return disc_return

    #A Compute exponentially decaying rewards
    #B Normalize the rewards to be within the [0,1] interval to improve numerical stability

    def loss_fn(self, preds, r): #A
        return -1 * torch.sum(r * torch.log(preds)) #B

def runRF(i, dur, gamma):
    MyGlobals.folder_name = f"Reinforce_800_30s_dec8_train120_iter9/gamma{gamma}/dur{dur}/{i}/"
    env = MixStateEnv()
    env.seed(123)
    model = Model()
    agent = RFAgent(env = env, model = model)
    agent.train(env = env, num_iters = 9, num_episodes = 121, max_dur = MAX_DUR, gamma = gamma, duration = dur)
    agent.test(env = env, num_episodes = 31)

# for i in range(1, 31):
#     runRF(i, dur = 10, gamma = 0.99)
    
# for i in range(1, 31):
#     runRF(i, dur = 15, gamma = 0.99)
    
# for i in range(1, 31):
#     runRF(i, dur = 20, gamma = 0.99)
    
# for i in range(1, 31):
#    runRF(i, dur = 25, gamma = 0.99)
    
# for i in range(1, 31):
#    runRF(i, dur = 30, gamma = 0.99)
    
# for i in range(1, 31):
#    runRF(i, dur = 10, gamma = 0.9)
    
# for i in range(1, 31):
#    runRF(i, dur = 15, gamma = 0.9)
    
# for i in range(1, 31):
#    runRF(i, dur = 20, gamma = 0.9)
    
# for i in range(1, 31):
#    runRF(i, dur = 25, gamma = 0.9)
    
# for i in range(1, 31):
#    runRF(i, dur = 30, gamma = 0.9)

if __name__ == "__main__":
    for i in range(1, 31):
        runRF(i, dur = 15, gamma = 0.9)