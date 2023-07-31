import os
import random
import gym
import pylab
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from keras import backend as K
from BinhRL.util import clone_model
#from BinhRL.core import Agent

class DQNAgent():
    def __init__(self, env, model, policy, test_policy = None, dueling = False, ddqn = False,
                 *args, **kwargs):
        #super(DQNAgent, self).__init__(*args, **kwargs)
        # self.env_name = env_name       
        # self.env = gym.make(env_name)
        # self.env.seed(0)  
        self.env = env
        self.model = model
        self.model.compile(optimizer='sgd', loss='mse')
        self.policy = policy
        self.test_policy = test_policy
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.model.layers[0].output_shape[0][1]
        self.action_size = self.model.layers[-1].output_shape

        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate

        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = 1.0 # exploration probability at start
        self.epsilon_min = 0.01 # minimum exploration probability
        self.epsilon_decay = 0.0005 # exponential decay rate for exploration prob
        
        self.batch_size = 32

        # defining model parameters
        self.ddqn = ddqn # use double deep q network
        self.Soft_Update = False # use soft parameter update
        self.dueling = dueling # use dealing network
        self.epsilon_greedy = True # use epsilon greedy strategy

        self.TAU = 0.1 # target network soft update hyperparameter

        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)

        #self.Model_name = os.path.join(self.Save_Path, self.env_name+"_e_greedy.h5")
        self.Model_name = os.path.join(self.Save_Path, "_e_greedy.h5")
        
        # create main model and target model
        self.action_size = self.model.output.shape[-1]
        self.dueling_type = 'a'
        if self.dueling:
            layer = model.layers[-2].output
            # caculate the Q(s,a;theta)
            # dueling_type == 'avg'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            # dueling_type == 'max'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            # dueling_type == 'naive'
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            state_value = Dense(1, kernel_initializer='he_uniform')(layer)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.action_size,))(state_value)

            action_advantage = Dense(self.action_size, kernel_initializer='he_uniform')(layer)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), 
                                      output_shape=(self.action_size,))(action_advantage)

            X = Add()([state_value, action_advantage])
    
            model = Model(inputs=model.input, outputs=X, name='dqn')
            
        self.target_model = clone_model(self.model, {})
        self.target_model.compile(optimizer='sgd', loss='mse')

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        self.memory.append((experience))

    # def act(self, state, decay_step):
    #     # EPSILON GREEDY STRATEGY
    #     if self.epsilon_greedy:
    #     # Here we'll use an improved version of our epsilon greedy strategy for Q-learning
    #         explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * decay_step)
    #     # OLD EPSILON STRATEGY
    #     else:
    #         if self.epsilon > self.epsilon_min:
    #             self.epsilon *= (1-self.epsilon_decay)
    #         explore_probability = self.epsilon

    #     if explore_probability > np.random.rand():
    #         # Make a random action (exploration)
    #         return random.randrange(self.action_size), explore_probability
    #     else:
    #         # Get action from Q-network (exploitation)
    #         # Estimate the Qs values state
    #         # Take the biggest Q value (= the best action)
    #         return np.argmax(self.model.predict(state)), explore_probability

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        # Randomly sample minibatch from the memory
        self.batch_size = 1
        minibatch = random.sample(self.memory, self.batch_size)
        # print(minibatch)
        # print("minibatch")
        minibatch = np.array(minibatch)
        #print(minibatch)

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []
        #print(state)

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        #state = minibatch[0]
        #print("--------------------------")
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        # predict Q-values for starting state using the main network
        target = self.model.predict(state)
        # predict best action in ending state using the main network
        target_next = self.model.predict(next_state)
        # predict Q-values for ending state using the target network
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn: # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])   
                else: # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    pylab.figure(figsize=(18, 9))
    # def PlotModel(self, score, episode):
    #     self.scores.append(score)
    #     self.num_episode.append(episode)
    #     self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
    #     pylab.plot(self.num_episode, self.average, 'r')
    #     pylab.plot(self.num_episode, self.scores, 'b')
    #     pylab.ylabel('Score', fontsize=18)
    #     pylab.xlabel('Steps', fontsize=18)
    #     dqn = 'DQN_'
    #     softupdate = ''
    #     dueling = ''
    #     greedy = ''
    #     if self.ddqn: dqn = 'DDQN_'
    #     if self.Soft_Update: softupdate = '_soft'
    #     if self.dueling: dueling = '_Dueling'
    #     if self.epsilon_greedy: greedy = '_Greedy'
    #     try:
    #         # pylab.savefig(dqn+self.env_name+softupdate+dueling+greedy+".png")
    #         pylab.savefig(dqn+softupdate+dueling+greedy+".png")
    #     except OSError:
    #         pass

    #     return str(self.average[-1])[:5]
    
    def fit(self, num_episode):
        num_episode = num_episode + 1
        for e in range(1, num_episode):
            state = self.env.reset()
            try:
                state = np.reshape(state, [1, self.state_size])
            except Exception as e:
                print(e)
            done = False
            score = 0
            i = 0
            
            while not done:
                # i = i + 1
                # print(i)
                q_values = self.model.predict(state).flatten()
                try:
                    action = self.policy.select_action(q_values)
                except Exception as e:
                    print(e)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                score += reward
                if done:
                    self.update_target_model()
                    
                    print("episode: {}/{}, score: {}, average: {}, e: {:.2f}".
                          format(e, num_episode, score, score/i, self.policy.get_config()['eps']))
                    # if i == self.num_episode:
                    #     print("Saving trained model to", self.Model_name)
                    #     self.save(self.Model_name)
                    #     break

                self.replay()
                
    def test(self, num_episode):
        num_episode = num_episode + 1
        for e in range(1, num_episode):
            state = self.env.reset()
            try:
                state = np.reshape(state, [1, self.state_size])
            except Exception as e:
                print(e)
            done = False
            score = 0
            i = 0
            while not done:
                #self.env.render()
                q_values = self.model.predict(state).flatten()
                action = self.test_policy.select_action(q_values)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                score += reward
                if done:
                    self.update_target_model()
                    
                    print("episode: {}/{}, score: {}, average: {}, e: {:.2f}".
                          format(e, num_episode, score, score/i, self.policy.get_config()['eps']))
                    # if i == self.num_episode:
                    #     print("Saving trained model to", self.Model_name)
                    #     self.save(self.Model_name)
                    #     break

                self.replay()

    # def test(self):
    #     self.load(self.Model_name)
    #     for e in range(self.num_episode):
    #         state = self.env.reset()
    #         state = np.reshape(state, [1, self.state_size])
    #         done = False
    #         i = 0
    #         while not done:
    #             self.env.render()
    #             action = np.argmax(self.model.predict(state))
    #             next_state, reward, done, _ = self.env.step(action)
    #             state = np.reshape(next_state, [1, self.state_size])
    #             i += 1
    #             if done:
    #                 print("episode: {}/{}, score: {}".format(e, self.num_episode, i))
    #                 break