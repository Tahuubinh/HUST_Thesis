from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import copy
import json
import timeit
import warnings
from tempfile import mkdtemp
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from rl.agents.ddpg import DDPGAgent
# from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SARSAAgent
from rl.callbacks import Callback, FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
# from rl.policy import EpsGreedyQPolicy
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.backend import cast
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Dropout,
                                     Flatten, Input, BatchNormalization)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import sys

from environment.environment import *
from environment.mix_state_env import MixStateEnv
from algorithm.value_based.model import *
from algorithm.value_based.policy import *
from algorithm.value_based.callback import *
import os
from config.config import *
from config.MyGlobal import MyGlobals
from keras.models import load_model

from algorithm.value_based.dqnMEC import DQNAgent
from algorithm.value_based.ExpectedTaskDQN import ExpectedTaskDQN
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def Run_Random(folder_name):
    MyGlobals.folder_name = folder_name + '/'
    env = BusEnv("Random")
    env.seed(123)
    observation = None
    done = False
    actions = [0, 1, 2, 3, 4, 5]
    for i in range(NB_STEPS):
        if observation is None:
            try:
                env.reset()
            except Exception as e:
                print(e)
        # Determine the percentage of offload to server
        action = random.choices(actions, weights=(4, 1, 1, 1, 1, 1))[0]
        observation, r, done, info = env.step(action)
        if done:
            done = False
            try:
                env.reset()
            except Exception as e:
                print(e)


def runShortestLatencyGreedy(folder_name):
    MyGlobals.folder_name = folder_name + '/'
    env = BusEnv("Random")
    env.seed(123)
    observation = None
    done = False
    actions = [range(NUM_ACTION)]
    nb_steps = NB_STEPS + NUM_TASKS_PER_TIME_SLOT * 31
    for i in range(nb_steps):
        if observation is None:
            try:
                env.reset()
            except Exception as e:
                print("runShortestLatencyGreedy: ", e)
        # Determine the percentage of offload to server
        min_est_latency = 100
        action = 0
        for server in range(NUM_ACTION):
            time_before_return, _ = env.estimate(server)
            if (min_est_latency > time_before_return):
                min_est_latency = time_before_return
                action = server

        observation, r, done, info = env.step(action)
        if done:
            done = False
            print(i // NUM_TASKS_PER_TIME_SLOT)
            print(env.old_avg_reward)
            try:
                env.reset()
            except Exception as e:
                print(e)


def runShortestExtraTimeGreedy(folder_name):
    MyGlobals.folder_name = folder_name + '/'
    env = BusEnv("Random")
    env.seed(123)
    observation = None
    done = False
    actions = [range(NUM_ACTION)]
    nb_steps = NB_STEPS + NUM_TASKS_PER_TIME_SLOT * 31
    for i in range(nb_steps):
        if observation is None:
            try:
                env.reset()
            except Exception as e:
                print("runShortestExtraTimeGreedy: ", e)
        # Determine the percentage of offload to server
        max_est_extra_time = -100
        action = 0
        for server in range(NUM_ACTION):
            time_before_return, est_observation = env.estimate(server)
            est_extra_time = min(0, est_observation[-1] - time_before_return)
            if (max_est_extra_time < est_extra_time):
                max_est_extra_time = est_extra_time
                action = server

        observation, r, done, info = env.step(action)
        if done:
            done = False
            try:
                env.reset()
                print(env.old_avg_reward)
            except Exception as e:
                print(e)


# using for DQL
# def build_model(state_size, num_actions):
#     input = Input(shape=(1,state_size))
#     x = Flatten()(input)
#     #x = Dense(16, activation='relu')(x)

#     x = Dense(32, activation='relu')(x)

#     x = Dense(32, activation='relu')(x)

#     x = Dense(16, activation='relu')(x)

#     output = Dense(num_actions, activation='linear')(x)
#     model = Model(inputs=input, outputs=output)
#     return model

# def build_model(state_size, num_actions):
#     input = Input(shape=(1,state_size))
#     x = Flatten()(input)

#     x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)

#     #x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)

#     x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)

#     x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)

#     output = Dense(num_actions, activation='linear')(x)
#     model = Model(inputs=input, outputs=output)
#     return model

# def build_model(state_size, num_actions):
#     input = Input(shape=(1,state_size))
#     x = Flatten()(input)

#     x = Dense(32, activation='relu')(x)

#     x = Dense(64, activation='relu')(x)

#     x = Dense(32, activation='relu')(x)

#     output = Dense(num_actions)(x)
#     model = Model(inputs=input, outputs=output)
#     return model

def build_model(state_size, num_actions):
    input = Input(shape=(1, state_size))
    x = Flatten()(input)
    x = Dense(128, activation='relu', name='dense1')(x)
    x = Dense(256, activation='relu', name='dense2')(x)

    output = Dense(num_actions)(x)
    model = Model(inputs=input, outputs=output)
    return model


def get_model():
    try:
        model = load_model('my_model.h5')
    except Exception as e:
        print("Error in get_model")
        print(e)
    return model


def Run_DQL(folder_name, target_model_update=1e-2, gamma=0.995):
    # model=build_model(NUM_STATE, NUM_ACTION)
    tf.keras.backend.clear_session()
    model = build_model(NUM_STATE, NUM_ACTION)
    # model = load_model('my_model9.h5')
    num_actions = NUM_ACTION
    # policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.1, decreased_quantity=.01,
                                           nb_hard_decreased_steps=NUM_TASKS_PER_TIME_SLOT * 9)
    # policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)

    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
                   batch_size=32, target_model_update=target_model_update, policy=policy, gamma=gamma, train_interval=8)
    # files = open("testDQL.csv","w")
    # files.write("kq\n")
    # create callback
    callbacks = CustomerTrainEpisodeLogger("DQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_DQL.h5f", interval=50000)
    # callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2)
        dqn.policy = EpsGreedyQPolicy(0.0)
        dqn.test(env, nb_episodes=31)
    except Exception as e:
        print(e)

def Run_ExpectedTaskDDQN(folder_name, target_model_update=1e-2, gamma=0.995, time_slot_policy=9, expectation_update_krate=0.9999):
    tf.keras.backend.clear_session()
    model = build_model(NUM_STATE, NUM_ACTION)
    # model.save('my_model9.h5')
    # model = load_model('my_model9.h5')
    num_actions = NUM_ACTION
    # policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.1, decreased_quantity=.01,
                                           nb_hard_decreased_steps=NUM_TASKS_PER_TIME_SLOT * time_slot_policy)
    # policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)

    # dqn = ExpectedTaskDQN(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
    #           batch_size = 32, target_model_update=1e-3, policy=policy,gamma=0.95,train_interval=5,
    #           enable_double_dqn=True)

    dqn = ExpectedTaskDQN(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
                          batch_size=32, target_model_update=target_model_update, policy=policy, gamma=gamma, train_interval=6,
                          enable_double_dqn=True, enable_dueling_network=True, expectation_update_krate=expectation_update_krate)
    # files = open("testDQL.csv","w")
    # files.write("kq\n")
    # create callback
    callbacks = CustomerTrainEpisodeLogger("ExpectedTaskDDQN_5phut.csv")
    callback2 = ModelIntervalCheckpoint(
        "weight_ExpectedTaskDDQN.h5f", interval=50000)
    # callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        begin = time.time()
        # for i in range(NUM_VEHICLE // 5):
        #     env.replay()
        #     dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2)
        dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2)
        dqn.policy = EpsGreedyQPolicy(0.0)
        endtrain = time.time()
        dqn.test(env, nb_episodes=31)
        endtest = time.time()
        timedict = {'train': endtrain - begin, 'test': endtest - endtrain}
        with open(f'{RESULT_DIR}/{folder_name}/time.json', 'w') as fp:
            json.dump(timedict, fp)
        model.save(f'{RESULT_DIR}/{folder_name}/model.h5')
    except Exception as e:
        print(e)


def Run_DDQL(folder_name, target_model_update=1e-2, gamma=0.995):
    # model=build_model(NUM_STATE, NUM_ACTION)
    tf.keras.backend.clear_session()
    model = build_model(NUM_STATE, NUM_ACTION)
    # model.save('my_model3.h5')
    num_actions = NUM_ACTION
    # policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.1, decreased_quantity=.01,
                                           nb_hard_decreased_steps=NUM_TASKS_PER_TIME_SLOT * 9)
    # policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)

    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
                   batch_size=32, target_model_update=target_model_update, policy=policy, gamma=gamma, train_interval=8,
                   enable_double_dqn=True)
    # files = open("testDQL.csv","w")
    # files.write("kq\n")
    # create callback
    callbacks = CustomerTrainEpisodeLogger("DDQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_DDQL.h5f", interval=50000)
    # callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2)
        dqn.policy = EpsGreedyQPolicy(0.0)
        dqn.test(env, nb_episodes=31)
    except Exception as e:
        print(e)

def Run_D3QL(folder_name, target_model_update=1e-2, gamma=0.995):
    # model=build_model(NUM_STATE, NUM_ACTION)
    tf.keras.backend.clear_session()
    model = build_model(NUM_STATE, NUM_ACTION)
    # model.save('my_model3.h5')
    num_actions = NUM_ACTION
    # policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.1, decreased_quantity=.01,
                                           nb_hard_decreased_steps=NUM_TASKS_PER_TIME_SLOT * 9)
    # policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)

    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
                   batch_size=32, target_model_update=target_model_update, policy=policy, gamma=gamma, train_interval=8,
                   enable_double_dqn=True, enable_dueling_network=True)
    # files = open("testDQL.csv","w")
    # files.write("kq\n")
    # create callback
    callbacks = CustomerTrainEpisodeLogger("DDQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_DDQL.h5f", interval=50000)
    # callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2)
        dqn.policy = EpsGreedyQPolicy(0.0)
        dqn.test(env, nb_episodes=31)
    except Exception as e:
        print(e)

def Run_DuelingDQL(folder_name, target_model_update=1e-2, gamma=0.995):
    # model=build_model(NUM_STATE, NUM_ACTION)
    tf.keras.backend.clear_session()
    model = build_model(NUM_STATE, NUM_ACTION)
    # model = load_model('my_model9.h5')
    num_actions = NUM_ACTION
    # policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.1, decreased_quantity=.01,
                                           nb_hard_decreased_steps=NUM_TASKS_PER_TIME_SLOT * 9)
    # policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)

    # dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
    #           batch_size = 32, target_model_update=1e-3, policy=policy,gamma=0.95,train_interval=5,
    #           enable_dueling_network=True)

    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
                   batch_size=32, target_model_update=target_model_update, policy=policy, gamma=gamma, train_interval=8,
                   enable_dueling_network=True)
    # files = open("testDQL.csv","w")
    # files.write("kq\n")
    # create callback
    callbacks = CustomerTrainEpisodeLogger("DuelingDQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint(
        "weight_DuelingDQL.h5f", interval=50000)
    # callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2)
        dqn.policy = EpsGreedyQPolicy(0.0)
        dqn.test(env, nb_episodes=31)
    except Exception as e:
        print(e)


def Run_Sarsa(folder_name, target_model_update=1e-2, gamma=0.995):
    # model=build_model(NUM_STATE, NUM_ACTION)
    tf.keras.backend.clear_session()
    model = build_model(NUM_STATE, NUM_ACTION)
    # model = load_model('my_model9.h5')
    num_actions = NUM_ACTION
    # policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.9, decreased_quantity=.01,
                                           nb_hard_decreased_steps=NUM_TASKS_PER_TIME_SLOT)
    # policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    try:
        dqn = SARSAAgent(model=model, nb_actions=num_actions, nb_steps_warmup=10,
                         policy=policy, gamma=gamma, train_interval=1)
    except Exception as e:
        print(e)

    dqn.compile(Adam(learning_rate=0.005), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2)
        dqn.policy = EpsGreedyQPolicy(0.0)
        dqn.test(env, nb_episodes=31)
    except Exception as e:
        print(e)


def runOtherExpectedTaskDDQN(folder_name):
    # model=build_model(NUM_STATE, NUM_ACTION)
    tf.keras.backend.clear_session()
    model = load_model('my_model9.h5')
    num_actions = NUM_ACTION
    # policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.2, decreased_quantity=.05,
                                           nb_hard_decreased_steps=NUM_TASKS_PER_TIME_SLOT * 25 / 2)
    # policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = NoFogEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)

    # train_interval for number of step before backtracking
    dqn = ExpectedTaskDQN(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
                          batch_size=32, target_model_update=1e-3, policy=policy, gamma=0.95, train_interval=5,
                          enable_double_dqn=True)
    # files = open("testDQL.csv","w")
    # files.write("kq\n")
    # create callback
    callbacks = CustomerTrainEpisodeLogger("ExpectedTaskDDQN_5phut.csv")
    callback2 = ModelIntervalCheckpoint(
        "weight_ExpectedTaskDDQN.h5f", interval=50000)
    # callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps=NB_STEPS, visualize=False,
                verbose=2, callbacks=[callbacks, callback2])
        dqn.policy = EpsGreedyQPolicy(0.0)
        dqn.test(env, nb_episodes=31)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    for i in range(1, 3):
        # Run_DQL("DQN_800_30s/" + str(i))
        # Run_ExpectedTaskDQN("ExpectedTaskDQN/" + str(i))
        # Run_ExpectedTaskDDQN("ExpectedTaskD3QN_800_30s_dec8_train120_gamma_0.9_a_0.01/" + str(i), gamma = 0.9, target_model_update=1e-2)
        # Run_ExpectedTaskDDQN("ExpectedTaskD3QN_800_30s_dec8_train120_gamma_0.95_a_0.01/" + str(i), gamma = 0.95, target_model_update=1e-2)
        # Run_ExpectedTaskDDQN("ExpectedTaskD3QN_800_30s_dec8_train120_gamma_0.99_a_0.01/" + str(i), gamma = 0.99, target_model_update=1e-2)
        try:
            # Run_ExpectedTaskDDQN("transfer/AODAI/15_VS/" + str(i), gamma=0.995,
            #                      target_model_update=1e-2)
            Run_ExpectedTaskDDQN("test/" + str(i), gamma=0.995,
                                 target_model_update=1e-2)
        except Exception as e:
            print(e)
        # Run_ExpectedTaskDDQN(f"ExpectedTaskD3QN_many_VEC/{NUM_TASKS_PER_TIME_SLOT}_{NUM_VEHICLE}",
        #                       gamma=0.995, target_model_update=1e-2, time_slot_policy=NUM_VEHICLE // 5 * 9)
        # Run_ExpectedTaskDDQN("ExpectedTaskD3QN_800_30s_dec8_train120_g0_995_a_0.05/" + str(i), gamma = 0.995, target_model_update=0.05)
        # Run_ExpectedTaskDDQN("ExpectedTaskD3QN_800_30s_dec8_train120_no_BS/" + str(i), gamma = 0.995, target_model_update=1e-2)
        # Run_ExpectedTaskDDQN("test/1", gamma = 0.995, target_model_update=1e-2)
        # Run_ExpectedTaskDDQN("ExpectedTaskDDQN_sync_step/a0.05/" + str(3))
        # Run_DDQL("DDQN_800_30s/" + str(i))
        # Run_DuelingDQL("DuelingDQN_800_30s/" + str(i))
        # Run_Sarsa("Sarsa_800_30s_dec8_train120/" + str(21))
        # runShortestLatencyGreedy(
        #     f"ShortestLatencyGreedy/{NUM_TASKS_PER_TIME_SLOT}_{NUM_VEHICLE}")

        # runShortestExtraTimeGreedy("ShortestExtraTimeGreedy/800")
        # runOtherExpectedTaskDDQN("a/" + str(11))
        # runOtherExpectedTaskDDQN("ExpectedTaskDDQN_0.2_0.1_25_normal_no_fog_servers/" + str(i))
