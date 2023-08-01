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

from environment import *
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


def build_model(state_size, num_actions):
    input = Input(shape=(1, state_size))
    x = Flatten()(input)
    x = Dense(128, activation='relu', name='dense1')(x)
    x = Dense(256, activation='relu', name='dense2')(x)

    output = Dense(num_actions)(x)
    model = Model(inputs=input, outputs=output)
    return model


def run_Transfer_AODAI(folder_name, gamma, target_model_update, type, pre_numVSs):
    tf.keras.backend.clear_session()
    # increase_numVSs = NUM_VEHICLE - pre_numVSs
    model = build_model(NUM_STATE, NUM_ACTION)
    if type == 'INNOCENCE':
        type = 0
    elif type == 'ALL':
        type = 1
    elif type == 'MID':
        type = 2
    elif type == 'SIDE':
        type = 3
    elif type == 'NO_TRAINING':
        type = 4
    print(type)
    if type > 0:
        if pre_numVSs == 5:
            premodel = load_model(f'{RESULT_DIR}transfer/AODAI/2/model.h5')
        elif pre_numVSs == 10:
            premodel = load_model(
                f'{RESULT_DIR}transfer/AODAI/10_VS/3/model.h5')
        elif pre_numVSs == 15:
            premodel = load_model(
                f'{RESULT_DIR}transfer/AODAI/15_VS/2/model.h5')

        for layer in range(2, 3):
            for o in range(len(model.layers[layer].get_weights()[1])):
                model.layers[layer].get_weights(
                )[1][o] = premodel.layers[layer].get_weights()[1][o]

            # for i in range(0, pre_numVSs * 2, 1):
            #     # model.layers[layer].get_weights(
            #     # )[1][i] = premodel.layers[layer].get_weights()[1][i]
            #     for k in range(len(model.layers[layer].get_weights()[0][i])):
            #         model.layers[layer].get_weights(
            #         )[0][i][k] = premodel.layers[layer].get_weights()[0][i][k]

            len_next_layer = len(model.layers[layer].get_weights()[1])
            for i in range(NUM_VEHICLE * 2):
                oldi = i % (pre_numVSs * 2)
                for k in range(len_next_layer):
                    model.layers[layer].get_weights(
                    )[0][i][k] = premodel.layers[layer].get_weights()[0][oldi][k]

            # for i in range(pre_numVSs * 2, NUM_STATE, 1):
            #     # model.layers[layer].get_weights(
            #     # )[1][i] = premodel.layers[layer].get_weights()[1][i - 10]
            #     for k in range(len(model.layers[layer].get_weights()[0][i])):
            #         model.layers[layer].get_weights(
            #         )[0][i][k] = premodel.layers[layer].get_weights()[0][i - increase_numVSs * 2][k]

            for i in range(NUM_VEHICLE * 2, NUM_STATE, 1):
                # model.layers[layer].get_weights(
                # )[1][i] = premodel.layers[layer].get_weights()[1][i - 10]
                oldi = i - NUM_VEHICLE * 2 + pre_numVSs * 2
                for k in range(len_next_layer):
                    model.layers[layer].get_weights(
                    )[0][i][k] = premodel.layers[layer].get_weights()[0][oldi][k]

        model.layers[3].set_weights(premodel.layers[3].get_weights())
        # model.layers[3].trainable = False

        for layer in range(4, 5):
            # for i in range(0, pre_numVSs + 1, 1):
            #     model.layers[layer].get_weights(
            #     )[1][i] = premodel.layers[layer].get_weights()[1][i]
            #     for j in range(len(model.layers[layer].get_weights()[0])):
            #         model.layers[layer].get_weights(
            #         )[0][j][i] = premodel.layers[layer].get_weights()[0][j][i]

            len_layer = len(model.layers[layer].get_weights()[0])
            model.layers[layer].get_weights(
            )[1][0] = premodel.layers[layer].get_weights()[1][0]
            for j in range(len_layer):
                model.layers[layer].get_weights(
                )[0][j][0] = premodel.layers[layer].get_weights()[0][j][0]
            for i in range(1, NUM_VEHICLE + 1, 1):
                oldi = i % pre_numVSs + 1
                model.layers[layer].get_weights(
                )[1][i] = premodel.layers[layer].get_weights()[1][oldi]
                for j in range(len_layer):
                    model.layers[layer].get_weights(
                    )[0][j][i] = premodel.layers[layer].get_weights()[0][j][oldi]

            # for i in range(pre_numVSs + 1, NUM_ACTION, 1):
            #     model.layers[layer].get_weights(
            #     )[1][i] = premodel.layers[layer].get_weights()[1][i - increase_numVSs]
            #     for j in range(len(model.layers[layer].get_weights()[0])):
            #         model.layers[layer].get_weights(
            #         )[0][j][i] = premodel.layers[layer].get_weights()[0][j][i - increase_numVSs]

            for i in range(NUM_VEHICLE + 1, NUM_ACTION, 1):
                oldi = i - NUM_VEHICLE + pre_numVSs
                model.layers[layer].get_weights(
                )[1][i] = premodel.layers[layer].get_weights()[1][oldi]
                for j in range(len_layer):
                    model.layers[layer].get_weights(
                    )[0][j][i] = premodel.layers[layer].get_weights()[0][j][oldi]

        if type == 2:
            model.layers[3].trainable = False
        elif type == 3:
            model.layers[2].trainable = False
            model.layers[4].trainable = False

    num_actions = NUM_ACTION
    # policy = EpsLinearDecreaseQPolicy(maxeps = 1, mineps = 0, subtrahend = 0.00001)
    policy = EpsGreedyHardDecreasedQPolicy(eps=.0, decreased_quantity=.005,
                                           nb_hard_decreased_steps=NUM_TASKS_PER_TIME_SLOT)
    # policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = MixStateEnv()
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)

    dqn = ExpectedTaskDQN(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
                          batch_size=32, target_model_update=target_model_update, policy=policy, gamma=gamma, train_interval=3,
                          enable_double_dqn=True, enable_dueling_network=True)
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
        dqn.fit(env, nb_steps=NUM_TASKS_PER_TIME_SLOT *
                41, visualize=False, verbose=2)
        dqn.policy = EpsGreedyQPolicy(0.0)
        endtrain = time.time()
        env.setTest()
        dqn.test(env, nb_episodes=31)
        endtest = time.time()
        timedict = {'train': endtrain - begin, 'test': endtest - endtrain}
        with open(f'{RESULT_DIR}/{folder_name}/time.json', 'w') as fp:
            json.dump(timedict, fp)
        model.save(f'{RESULT_DIR}/{folder_name}/model.h5')
    except Exception as e:
        print(e)


def getNameApproach(type=0):
    name = None
    if type == 0:
        name = 'innocence'
    elif type == 1:
        name = 'all'
    elif type == 2:
        name = 'mid'
    elif type == 3:
        name = 'side'
    elif type == 4:
        name = 'no_training'
    return name


if __name__ == "__main__":
    # type = 3
    # for i in range(1, 2):
    #     try:
    #         Run_ExpectedTaskDDQN(f"transfer/WithDueling/{pre_numVSs}_{NUM_VEHICLE}/{name}/{i}", gamma=0.995,
    #                              target_model_update=0.05, type=type)
    #         # Run_ExpectedTaskDDQN("test/transfer/" + str(i), gamma=0.995,
    #         #                      target_model_update=1e-2)
    #     except Exception as e:
    #         print(e)

    for type in range(4, 5):
        for i in range(1, 2):
            name = getNameApproach(type)
            pre_numVSs = 15
            try:
                run_Transfer_AODAI(f"transfer/WithDueling/{pre_numVSs}_{NUM_VEHICLE}/{name}/{i}", gamma=0.995,
                                     target_model_update=0.05, pre_numVSs = pre_numVSs, type=type)
                # Run_ExpectedTaskDDQN("test/transfer/" + str(i), gamma=0.995,
                #                      target_model_update=1e-2)
            except Exception as e:
                print(e)
