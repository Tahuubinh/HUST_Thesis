# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:25:42 2022

@author: MrBinh
"""
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

def getRateTransData(channel_banwidth, pr, distance, path_loss_exponent, sigmasquare):
    return (channel_banwidth * np.log2(
            1 + pr / np.power(distance,path_loss_exponent) / sigmasquare
        )
    ) 

# Reinforce
# def running_mean(x, N=50):
#     kernel = np.ones(N)
#     conv_len = x.shape[0]-N
#     y = np.zeros(conv_len)
#     for i in range(conv_len):
#         y[i] = kernel @ x[i:i+N]
#         y[i] /= N
#     return y

# def discount_rewards(rewards, gamma=0.99):
#     lenr = len(rewards)
#     disc_return = torch.pow(gamma,torch.arange(lenr).float()) * rewards
#     disc_return /= disc_return.max() #B
#     return disc_return
































