# Utilities for DDPG Implementations

import os
import random
from collections import deque
import numpy as np
import copy

import torch


class ReplayBuffer:
    '''
    Implementation of basic replay buffer using deque.

    :param max_size: the max number of (s,a,r,s',d) stored in our buffer
    :param batch_size: the size of batches for training updates
    :param state: the original state of robot arm 
    :param action: action taken by robot arm
    :param reward: reward received from enviornment
    :param state_: resulting state_ from taking action from state
    :param done: whether the episode is terminated by this step or not
    '''
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
        self.max_size = max_size

    def store_transition(self, state, action, reward, state_, done):
        experience = (state, action, np.array([reward]), state_, np.array([done]))
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class OUActionNoise(object):
    '''
    Ornstein-Uhlenbeck process implementation sourced from:
    https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/DDPG/pytorch/lunar-lander/ddpg_torch.py
    '''
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

def soft_update(target, source, tau):
    '''
    Soft update of network weights
    '''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    '''
    Hard update of network weights
    '''
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def to_tensor(ndarray):
    '''
    Convert numpy array to tensor
    '''
    return torch.FloatTensor(ndarray)


def fanin_init(size, fanin=None):
    '''
    Initilise network weights
    '''
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)