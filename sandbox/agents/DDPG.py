import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sandbox.utils.ddpg_utils import *


class DDPGActor(nn.Module):
    '''
    Pytorch neural network for Actor model
    '''

    def __init__(self, state_dim, action_dim, action_bound, hidden_size, init_w=3e-3):
        super(DDPGActor, self).__init__()
        self.action_bound = action_bound
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)
        self.init_weights(init_w)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def init_weights(self, init_w):
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        x = torch.tanh(self.l3(x))
        x = x*self.action_bound

        return x


class DDPGCritic(nn.Module):
    '''
    Pytorch neural network for critic model
    '''

    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3):
        super(DDPGCritic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.l2 = nn.Linear(hidden_size+action_dim, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)
        self.init_weights(init_w)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def init_weights(self, init_w):
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(torch.cat([x,a],1))))
        x = self.l3(x)

        return x


class DDPG:
    '''
    Implementation of Deep Deterministic Policy Gradient according to 
    https://arxiv.org/pdf/1509.02971.pdf
    '''

    def __init__(self, state_dim, action_dim, action_high,
                action_low, hidden_size, lr_actor, 
                lr_critic, tau, gamma,eps, decay_eps, 
                batch_size, max_mem_size, load=False, epoch=0):

        self.name = "DDPG Agent"   # Used for logging purposes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high 
        self.action_low = action_low

        self.actor = DDPGActor(state_dim, action_dim, action_high, hidden_size)
        self.actor_target = DDPGActor(state_dim, action_dim, action_high, hidden_size)
        self.critic = DDPGCritic(state_dim, action_dim, hidden_size)
        self.critic_target = DDPGCritic(state_dim, action_dim, hidden_size)
        
        if load:
                print("Loading Agent and critic....")
                self.actor.load_state_dict(torch.load("./weights/model_{}.pt".format(epoch)))
                self.actor_target.load_state_dict(torch.load("./weights/model_actor_target_{}.pt".format(epoch)))

                self.critic.load_state_dict(torch.load("./weights/model_critic_{}.pt".format(epoch)))
                self.critic_target.load_state_dict(torch.load("./weights/model_critic_target_{}.pt".format(epoch)))
                print(f"Loaded the networks..")


        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.criterion = nn.MSELoss()

        if load == False:
            hard_update(self.actor_target, self.actor)
            hard_update(self.critic_target, self.critic)

        self.max_mem_size = max_mem_size
        self.memory = ReplayBuffer(max_mem_size)

        self.random_process = OUActionNoise(mu=np.zeros(action_dim), sigma=0.05 * self.action_high)   # Generate random noise centered around zero.

        self.tau = tau
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.eps = eps 
        self.decay_eps = decay_eps

        self.s_t = None
        self.a_t = None
        self.is_training = True



    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.store_transition(self.s_t, self.a_t, r_t, s_t1, done)
            self.s_t = s_t1


    def select_action(self, state):
        self.actor.eval()
        action = self.actor(to_tensor(state).to(self.actor.device)).to('cpu').detach().numpy()
        action += self.is_training*max(self.eps, 0)*self.random_process()
        action = np.clip(action, self.action_low, self.action_high)

        self.eps -= self.decay_eps
        self.actor.train()
        self.a_t = action
        return action


    def random_action(self):
        action = np.random.uniform(self.action_low, self.action_high,\
                                    self.action_dim)
        self.a_t = action
        return action


    def update_parameters(self):
        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, \
        next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = to_tensor(state_batch).to(device)
        action_batch = to_tensor(action_batch).to(device)
        reward_batch = to_tensor(reward_batch).to(device)
        next_state_batch = to_tensor(next_state_batch).to(device)
        done_batch = abs(to_tensor(done_batch) - 1).to(device)   # Need to switch boolean

        # Calculate next q-values
        with torch.no_grad():
            q_next = self.critic_target([next_state_batch, \
                         self.actor_target(next_state_batch)])

            target_q_batch = reward_batch + \
                self.gamma*q_next*done_batch


        # Critic update
        self.critic.zero_grad()
        self.critic.train()

        q_batch = self.critic([state_batch, action_batch])
        critic_loss = self.criterion(q_batch, target_q_batch)
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update 
        self.critic.eval()
        self.actor.zero_grad()
        self.actor.train()

        actor_loss = self.critic([
            state_batch,
            self.actor(state_batch)
        ])

        actor_loss = -actor_loss.mean()
        actor_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)        
  
