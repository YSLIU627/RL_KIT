import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
torch.set_default_tensor_type(torch.FloatTensor)


class Actor_CriticModel_Continuous(nn.Module):
    def __init__(self, variant):
        super(Actor_CriticModel_Continuous,self).__init__()
        # here activator tanh is adopted because action mean ranges -1 to 1
        # actor : output the mean of the policy
        self.actor =  nn.Sequential(
                nn.Linear(variant["state_dim"], 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, variant["action_dim"]),
                nn.Tanh()
                )
        # critic : estimate Advantage function 
        self.critic = nn.Sequential(
                nn.Linear(variant["state_dim"], 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.device = variant["device"]
        # define the variance of the policy (here we adopt Gaussian policy)
        self.action_var = torch.full((variant["action_dim"],), variant["action_std"]**2).to(self.device)
        
    def forward(self):
        raise NotImplementedError
    
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action,action_logprob
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
