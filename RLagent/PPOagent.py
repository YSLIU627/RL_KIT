from RLagent.Agent_base import Agent 
import torch
from model.Actor_CriticModel import Actor_CriticModel_Continuous
import torch.nn as nn
import torch.optim as optim
import numpy as np
torch.set_default_tensor_type(torch.FloatTensor)


class PPO_agent(Agent) :
    def __init__(self, variant, memory):
        self.variant = variant
        self.lr = variant["lr"]
        self.betas = variant["betas"]
        self.gamma = variant["gamma"]
        self.eps_clip = variant["eps_clip"]
        self.K_epochs = variant["K_epochs"]
        self.device = variant["device"]
        self.policy = Actor_CriticModel_Continuous(variant).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old = Actor_CriticModel_Continuous(variant).to(self.device)
        # initial 
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.memory = memory
        self.logprob = None
        self.action = None
        self.update_timesteps = variant["update_timesteps"]
        self.loss = 0
        
    def action_selection(self, state, memory, time_step):
        'select action according to the old policy'
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        self.action, self.logprob= self.policy_old.act(state, memory)
        action = self.action
        return action.detach().cpu().data.numpy().flatten()
    
    def step(self,state,next_state, reward, done, time_step):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        next_state = torch.FloatTensor(next_state.reshape(1, -1)).to(self.device)
        self.memory.rewards.append(reward)
        self.memory.dones.append(done)
        #done = torch.tensor(done).to(self.device)
        # Only reward is not tensor #reward = torch.tensor(reward).float().to(self.device)
        #self.memory.push (state, self.action, reward, next_state,done,self.logprob)
    def learn(self, memory,time_step):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        device = self.device
        for reward, done in zip(reversed(memory.rewards), reversed(memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device).float()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        self.loss =0 
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach()).float()

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            L_CPI = ratios * advantages
            surr = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(L_CPI, surr) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            self.loss += loss.mean().detach().cpu().numpy()
            loss.mean().backward()
            self.optimizer.step()
        self.loss /= self.K_epochs
        # Copy new weights into old policy:
        if time_step % self.update_timesteps ==0:
            self.update()
    def update(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear_memory()
    def save(self):
        torch.save(self.policy.state_dict(), './data/PPO_continuous_{}.pth'.format(self.variant["env_name"]))
    def load(self):
        pass
