from RLagent.Agent_base import Agent 
import torch
import numpy as np
from exploration_policy.epsilon_greedy import EpsilonGreedyAndArgMax
from model.DQNModel import DQNetwork
import torch.nn.functional as F
import torch.optim as optim
import random

class DQN_agent(Agent):

    def __init__(self, variant, memory):
        self.variant = variant
        self.state_size = variant["state_dim"]
        self.action_size = variant["action_dim"]
        self.seed = random.seed(variant["SEED"])
        
        # Q-Network
        self.DQNetwork_local = DQNetwork(variant).to(self.variant["device"])
        self.DQNetwork_target = DQNetwork(variant).to(self.variant["device"])
        self.optimizer = optim.Adam(self.DQNetwork_local.parameters(), lr=variant["lr"])
        self.memory = memory
        self.time_step = 0
        self.loss =0
        self.loss_list =[]
        self.action = None
        self.eps = variant["eps_start"]

    
    def step(self, state, next_state, reward, done, time_step):
        # Save experience in replay memory
        self.memory.push(state, self.action, reward, next_state, done)

    def action_selection(self, state, memory, time_step):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.variant["device"])
        self.DQNetwork_local.eval()
        with torch.no_grad():
            action_values = self.DQNetwork_local(state)
        self.DQNetwork_local.train()
        action_values = self.DQNetwork_local.forward(state)

        self.action, self.eps= EpsilonGreedyAndArgMax(action_values, self.time_step ,self.eps, self.variant)
        return self.action
        
    def learn(self, memory, time_step):
        """Update value parameters using given batch of experience tuples.
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            
        """

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) < self.variant["INITIAL_BUFFERSIZE"]:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # compute Q_target from the target network inputing next_state
        Q_target_av = self.DQNetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + self.variant["gamma"]*(Q_target_av)*(1-dones) # if done, than the second will not be added
        # compute the Q_expected 
        Q_expected = self.DQNetwork_local(states).gather(1, actions) # get q value for corrosponding action along dimension 1 of 64,4 matrix
        
        #apply gradient descent
        #compute loss
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward() # since we detached the Q_target, it becomes a constant and the gradients wrt Q_expected is computed only
        self.optimizer.step() # update weights

        #  update target network #
        if self.time_step % self.variant["update_timesteps"] ==0:
            self.soft_update(self.DQNetwork_local, self.DQNetwork_target, self.variant["TAU"])                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = tau*θ_local + (1 - tau)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data=(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self):
        torch.save(self.DQNetwork_local.state_dict(), './data/DQN_{}_agent.pt'.format(self.variant["env_name"]))
    def load(self):
        pass