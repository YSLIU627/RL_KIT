import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):

    def __init__(self, variant):
        
        super(DQNetwork, self).__init__()   
        self.seed = torch.manual_seed(variant["SEED"])
        self.fc1 = nn.Linear(variant["state_dim"], 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.out = nn.Linear(64, variant["action_dim"])
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_vals = self.out(x)
        return q_vals  
