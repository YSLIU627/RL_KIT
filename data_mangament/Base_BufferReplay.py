import numpy as np
from collections import deque,namedtuple
import torch
import random

class BufferReplay():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, variant):
        """Initialize a ReplayBuffer object."""
        self.variant = variant
        self.action_size = variant["action_dim"]
        self.memory = deque(maxlen=variant["BUFFER_SIZE"])  
        self.batch_size = variant["BATCH_SIZE"]
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(variant["SEED"])
    
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.variant["device"])
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.variant["device"])
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.variant["device"])
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.variant["device"])
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.variant["device"])
  
        return (states, actions, rewards, next_states, dones)
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    