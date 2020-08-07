import torch
class Memory_OnPolicy():
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
    
    def push(self, state, action, reward,next_state,done, logprob):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.logprobs.append(logprob)

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]