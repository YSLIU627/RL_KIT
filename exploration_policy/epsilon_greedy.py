import numpy as np
import random

#from base.py import ExplorationPolicy

def EpsilonGreedyAndArgMax(action_values, time_step ,eps, variant):
    
    eps = max(variant["eps_end"], variant["eps_decay"]*eps)
    if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) , eps
    else:
            return random.choice(np.arange(variant["action_dim"])) ,eps
        