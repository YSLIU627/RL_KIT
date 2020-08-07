import sys
sys.path.append(".")
from RLframe.Frame import RL_ALG

from data_mangament.Base_BufferReplay import BufferReplay 
from RLagent.DQNagent import DQN_agent
import torch
import gym

def experiment(variant):
    env = gym.make(variant["env_name"])
    variant["state_dim"] = env.observation_space.shape[0]
    variant["action_dim"] = env.action_space.n
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    variant["device"] = device
    memory= BufferReplay(variant)
    agent = DQN_agent(variant, memory)
    RL_ALG(env, agent, memory, variant)
    


if __name__ == "__main__":
    variant = dict(
        env_name = 'LunarLander-v2',
        max_episodes = 800,
        max_timesteps= 1000,
        render = False,
        log_episodes = 20,
        save_episodes = 100,
        gamma = 0.99,
        Logger= False,
        SEED = 0,
        solved_reward = 200,

        betas = (0.9, 0.999),
        lr = 0.0005,
        ALGORITHM = 'DQN',
        learn_timestep = 4,
        update_timesteps = 1,
        INITIAL_BUFFER = True,
        BUFFER_SIZE = int(1e5),
        INITIAL_BUFFERSIZE = int(1e3),
        TAU = 2e-3,
        BATCH_SIZE = 64,
        #SOFT_UPDATE = True,

        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995

    
    )
    experiment(variant)