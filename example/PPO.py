import sys
sys.path.append(".")
from RLframe.Frame import RL_ALG

from data_mangament.Memory_for_OnPolicy import Memory_OnPolicy 
from RLagent.PPOagent import PPO_agent 
import torch
import gym

def experiment(variant):
    env = gym.make(variant["env_name"])
    variant["state_dim"] = env.observation_space.shape[0]
    variant["action_dim"] = env.action_space.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    variant["device"] = device
    memory= Memory_OnPolicy()
    agent = PPO_agent(variant, memory)
    RL_ALG(env, agent, memory, variant)
    


if __name__ == "__main__":
    variant = dict(
        env_name = "BipedalWalker-v3",
        max_episodes = 5000,
        max_timesteps= 1500,
        render = False,
        log_episodes = 10,
        save_episodes = 500,
        gamma = 0.99,
        Logger= False,

        solved_reward = 300,

        betas = (0.9, 0.999),
        lr = 0.0003,

        ALGORITHM = 'PPO',
        learn_timestep = 4000,
        update_timesteps = 1,
        K_epochs = 80,
        eps_clip = 0.2,
        action_std = 0.5

    
    )
    experiment(variant)