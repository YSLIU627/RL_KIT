import torch
import os
import numpy as np
import pickle
import random
def RL_ALG(env, agent, memory, variant):
    # logging variables
    running_reward = 0
    record = []
    time_step = 0
    log_reward = 0 
    
    # initial buffer
    if ("INITIAL_BUFFER" in variant and variant["INITIAL_BUFFER"] ) :
        while len(memory) < variant["INITIAL_BUFFERSIZE"]:
            state = env.reset()
            for t in range(variant["max_timesteps"]):
                action = random.choice(np.arange(variant["action_dim"]))
                next_state, reward, done, _ = env.step(action)
                memory.push(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

    # training loop
    for i_episode in range(1, variant["max_episodes"]+1):
        state = env.reset()
        for t in range(variant["max_timesteps"]):
            time_step +=1
            
            action = agent.action_selection(state, memory, time_step)
            next_state, reward, done, _ = env.step(action)
            running_reward += reward
            # memorize
            agent.step(state,next_state, reward, done, time_step)

            state = next_state
            if time_step % variant["learn_timestep"] == 0:
                agent.learn(memory, time_step)
            
            if variant["render"]:
                env.render()
            if done:
                break
        record.append([time_step,running_reward])
        log_reward += running_reward
        running_reward = 0
        # save every 500 episodes
        if i_episode % variant["save_episodes"] == 0:
            agent.save()
            pickle.dump(record, open('./data/{}_{}_record.pickle'.format(variant["ALGORITHM"],variant["env_name"]), 'wb'))
            record.clear()
        if variant["Logger"] :
            SetupMylogger()    
        # logging
        if i_episode % variant["log_episodes"] == 0:
            print('Episode {} \t Step {}\taverage reward: {}'.format(i_episode,time_step, int(log_reward/variant["log_episodes"])))
            log_reward = 0
            if variant["Logger"] : 
                Mylogger(i_episode, time_step, log_reward)

def SetupMylogger():
    pass

def Mylogger(i_episode, time_step, log_reward):
    pass
            
