import gym
from gym import spaces
import numpy as np
import os, sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI environments

path_agent_parent_dir = '../' 
sys.path.append(path_agent_parent_dir + '../')
sys.path.append(os.path.dirname('bdq') + path_agent_parent_dir)
path_logs = path_agent_parent_dir + 'bdq/' 

import envs
from bdq import deepq

# Enter environment name and numb sub-actions per joint 
# env_name = 'Reacher6DOF-v0' ; num_actions_pad = 33 # ensure it's set correctly to the value used during training   
env_name = 'emulator-v0' ; num_actions_pad = 6 

# Morning
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-23_21-37-58_[0, 0, 0, 0, 0, 1, 1, 1].pkl" #1.12
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-23_21-37-47_[1, 0, 0, 0, 0, 1, 1, 1].pkl" #1.11
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-23_21-37-20_[0, 1, 0, 0, 0, 1, 1, 1].pkl" #1.10
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-23_21-37-07_[1, 1, 0, 0, 0, 1, 1, 1].pkl" #1.9

# Noon
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-24_00-26-53_[0, 0, 0, 0, 0, 0, 0, 1].pkl" #1.8
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-24_00-27-02_[1, 0, 0, 0, 0, 0, 0, 1].pkl" #1.7
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-24_00-27-32_[0, 1, 0, 0, 0, 0, 0, 1].pkl" #1.6
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-24_00-27-41_[1, 1, 0, 0, 0, 0, 0, 1].pkl" #1.5

# Afternoon
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-22_18-11-42_[0, 0, 0, 1, 1, 1, 1, 1].pkl" #1.4
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-22_18-11-23_[1, 0, 0, 1, 1, 1, 1, 1].pkl" #1.3
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-22_18-11-06_[0, 1, 0, 1, 1, 1, 1, 1].pkl" #1.2
model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-22_18-10-44_[1, 1, 0, 1, 1, 1, 1, 1].pkl" #1.1

# Transitions
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-24_02-58-02_[0, 0, 0, 1, 1, 0, 0, 1].pkl" #1.16
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-24_03-00-05_[1, 0, 0, 1, 1, 0, 0, 1].pkl" #1.15
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-24_03-00-53_[0, 1, 0, 1, 1, 0, 0, 1].pkl" #1.14
#model_file_name = "Branching_Dueling-reduceLocalMax_TD-target-mean_TD-errors-aggregation-v2_granularity-6_2021-10-24_03-01-00_[1, 1, 0, 1, 1, 0, 0, 1].pkl" #1.13


model_dir = '{}/trained_models/{}'.format(os.path.abspath(path_logs), env_name)

def extract_occ_status_from_filename(filename):
    """Extracts the occupancy status from the model filename."""
    start_idx = filename.rfind('[')
    end_idx = filename.rfind(']')
    occ_status_str = filename[start_idx+1:end_idx]
    occ_status = list(map(int, occ_status_str.split(', ')))
    return occ_status

################# run the trained RL agent on the emulator for some days ###############
episodes = 7 # one day is one episodes
steps = 24*4
def main():
    # Extract occ_status from the model file name
    occ_status = extract_occ_status_from_filename(model_file_name)

    env = gym.make(env_name)
    act = deepq.load("{}/{}".format(model_dir, model_file_name))
    
    #define the dimention of action, range of action and action stream
    num_action_dims = env.action_space.shape[0] 
    num_action_streams = num_action_dims
    num_actions = num_actions_pad*num_action_streams
    low = env.action_space.low 
    high = env.action_space.high 
    actions_range = np.subtract(high, low) 

    #Initialize the environment
    obs = env.reset(occ_status=occ_status,eta=1)
    start_time = pd.datetime(year = env.year, month = env.month, day = env.day)
    cur_time = start_time + pd.Timedelta(seconds = 7*3600)
    observations = [obs] # save for record
    timeStamp = [start_time]
    episode_rewards = []
    
    #Simulation loop
    for i in range(episodes):
        episode_rew = 0
        for t in range(steps):
            action_idxes = np.array(act(np.array(obs)[None], stochastic=False)) # use deterministic way to select actions
            horizon = int(i*24*3600 + t*3600/4 + 7*3600)
            new_obs, rew, done, info = env.step(action_idxes, horizon)
            cur_time = start_time + pd.Timedelta(seconds = info[0])
            observation = np.concatenate((new_obs, info, np.array([rew]+ action_idxes)))
            observations.append(observation)
            timeStamp.append(cur_time)
            obs = new_obs
            episode_rew += rew
            print ("episode:", i, "steps:", t)
            if done: break
        # print ('Mean episode reward: {}'.format(episode_rew/steps))
        episode_rewards.append(episode_rew)
    # visualize the rewards
    plt.plot(episode_rewards)
    plt.xlabel('episodes')
    plt.ylabel('daily rewards')
    plt.savefig('episode_rewards.png')  # Save the plot to a file
    print("Plot saved as episode_rewards.png")


################# run the trained RL agent for single step contral based on BMS data ###############
# define the upper and lower limits for valid observations
'''
    "T_room" [295.15,302.15] #22c-29c
    "f1": [0,5] #fan modes
    "f2": [0,5]
    "f3": [0,5]
    "f4": [0,5]
    "Tout": [295.15,308.15] #22c-35c
    "HGloHor": [0,1300]
    "RH": [0,100]
    "Hour": [0,23]
    "Occupancy Flag": [0,1]
'''

'''
observation_space = spaces.Box(low=np.array([24, 0, 0, 0, 0, 21.5, 0, 0.4, 0, 0]), high=np.array([29, 5, 5, 5, 5, 34.5, 1150, 1, 23, 1]))
# user input the following 10 observations from BMS, can do it better by automation!
# use 5-min avg value for the solar radiation 
T_room = 27.8
f1 = 3
f2 = 2
f3 = 3
f4 = 3
Tout = 30.5
HGloHor = 96.4
RH = 0.717
Hour = 17
OccupancyFlag = 1 if (Hour>7 and Hour<18) else 0 # =1 during occupied hours 8am to 6pm
'''

def _setpoint(action_index):
    # temp action space in C: 26.,26.5,27.,27.5,28.,35.
    temp_action_space = [299.15, 299.65, 300.15, 300.65, 301.15, 308.15] # for multi discreate action space 26 to 28
    return temp_action_space[int(action_index)]

def _fanSpeed(action_indexes):
    # fan action space in C: 0, 0.2, 0.4, 0.6, 0.8, 1.0
    fan_action_space = [0.06, 0.17, 0.34, 0.4, 0.59, 0.71]
    fan_action = []
    for i in action_indexes:
        fan_action.append(fan_action_space[int(i)]) 
    return fan_action

'''
def main():
    # Assert if the input is valid obervation
    obs = np.array([T_room, f1, f2, f3, f4, Tout, HGloHor, RH, Hour, OccupancyFlag])
    assert observation_space.contains(obs)
    # normalize the obs
    low_obs = observation_space.low 
    high_obs = observation_space.high
    obs_range = np.subtract(high_obs, low_obs)
    obs_normal = (obs - low_obs) / obs_range # normalize obs to 0-1 scale
    # use trained RL agent to select action index
    act = deepq.load("{}/{}".format(model_dir, model_file_name))
    action_idxes = np.array(act(np.array(obs_normal)[None], stochastic=False)) # use deterministic way to select actions
    action_C = _setpoint(action_idxes[0]) - 273.15
    vAir_idx = action_idxes[1:]
    vAir = _fanSpeed(vAir_idx)
    #print ("TSP:", action_C, "fans", vAir_idx)
    print ("TSP:", action_C, "fan1: {}; fan2: {}; fan3: {}; fan4: {}.".format(vAir_idx[0],vAir_idx[1],vAir_idx[2],vAir_idx[3]))
'''

if __name__ == '__main__':
    main()