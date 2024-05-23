#For baseline model with emulator99

import gym
import time
import os, sys
import warnings
import datetime

path_agent_parent_dir = '../'  
sys.path.append(path_agent_parent_dir + '../')
sys.path.append(os.path.dirname('baseline') + path_agent_parent_dir)
path_logs = path_agent_parent_dir + 'baseline/'

import envs
from pyfmi import load_fmu
import pandas as pd
import numpy as np

# Set environment and number of training episodes
env_name = 'emulator-v99'; days = 365

eta = 600
occStatus = [1, 1, 1, 1, 1, 1, 1, 1]
results_version = "270_v4"
dir_path = '../bdq-2/'

def main():
    
    env = gym.make(env_name)
    env.eta = eta
    env.occStatus = occStatus

    start_time = pd.datetime(year = 2020, month = 1, day = 1)
    start_timestep = 365*2*24*3600 # skip 2018 and 2019

    init_obs = env.reset(start_timestep)

    # calculate the time after 7 hours
    start_time = datetime.datetime(year=2020, month=1, day=1) + datetime.timedelta(seconds=start_timestep)

    cur_time = start_time + datetime.timedelta(hours=7)
    observations = [init_obs]  # save for record
    timeStamp = [start_time]
    daily_reward = []

    action_idxes = [2, 3, 3, 3, 3]  # baseline action 27C + fan mode 3

    steps = 96
    for day in range(days):
        if day == 0:
            pass
        else:
            daily_reward.append(reward_sum)
        reward_sum = 0
        for step in range(steps):
            horizon = int(day*24*3600 + step*3600/4 + 7*3600 + start_timestep)
            # Adjust the start time to match the model's current time
            new_obs, rew, done, info = env.step(action_idxes, horizon=horizon)
            cur_time = start_time + datetime.timedelta(seconds=info[0] - start_timestep)
            observation = np.concatenate((new_obs, info, np.array([rew] + action_idxes)))
            observations.append(observation)
            timeStamp.append(cur_time)
            reward_sum += rew
            print("episode:", day, "steps:", step)

    results_dir = "{}/results/{}".format(dir_path, env_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    rewards_save_name = "/daily_rewards_" + results_version + ".csv"
    rewards_df = pd.DataFrame({'episode_rewards': daily_reward})
    rewards_df.to_csv(results_dir + rewards_save_name)

    obs_save_name = "/observations_" + results_version + ".csv"
    obs_name = ["Room T", "vAir1", "vAir2", "vAir3", "vAir4", "Outdoor T", "Solar Radiation", "RH", "hour", "nextOccupancyFlag", "Time", "Air Flow", "Energy", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "comfort penalty", "Reward", "Act_T", "Act_F1", "Act_F2", "Act_F3", "Act_F4"]
    obs_df = pd.DataFrame(observations, index=np.array(timeStamp), columns=obs_name)
    obs_df.to_csv(results_dir + obs_save_name)

    print('Results saved...')

if __name__ == '__main__':
    main()
