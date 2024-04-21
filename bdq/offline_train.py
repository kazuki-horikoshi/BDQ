import gym
import time
import os, sys
import warnings

subfolderName = 'bdq' # change the name!!! 
path_agent_parent_dir = '../'  
sys.path.append(path_agent_parent_dir + '../')
sys.path.append(os.path.dirname(subfolderName) + path_agent_parent_dir)
path_logs = path_agent_parent_dir + subfolderName + "/"

#import envs
from pyfmi import load_fmu
from bdq import deepq

# Set environment and number of training episodes
env_name = 'emulator-v0' ; total_num_episodes = 1095

def main():
    dueling = True # with dueling (best-performing)
    agg_method = 'reduceLocalMax' # naive, reduceLocalMax, reduceLocalMean (best-performing)   
    target_version = 'mean' # indep, max, mean (best-performing)
    losses_version = 2 # 1,2 (best-performing),3,4,5 
    num_actions_pad = 6 # numb discrete sub-actions per action dimension
    independent = False # only set to True for trying training without the shared network module (does not work well)
    
    env = gym.make(env_name)

    if dueling: duel_str = 'Dueling-' + agg_method + '_'
    else: 
        duel_str = '' 
        agg_method = None
    
    if not independent:
        method_name = '{}{}{}{}{}'.format('Branching_', duel_str, 'TD-target-{}_'.format(target_version), 'TD-errors-aggregation-v{}_'.format(losses_version), 'granularity-{}'.format(num_actions_pad))
    else: 
        method_name = '{}{}{}{}{}'.format('Independent_', duel_str, 'TD-target-{}'.format(target_version), 'TD-errors-aggregation-v{}_'.format(losses_version), 'granularity-{}'.format(num_actions_pad))
    
    time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S') 

    model = deepq.models.mlp_branching(
        #hiddens_common=[512, 256], 
        hiddens_common=[256, 128], 
        #hiddens_actions=[128], 
        hiddens_actions=[64],
        #hiddens_value=[128],
        hiddens_value=[64],
        independent=independent,
        num_action_branches=env.action_space.shape[0], # for continuous box space
        # num_action_branches=env.action_space.shape, # for multi discrete space
        dueling=dueling,
        aggregator=agg_method  
    )

    act = deepq.learn_continuous_tasks(
        env,
        q_func=model,
        env_name=env_name, 
        method_name=method_name,
        dir_path=os.path.abspath(path_logs),
        time_stamp=time_stamp,
        total_num_episodes=total_num_episodes,
        #lr=1e-4,
        lr=1e-3,
        #gamma=0.99,
        gamma=0.9,
        batch_size=64,
        #buffer_size=int(1e6),
        buffer_size=int(5e3),
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        #prioritized_replay_beta_iters=2e6,
        prioritized_replay_beta_iters=2e6,
        dueling=dueling,
        independent=independent,
        target_version=target_version,
        losses_version=losses_version,
        num_actions_pad=num_actions_pad,
        grad_norm_clipping=10,
        learning_starts=1000, 
        #target_network_update_freq=1000,
        target_network_update_freq=96*3,
        #train_freq=1,
        train_freq=1,
        initial_std=0.5,
        final_std=0.1,
        #timesteps_std=1e8,
        #timesteps_std=1e3, # used for linear schedule
        timesteps_std=1095, # used for linear schedule
        #eval_freq=50,
        eval_freq=14, #evaluate every other week
        #n_eval_episodes=30, 
        n_eval_episodes=7, #evaluate for one week period
        eval_std=0.0,
        num_cpu=14,
        print_freq=10, 
        callback=None,
        # below are added inputs
        eta=600, #Hyper Parameter for Balancsssing Comfort and Energy
        occStatus = [1,1,0,0,0,1,1,1], # occupancy status, 1 is for present and 0 is for non present
        results_version="9.9", #csv results version for different run
        truncate=True, #show data from only testing period if TRUE, else show data from whole learning period 
    )

    print('Saving model...')
    model_dir = '{}/trained_models/{}'.format(os.path.abspath(path_logs), env_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    act.save('{}/{}_{}_{}.pkl'.format(model_dir, method_name, time_stamp, str(env.occStatus)))
    print('Model saved to: {}_{}_{}.pkl'.format(method_name, time_stamp, str(env.occStatus)))
    print(time.strftime('%Y-%m-%d_%H-%M-%S')) 

if __name__ == '__main__':
    main()