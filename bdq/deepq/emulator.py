#Wrap the Modelica simulator into the Openai gym env

import gym
from gym import error
from gym.utils import closer


from pyfmi import load_fmu
import pandas as pd

import gym
from gym import spaces
space = spaces.Discrete(7) # Set with 7 elements {0, 1, 2, ..., 6}
x = space.sample()
assert space.contains(x)
assert space.n == 8


obs_name = ["time", "Tout", "HGloHor", "T_room", "elec", "CO2", "intGain", "m_flow", "cooPwr", "schedule", "Occupancy Flag", "temp setpoint"]
building_state_name = ["T_room"]
env_state_name = ["Tout", "HGloHor", "Occupancy Flag"]
'''
"Tout": [295.15,308.15] #22c-35c
"HGloHor": [0,1300]
"Occupancy Flag": [0,1]
"T_room" [295.15,302.15] #22c-29c
"f1": [0,5] #fan modes
"f2": [0,5]
'''
temp_action_space = [25.5,26.,26.5,27.,27.5,35.]
temp_action_space = [x+273.15 for x in action_space]

temp_action_space = [0,1,2,3,4,5]

env = load_fmu('win/OccTest_emulator_forRL_wind.fmu')
env.set("room.occDir", "C:/Users/bdgleiy.NUSSTF/Desktop/Test/occ.mos")
env.set("room.plugDir", "C:/Users/bdgleiy.NUSSTF/Desktop/Test/elec.mos")
env.set("room.weatherDir", "C:/Users/bdgleiy.NUSSTF/Desktop/Test/SGP_SINGAPORE-CHANGI-AP_486980_18.mos")



class emulator(object):
	"""docstring for emulator"""
	def __init__(self, emulator_path, occ_path, plug_path, weather_path, start, end, env_name):
		super(emulator, self).__init__()
		self.arg = arg
		self.action_space = spaces.Discrete(7)
		self.observation_space = spaces.Box(low=np.array([295.15, 0, 0, 295.15, 0, 0]), high=np.array([308.15, 1300, 1, 302.15, 5, 5]))
		self.reward_range = (-float('inf'), 0)
		self._env_name = env_name
		#Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
		# model path
		self.emulator_path = emulator_path
		self.occ_path = occ_path
		self.plug_path = plug_path
		self.weather_path = weather_path
		self.interval = 15
		# set_path(emulator_path, occ_path, plug_path, weather_path)




	# def set_path(self):
	# 	env = load_fmu(emulator_path)
	# 	env.set("room.occDir", occ_path)
	# 	env.set("room.plugDir", plug_path)
	# 	env.set("room.weatherDir", weather_path)


		
	def step(self, action): 
		'''
		Parameters
        ----------
        action: python list of float
        	Control actions that will be passed to the EnergyPlus
        Return: (float, [float], boolean)
            The index 0 is current_simulation_time in second, 
            index 1 is EnergyPlus results in 1-D python list requested by the 
            variable.cfg, index 2 is the boolean indicating whether the episode terminates.
        '''
        # play_one_step(env, state, epsilon, horizon, interval)
        action_K = setpoint(action) # convert action index into actual executable action in kelvin
        vAir1 = 0.9
        vAir2 = 0.3
        action_FMU = np.transpose(np.vstack([np.asarray([horizon]), np.asarray([action_K]), np.asarray([vAir1]), np.asarray([vAir2])]))
        #next_state, reward, done, info = env.step(action)
        opts = env.simulate_options()
        opts['initialize'] = False
        opts['ncp'] = 1
        res = env.simulate(start_time=horizon,final_time=horizon+60*interval,options = opts,input=(['TSP', 'vAir1', 'vAir2'],action_FMU))
        next_state, reward, done, info = to_result(res)

        # return next_state, reward, done, info, action_K
		return observation, reward, done, info


	def reset(self):
		# Create Modelica simulaton process
		self.env = load_fmu(emulator_path)
		env.set("room.occDir", occ_path)
		env.set("room.plugDir", plug_path)
		env.set("room.weatherDir", weather_path)

		# Warm up for 6.5 hours
		n = 60/interval

		opts = env.simulate_options()
		opts['initialize'] = True
		opts['ncp'] = n*6.5

		TSP = 26 + 273.15 # use 26C to initialize the emulator
		vAir1 = 0.9
		vAir2 = 0.3
		action_26 = np.transpose(np.vstack([np.asarray([0]), np.asarray([TSP]), np.asarray([vAir1]), np.asarray([vAir2])]))
		res = env.simulate(final_time=3600*6.5,options = opts,input=(['TSP', 'vAir1', 'vAir2'],action_26))

		start_time = pd.datetime(year = 2018, month = 1, day = 1)
		cur_time = start_time + pd.Timedelta(seconds = 6.5*3600)
		# print(cur_time)

		init_obs, _, _, _ = to_result(res)


		ret = []
		ret.append(curSimTim)
        ret.append(Dblist) ?????
        ret.append(is_terminal)
        return tuple(ret)

	def render(self, mode='human', close=False):
        pass;


	def _get_obs(self): pass


	def reset_model(self): pass


	def viewer_setup(self): pass

	#opts = env.simulate_options()

	def action_for_emulator(self, action):
		temp_act_space = [25.5,26.,26.5,27.,27.5,35.]
		temp_act_space = [x+273.15 for x in temp_act_space] # convert to K
		# find temperature setpoint for emulator
		temp_act_index = action[0]
		index = list(range(len(action_space)))
		action_dict = dict(zip(index, action_space))
		temp_stpt = action_dict[temp_act_index] # can be simplified!
		actions = [temp_stpt] + temp_act_space[1:]
		return actions


	def calcReward(occupancyFlag, energy, pmv1, pmv2):
		eta = 60000 # Hyper Parameter for Balancing Comfort and Energy
		reward = - (delta[int(occupancyFlag)] * (abs(pmv2) * eta)) - energy**2
		return reward

    def to_result(res): 
    	timeStep = res['time'][-1].astype(int)
    	next_state = []
    	next_state.append(res['room.TRooAir'][-1]) # building state
    	next_state.append(res['room.weaBus.TDryBul'][-1] - 273.15) # environment states
    	next_state.append(res['room.weaBus.HGloHor'][-1]) # environment states

    	hour = int(timeStep/3600) % 24
    	occupancyFlag = 1 if (hour>6 and hour<19) else 0
    	next_state.append(int(hour)) # environment states
    	next_state.append(int(occupancyFlag)) # environment states
    	# calc rewardsss
    	pmv1 = res['PMV1'][-1]
    	pmv2 = res['PMV2'][-1]
    	energy = res['cooPwr_{}min.y'.format(15)][-1]
    	reward = calcReward(occupancyFlag, energy, pmv1, pmv2)
    	done = 0 # double check!!!!!
    	info = to_observation(res) # other useful outputs
    	return np.array(next_state), reward, done, info


    @property
    def env_name(self):
    	"""
        Return the environment name.
        Return: string
        """
        return self.env_name



# https://gym.openai.com/docs/
# https://github.com/openai/gym/blob/master/gym/core.py
'''
env.action_space
env.observation_space
env.observation_space.high
env.observation_space.low
'''


