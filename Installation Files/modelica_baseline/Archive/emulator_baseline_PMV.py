#Wrap the Modelica simulator into the Openai gym env

import gym
from gym import error
from gym.utils import closer
from gym import spaces

from pyfmi import load_fmu

import pandas as pd
import numpy as np

from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
#from pythermalcomfort.utilities import met_typical_tasks
#from pythermalcomfort.utilities import clo_individual_garments

############################ Wrap Modelica FMU into Gym Environment ##########################

class emulator_baseline(gym.Env):
	"""modelica emulator"""
	def __init__(self, emulator_path, occ_path, plug_path, lgt_path, weather_path, year, month, day, env_name):
		super(emulator_baseline, self).__init__()
		'''
		"TSP" [298.65, 301.15]
		"vAir1" [0,1,2,3,4,5]
		"vAir2" [0,1,2,3,4,5]
		'''
		# An old version of OpenAI Gym's multi_discrete.py. (Was getting affected by Gym updates)
		# (https://github.com/openai/gym/blob/1fb81d4e3fb780ccf77fec731287ba07da35eb84/gym/spaces/multi_discrete.py)
		#self.action_space = spaces.MultiDiscrete([ [0,5], [0,5], [0,5] ])
		self.action_space = spaces.Box(low=np.array([-1.,-1.,-1.,-1.,-1.]), high=np.array([1.,1.,1.,1.,1.])) # make it consistent with default GYM environment

		# self.action_space = spaces.Box(low=np.array([298.65, 0, 0]), high=np.array([301.15, 5, 5]))
		'''
		"T_room" [295.15,302.15] #22c-29c
		"f1": [0,5] #fan modes
		"f2": [0,5]
		"f3": [0,5]
		"f4": [0,5]
		"Tout": [295.15,308.15] #22c-35c
		"HGloHor": [0,1300]
		"Hour": [0,23]
		"Occupancy Flag": [0,1]
		'''
		#self.observation_space = spaces.Box(low=np.array([295.15, 0, 0, 0, 0, 295.15, 0, 0, 0]), high=np.array([302.15, 5, 5, 5, 5, 308.15, 1300, 23, 1]))
		self.observation_space = spaces.Box(low=np.array([22, 0, 0, 0, 0, 22, 0, 0, 0]), high=np.array([29, 5, 5, 5, 5, 35, 1300, 23, 1]))
		#Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
		self.reward_range = (-float('inf'), 0)
		
		self.interval = 15 # control frequency

		self.emulator_path = emulator_path
		self.occ_path = occ_path
		self.plug_path = plug_path
		self.lgt_path = lgt_path
		self.weather_path = weather_path
		self.year = year
		self.month = month
		self.day = day
		self.eta = 4e3 # Hyper Parameter for Balancing Comfort and Energy
		self.fan_limit = np.array([3, 2, 3, 4, 4, 3, 3, 4]) # fan speed limits
		self.fan_penalty = 3 # double the PMV value when exceed the fan limits
		#self.env_name = env_name


	def step(self, action, horizon):
		# set the minimum outdoor air 10% during occupied hours
		act = False
		_, occupancyFlag = self._calcHour(horizon + 60*self.interval)
		if int(occupancyFlag) == 0:
			act = True
		action_K = self._setpoint(action[0]) # convert action index into actual executable action in kelvin
		vAir_idx = action[1:] # convert action index into actual fan speed
		#horizon = int(episode*24*3600 + step*3600/4 + 6.5*3600)
		#action_FMU = np.transpose(np.vstack([np.asarray([horizon]), np.asarray([action_K])]))
		action_FMU = np.transpose(np.vstack(np.asarray([horizon,action_K,act,0])))
		opts = self.ENV.simulate_options()
		opts['initialize'] = False
		opts['ncp'] = 1
		res = self.ENV.simulate(start_time=horizon,final_time=horizon+60*self.interval,options = opts,input=(['TSP','act','u_ext'],action_FMU))
		next_state, reward, done, info = self._to_result(res, vAir_idx)
		# return observation, reward, done, info
		# info include other useful outputs (time, mass flow, cooling power, PMVs)
		return next_state, reward, done, info


	def reset(self, startTime):
		# Create Modelica simulaton process
		self.ENV = load_fmu(self.emulator_path, log_level=4)
		self.ENV.set("room1.occDir", self.occ_path)
		self.ENV.set("room1.plugDir", self.plug_path)
		self.ENV.set("room1.lgtDir", self.lgt_path)
		self.ENV.set("room1.weatherDir", self.weather_path)

		# Warm up for 6.5 hours
		n = 60/self.interval

		opts = self.ENV.simulate_options()
		opts['initialize'] = True
		opts['ncp'] = n*6.5

		TSP = 26 + 273.15 # use 26C to initialize the emulator
		#action_26 = np.transpose(np.vstack([np.asarray([0]), np.asarray([TSP])]))
		action_26 = np.transpose(np.vstack(np.asarray([0, TSP, True, 0])))
		res = self.ENV.simulate(start_time = startTime, final_time=startTime+3600*6.5,options = opts,input=(['TSP','act','u_ext'],action_26))

		#start_time = pd.datetime(year = self.year, month = self.month, day = self.day)
		#cur_time = start_time + pd.Timedelta(seconds = 6.5*3600)
		# print(cur_time)
		init_obs, _, _, _ = self._to_result(res, vAir_idx = [0, 0, 0, 0])
		# return a tuple of initial obs (Troom, vAir1, vAir2, vAir3, vAir4, Tdry, Solar, hour, occupancyFsslag)
		return tuple(init_obs)


	def render(self, mode='human', close=False): pass


	def _get_obs(self): pass


	def reset_model(self): pass


	def viewer_setup(self): pass


	"""below are helper functions used in the emulator class"""
	# track the outputs
	def _to_observation(self, res, PMVs):
	    results = []
	    results.append(res['time'][-1].astype(int)) # timestep in second
	    results.append(res['m_flow_{}min.y'.format(15)][-1]*3600/1.225) # kg/s to m3/h
	    # deltaT = res['room1.weaBus.TDryBul'][-1] - 290.55
	    # results.append(deltaT*res['m_flow_{}min.y'.format(15)][-1]*60*self.interval/1.225) # manually calc energy
	    results.append(res['coo_Pwr_{}min.y'.format(15)][-1])
	    results.extend(PMVs)
	    
	    #results.append(res['room.internalLoad.sch.y[1]'][-1])
	    #results.append(res['room.CO2'][-1])
	    #results.append(res['room.roo.heaGai.QCon_flow'][-1]+res['room.roo.heaGai.QRad_flow'][-1])
	    return np.array(results)


	'''
	def _calcPMV(self, tdb, tr, vAir_idx, rh):
		clo_1 = 0.4
		clo_2 = 0.75
		clo_3 = 0.6
		clo_4 = 0.45
		clothes = [clo_1, clo_2, clo_3, clo_4]
		vAir = self._fanSpeed(vAir_idx)
		clo_v = list(zip(clothes, vAir))
		met = 1 # default met rate
		penalty = (np.asarray(vAir_idx) > self.fan_limit) * (self.fan_penalty - 1) + 1 # return an array with 2x penalty when vAir exceed the limit
		results = []
		for (clo, v) in clo_v:
			vr = v_relative(v=v, met=met) # relative air movement defined in Code
			pmv = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, standard="ASHRAE")
			results.append(pmv['pmv'])
		# return original PMVs and associated penalty array
		return results, penalty
	'''

	def _calcPMV(self, tdb, tr, vAir_idx, rh):
		clo_1 = 0.68
		clo_2 = 0.51
		clo_3 = 0.51
		clo_4 = 0.6
		clo_5 = 0.6
		clo_6 = 0.68
		clo_7 = 0.6
		clo_8 = 0.6
		clothes = [clo_1, clo_2, clo_3, clo_4, clo_5, clo_6, clo_7, clo_8]
		vAir_all_idx = [vAir_idx[0],vAir_idx[0],vAir_idx[0],vAir_idx[1],vAir_idx[1],vAir_idx[2],vAir_idx[2],vAir_idx[3]] # fan mode for each person
		vAir_all = self._fanSpeed(vAir_all_idx)
		clo_v = list(zip(clothes, vAir_all))
		met = 1 # default met rate
		penalty = (np.asarray(vAir_all_idx) > self.fan_limit) * (self.fan_penalty - 1) + 1 # return an array with 2x penalty when vAir exceed the limit
		results = []
		for (clo, v) in clo_v:
			vr = v_relative(v=v, met=met) # relative air movement defined in Code
			pmv = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, standard="ASHRAE")
			results.append(pmv['pmv'])
		# return original PMVs and associated penalty array
		return results, penalty


	def _calcReward(self, occupancyFlag, energy, PMVs):
		delta = [1, 10] # delta: Weight for comfort during unoccupied and occupied mode
		comfort_penalty = delta[int(occupancyFlag)] * (np.mean(np.abs(PMVs)) * self.eta)
		#energy_penalty = [10,1][int(occupancyFlag)] * energy**2
		energy_penalty = [10,1][int(occupancyFlag)] * energy # energy value is negative
		#return - comfort_penalty + energy_penalty
		return - comfort_penalty


	# given the time in second, calculate corresponding hour and occupancy flag
	def _calcHour(self, timeStep):
		hour = int(timeStep/3600) % 24
		occupancyFlag = 1 if (hour>7 and hour<19) else 0 # =1 during occupied hours
		return hour, occupancyFlag


	def _to_result(self, res, vAir_idx):
		timeStep = res['time'][-1].astype(int)
		next_state = []
		next_state.append(res['room1.roomT'][-1] - 273.15) # building state
		vAir = self._fanSpeed(vAir_idx)
		next_state.extend(vAir) # building state
		next_state.append(res['room1.weaBus.TDryBul'][-1] - 273.15) # environment states
		next_state.append(res['room1.weaBus.HGloHor'][-1]) # environment states
		hour, occupancyFlag = self._calcHour(timeStep)
		next_state.append(int(hour)) # environment states
		next_state.append(int(occupancyFlag)) # environment states
		tdb = res['room1.roomT'][-1] - 273.15 # dry bulb temp.
		tr = res['room1.roomT'][-1] - 273.15 # mean radiant temp.
		rh = 70 # relative humidity
		PMVs, penalty = self._calcPMV(tdb, tr, vAir_idx, rh)
		# deltaT = res['room1.weaBus.TDryBul'][-1] - 290.55 # 17.4C supply air temperature
		# energy = deltaT*res['m_flow_{}min.y'.format(15)][-1]*60*self.interval/1.225 # manual calculate energy
		energy = res['coo_Pwr_{}min.y'.format(15)][-1]
		# use penalized PMVs for calculating reward
		reward = self._calcReward(occupancyFlag, energy, PMVs * penalty) 
		done = 0 # double check!!!!! Maybe for whole weather file time step in second?
		info = self._to_observation(res, PMVs) # other useful outputs (time, air flow, cooling power, and PMVs)
		return np.array(next_state), reward, done, info


	# map outputs (action index) to actual setpoint
	def _setpoint(self, action_index):
		# temp action space in C: 26.,26.5,27.,27.5, 28., 35.
		temp_action_space = [299.15, 299.65, 300.15, 300.65, 301.15, 308.15] # for multi discreate action space
		#temp_action_space = [298.65, 299.15, 299.65, 300.15, 300.65, 301.15] # for box action space
		return temp_action_space[int(action_index)]

	# def setpoint(self, action_index):
	#     index = list(range(len(action_space)))
	#     action_dict = dict(zip(index, action_space)) 
	#     return action_dict[action_index]


	# map outputs (action index) to actual fan speed
	def _fanSpeed(self, action_indexes):
		# fan action space in C: 0, 0.2, 0.4, 0.6, 0.8, 1.0
		fan_action_space = [0.06, 0.17, 0.34, 0.4, 0.59, 0.71]
		fan_action = []
		for i in action_indexes:
			fan_action.append(fan_action_space[int(i)])	
		return fan_action

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


#action = np.transpose(np.vstack([np.asarray([horizon]), np.asarray([TSP]), np.asarray([True]), np.asarray([0])]))
#res = self.ENV.simulate(start_time=horizon,final_time=horizon+60*interval,options = opts,input=(['TSP','act','u_ext'],action))