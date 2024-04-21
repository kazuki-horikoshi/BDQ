#Wrap the Modelica simulator into the Openai gym env

import gym
from gym import error
from gym.utils import closer
from gym import spaces

from pyfmi import load_fmu

import pandas as pd
import numpy as np

#from pythermalcomfort.models import pmv_ppd
#from pythermalcomfort.utilities import v_relative, clo_dynamic

############################ Wrap Modelica FMU into Gym Environment ##########################

class emulator(gym.Env):
	"""modelica emulator"""
	def __init__(self, emulator_path, occ_path, plug_path, lgt_path, weather_path, year, month, day, env_name):
		super(emulator, self).__init__()
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
		"T_room" [24,29]C
		"f1": [0,5] #fan modes
		"f2": [0,5]
		"f3": [0,5]
		"f4": [0,5]
		"Tout": [21.5,34.5]C
		"HGloHor": [0,1150]
		"RH": [0.4,1] x100%
		"Hour": [0,23]
		"Occupancy Flag": [0,1]
		'''
		#self.observation_space = spaces.Box(low=np.array([295.15, 0, 0, 0, 0, 295.15, 0, 0, 0]), high=np.array([302.15, 5, 5, 5, 5, 308.15, 1300, 23, 1]))
		self.observation_space = spaces.Box(low=np.array([24, 0, 0, 0, 0, 21.5, 0, 0.4, 0, 0]), high=np.array([29, 5, 5, 5, 5, 34.5, 1150, 1, 23, 1]))
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
		self.pathInitial = "ThermalComfort/Initialization/PP" # path to initial comfort profile
		self.dataInitial = self._readProfile(self.pathInitial) # read initial comfort profile
		
		self.eta = None # Hyper Parameter for Balancing Comfort and Energy
		self.occStatus = None # occupancy status, 1 is for present and 0 is for non present
		self.alpha = 2 # alpha: penalize large discomfort values
		self.beta = 0.4 # beta: handle integer feedback value, to make the difference in comfort penalties smaller
		#self.env_name = env_name


	def step(self, action, horizon):
		# set the minimum outdoor air 10% during occupied hours
		act = False
		_, occupancyFlag = self._calcHour(horizon + 60*self.interval)
		if int(occupancyFlag) == 0 and int(action[0]) == 5:
			act = True
		action_K = self._setpoint(action[0]) # convert action index into actual executable action in kelvin
		vAir_idx = action[1:] # convert action index into actual fan speed
		#horizon = int(episode*24*3600 + step*3600/4 + 7*3600)
		#action_FMU = np.transpose(np.vstack([np.asarray([horizon]), np.asarray([action_K])]))
		action_FMU = np.transpose(np.vstack(np.asarray([horizon,action_K,act,0])))
		opts = self.ENV.simulate_options()
		opts['initialize'] = False
		opts['ncp'] = 1
		res = self.ENV.simulate(start_time=horizon,final_time=horizon+60*self.interval,options = opts,input=(['TSP','act','u_ext'],action_FMU))
		next_state, reward, done, info = self._to_result(res, vAir_idx)
		# return observation, reward, done, info
		# info include other useful outputs (time, air flow, cooling power, comfortPenalty)
		return next_state, reward, done, info


	def reset(self):
		# Create Modelica simulaton process
		self.ENV = load_fmu(self.emulator_path, log_level=4)
		self.ENV.set("room1.occDir", self.occ_path)
		self.ENV.set("room1.plugDir", self.plug_path)
		self.ENV.set("room1.lgtDir", self.lgt_path)
		self.ENV.set("room1.weatherDir", self.weather_path)

		# Warm up for 7 hours
		n = 60/self.interval

		opts = self.ENV.simulate_options()
		opts['initialize'] = True
		opts['ncp'] = n*7

		TSP = 26 + 273.15 # use 26C to initialize the emulator
		#action_26 = np.transpose(np.vstack([np.asarray([0]), np.asarray([TSP])]))
		action_26 = np.transpose(np.vstack(np.asarray([0, TSP, True, 0])))
		res = self.ENV.simulate(final_time=3600*7,options = opts,input=(['TSP','act','u_ext'],action_26))

		start_time = pd.datetime(year = self.year, month = self.month, day = self.day)
		cur_time = start_time + pd.Timedelta(seconds = 7*3600)
		# print(cur_time)
		init_obs, _, _, _ = self._to_result(res, vAir_idx = [0, 0, 0, 0])
		# return a tuple of initial obs (Troom, vAir1, vAir2, vAir3, vAir4, Tdry, Solar, hour, nextOccupancyFlag)
		return tuple(init_obs)


	def render(self, mode='human', close=False): pass


	def _get_obs(self): pass


	def reset_model(self): pass


	def viewer_setup(self): pass


	"""below are helper functions used in the emulator class"""
	# track the outputs
	def _to_observation(self, res, comfortPenalty, avgComfortPenalty, occupancyFlag):
	    results = []
	    results.append(res['time'][-1].astype(int)) # timestep in second
	    results.append(res['m_flow_{}min.y'.format(15)][-1]*3600/1.225) # kg/s to m3/h
	    # deltaT = res['room1.weaBus.TDryBul'][-1] - 290.55
	    # results.append(deltaT*res['m_flow_{}min.y'.format(15)][-1]*60*self.interval/1.225) # manually calc energy
	    results.append(res['coo_Pwr_{}min.y'.format(15)][-1])
	    results.extend(comfortPenalty)
	    # also save the penalized PMV for record
	    comfort_penalty = [1, 10][int(occupancyFlag)] * avgComfortPenalty
	    results.append(comfort_penalty)
	    
	    #results.append(res['room.internalLoad.sch.y[1]'][-1])
	    #results.append(res['room.CO2'][-1])
	    #results.append(res['room.roo.heaGai.QCon_flow'][-1]+res['room.roo.heaGai.QRad_flow'][-1])
	    return np.array(results)

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


	'''



	# helper fcn to add labels for comfort profile
	def _addLabel(self, data):
		degreeSign= u'\N{DEGREE SIGN}'
		#data.columns =[x+degreeSign+'C' for x in ['25.5','26','26.5','27','27.5']]
		data.index = ['Level '+i for i in ['0','1','2','3','4','5']]
		return data


	# helper fcn to read comfort profiles
	def _readProfile(self, path):
		occupant = 8
		profiles = []
		for i in range(occupant):
			profile_path = path + "{}.csv".format(i+1)
			profile = pd.read_csv(profile_path)
			profiles.append(self._addLabel(profile))
		return profiles


	def _roundOff(self, number):
		"""Round a number to the closest half integer and match the profile column name
		>>> round_of_rating(1.3)
		1.5
		>>> round_of_rating(2.6)
		2.5
		>>> round_of_rating(3.0)
		3
		>>> round_of_rating(4.1)
		4"""
		idx = str(round(number * 2) / 2)
		if idx[-2:] == '.0':
			idx = idx[:-2]
		return idx


	def _comfortPenalty(self, tdb, vAir_idxes, occupancyFlag):
		# pathInitial = "ThermalComfort/Initialization/PP"
		# dataInitial = self._readProfile(pathInitial)
		comfortPenalty = []
		t_idx = self._roundOff(tdb)
		for i in range(len(self.dataInitial)):
			if i < 3: vAir_idx = int(vAir_idxes[1]) # fan group 2
			elif i < 5: vAir_idx = int(vAir_idxes[0]) # fan group 1
			elif i < 7: vAir_idx = int(vAir_idxes[3]) # fan group 4
			else: vAir_idx = int(vAir_idxes[2]) # fan group 3
			try:
				penalty = self.dataInitial[i].iloc[vAir_idx][t_idx]
			except:
				penalty = -3 # outside the profile range
			comfortPenalty.append(penalty)
			# print (self.dataInitial[i],penalty,vAir_idx)
		# set zeros for non present occupants
		comfortPenalty = np.asarray(comfortPenalty) * np.asarray(self.occStatus)
		if np.sum(self.occStatus) == 0:
			meanPenalty = 0 # zero divider situation
		else:
			if occupancyFlag == 0:
				# meanPenalty = np.sum(comfortPenalty) / np.sum(self.occStatus) # use average
				meanPenalty = 0

			# order of alpha to penalize large discomfort (>=2) during occupied hours
			else:
				# meanPenalty = - np.sum(np.abs(comfortPenalty)**self.alpha) / np.sum(self.occStatus) - self.beta
				meanPenalty = - (np.sum(comfortPenalty) / np.sum(self.occStatus) - self.beta)**self.alpha
		# print(t_idx, comfortPenalty, np.mean(comfortPenalty))
		return comfortPenalty, meanPenalty


	def _calcReward(self, occupancyFlag, energy, comfortPenalty):
		delta = 10 # delta: Weight for comfort during unoccupied and occupied modes
		comfort_penalty = [1,delta][int(occupancyFlag)] * comfortPenalty
		energy_penalty = [delta,1][int(occupancyFlag)] * energy # energy value is negative
		return comfort_penalty * self.eta + energy_penalty


	# given the time in second, calculate corresponding hour and occupancy flag
	def _calcHour(self, timeStep):
		hour = int(timeStep/3600) % 24
		occupancyFlag = 1 if (hour>7 and hour<18) else 0 # =1 during occupied hours
		return hour, occupancyFlag


	def _to_result(self, res, vAir_idx):
		timeStep = res['time'][-1].astype(int)
		next_state = []
		next_state.append(res['room1.roomT'][-1] - 273.15) # building state, indoor air T
		# vAir = self._fanSpeed(vAir_idx)
		next_state.extend(vAir_idx) # building state, indoor air speed
		next_state.append(res['room1.weaBus.TDryBul'][-1] - 273.15) # environment states, outdoor air T
		next_state.append(res['room1.weaBus.HGloHor'][-1]) # environment states, solar radiation
		next_state.append(res['OARH.phi'][-1]) # environment states, RH
		hour, occupancyFlag = self._calcHour(timeStep)
		_, nextOccupancyFlag = self._calcHour(timeStep + 60*self.interval) # use occupancy flag at next timestep for state
		next_state.append(int(hour)) # environment states
		next_state.append(int(nextOccupancyFlag)) # environment states
		tdb = res['room1.roomT'][-1] - 273.15 # dry bulb temperature
		comfortPenalty, avgComfortPenalty = self._comfortPenalty(tdb, vAir_idx, occupancyFlag)
		# deltaT = res['room1.weaBus.TDryBul'][-1] - 290.55 # 17.4C supply air temperature
		# energy = deltaT*res['m_flow_{}min.y'.format(15)][-1]*60*self.interval/1.225 # manual calculate energy
		energy = res['coo_Pwr_{}min.y'.format(15)][-1]
		reward = self._calcReward(occupancyFlag, energy, avgComfortPenalty) 
		done = 0 # double check!!!!! Maybe for whole weather file time step in second?
		info = self._to_observation(res, comfortPenalty, avgComfortPenalty, occupancyFlag) # other useful outputs (time, air flow, cooling power, and comfortPenalty)
		return np.array(next_state), reward, done, info


	# map outputs (action index) to actual setpoint
	def _setpoint(self, action_index):
		# temp action space in C: 25.5,26.,26.5,27.,27.5,35.
		# temp_action_space = [298.65, 299.15, 299.65, 300.15, 300.65, 308.15] # for multi discreate action space
		temp_action_space = [299.15, 299.65, 300.15, 300.65, 301.15, 308.15] # for multi discreate action space 26 to 28?
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


	# Normalize the observation space to [0,1] scale
	def _normalizeObs(self, obs):
		low_obs = self.observation_space.low 
		high_obs = self.observation_space.high
		obs_range = np.subtract(high_obs, low_obs)
		return (obs - low_obs) / obs_range

	@property
	def env_name(self):
		"""
		Return the environment name.
		Return: string
		"""
		return self.env_name


# References
# https://gym.openai.com/docs/
# https://github.com/openai/gym/blob/master/gym/core.py
