# -*- coding: utf-8 -*-
"""
Agent to use for reinforce learning using Pytorch + stable baselines 3
The agent uses the mathematical model implemented in interbank.py

@author: hector@bith.net
@date:   05/2023
"""
import numpy as np
#import gymnasium
# from gymnasium import spaces
import gym
import interbank


class InterbankPPO(gym.Env):
    """
    using PPO as model, execute the interbank.Model().
    """

    environment = interbank.Model()

    export_datafile = None
    export_description = None

    current_liquidity = ()
    current_ir = ()
    current_fitness = 0

    def __init__(self, **config):
        if config:
            self.environment.configure(**config)
        self.steps = 0
        self.done = False
        self.last_action = None
        # observation = [liq_max,liq_min,liq_avg,r_max,r_min,r_avg]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([7000, 1800, 3100, 2.55, 0.01, 0.18]),
            shape=(6,),
            dtype=np.float64)

        # Allowed actions will be: ŋ = [0,0.5,1]
        self.action_space = gym.spaces.discrete.Discrete(3)  # spaces.Discrete(3)
        gym.Env.__init__(self)

    def get_last_action(self):
        return self.last_action

    def define_savefile(self, export_datafile=None, export_description=None):
        self.export_description = export_description
        self.export_datafile = export_datafile

    def define_log(self, log: str, logfile: str = '', modules: str = '', script_name: str = ''):
        self.environment.log
        self.environment.log.define_log(log, logfile, modules, script_name)

    def __get_observations(self):
        self.current_fitness = self.environment.get_current_fitness()
        self.current_ir = self.environment.get_current_interest_rate_info()
        self.current_liquidity = self.environment.get_current_liquidity_info()
        return np.array(self.current_liquidity + self.current_ir)

    def reset(self, seed=None, options=None):
        """
        Set to the initial state the Interbank.Model
        """
        super().reset(seed=seed)
        self.environment.initialize(seed)
        self.steps = 0
        self.done = False
        return self.__get_observations(), {}

    def step(self, action):
        self.steps += 1
        self.environment.set_policy_recommendation(action)
        self.last_action = self.environment.ŋ
        self.environment.forward()
        observation = self.__get_observations()

        self.done = self.environment.t >= self.environment.config.T
        return observation, self.current_fitness, self.done, False, {}

    def close(self):
        self.environment.finish(self.export_datafile, self.export_description)

    def render(self, mode='human'):
        print(f"{type(self).__name__} t={self.environment.t - 1:3}: ŋ={self.get_last_action():3} " +
              f"avg.μ={self.current_fitness:5.2f} ƩC={self.current_liquidity[2]:10.2f} " +
              f"avg.ir=%{self.current_ir[2]:5.2} reward={self.current_fitness}")

