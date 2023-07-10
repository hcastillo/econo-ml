# -*- coding: utf-8 -*-
"""
Agent to use for reinforce learning using Pytorch + stable baselines 3
The agent uses the mathematical model implemented in interbank.py

@author: hector@bith.net
@date:   05/2023
"""
import numpy as np
import gymnasium
import interbank


class InterbankAgent(gymnasium.Env):
    """
    using PPO as model, execute the interbank.Model().
    """

    interbank_model = interbank.Model()

    export_datafile = None
    export_description = None

    current_liquidity = ()
    current_ir = ()
    current_fitness = 0

    def __init__(self, **config):
        if config:
            self.interbank_model.configure(**config)
        self.steps = 0
        self.done = False
        self.last_action = None
        # observation = [liq_max,liq_min,liq_avg,r_max,r_min,r_avg]
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([7000, 1800, 3100, 2.55, 0.01, 0.18]),
            shape=(6,),
            dtype=np.float32)

        # Allowed actions will be: ŋ = [0.0,1.0]
        self.action_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32)  # spaces.Discrete(3)
        gymnasium.Env.__init__(self)

    def get_last_action(self):
        return self.last_action

    def define_savefile(self, export_datafile=None, export_description=None):
        self.export_description = export_description
        self.export_datafile = export_datafile

    def define_log(self, log: str, logfile: str = '', modules: str = '', script_name: str = ''):
        self.interbank_model.log.define_log(log, logfile, modules, script_name)

    def __get_observations(self):
        self.current_fitness = self.interbank_model.get_current_fitness()
        self.current_ir = self.interbank_model.get_current_interest_rate_info()
        self.current_liquidity = self.interbank_model.get_current_liquidity_info()
        return np.array((self.current_liquidity + self.current_ir))

    def reset(self, seed=None, options=None, dont_seed=False):
        """
        Set to the initial state the Interbank.Model
        """
        super().reset(seed=seed)
        self.interbank_model.initialize(seed=seed, dont_seed=dont_seed,
                                        export_datafile=self.export_datafile,
                                        export_description=self.export_description)
        self.interbank_model.limit_to_two_policies()
        self.steps = 0
        self.done = False
        return self.__get_observations(), {}

    def step(self, action):
        self.steps += 1
        self.interbank_model.set_policy_recommendation(ŋ1=action[0])
        self.last_action = self.interbank_model.ŋ
        self.interbank_model.forward()
        observation = self.__get_observations()

        self.done = self.interbank_model.t >= self.interbank_model.config.T
        return observation, self.current_fitness, self.done, False, {}

    def close(self):
        self.interbank_model.finish()

    def render(self, mode='human'):
        print(f"{type(self).__name__} t={self.interbank_model.t - 1:3}: ŋ={self.get_last_action():3} " +
              f"avg.μ={self.current_fitness:5.2f} ƩC={self.current_liquidity[2]:10.2f} " +
              f"avg.ir=%{self.current_ir[2]:5.2} reward={self.current_fitness}")

