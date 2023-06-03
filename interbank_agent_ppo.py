# -*- coding: utf-8 -*-
"""
Agent to use for reinforce learning using Pytorch + stable baselines 3
The agent uses the mathematical model implemented in interbank.py

@author: hector@bith.net
@date:   05/2023
"""

import numpy as np
import gymnasium
from gymnasium import spaces
import random
import interbank


class InterbankPPO(gymnasium.Env):
    """
    using PPO as model, execute the interbank.Model().
    Considerations:
    - episodes_per_step = how many Model.t executions (Model.forward()) will be done in each step of the RL

    """
    episodes_per_step = 1

    # action_space.Discrete(3) -> returns 0,1 or 2 -> we translate it to [0,0.5,1]
    actions_translation = [0, 0.5, 1]

    environment = interbank.Model()

    def __init__(self, **config):
        if config:
            self.environment.configure(config)
        self.steps = 0
        self.done = False
        self.last_action = None
        # observation = [ir,cash]
        self.observation_space = spaces.Box(
            low=np.array( [   0.0, 0.0]),
            high=np.array([np.inf, 1.0]),
            shape=(2,),
            dtype=np.float64)

        # Allowed actions will be: ŋ = [0,0.5,1]
        self.action_space = spaces.Discrete(3)


    def define_log(self,log,logfile,modules):
        self.environment.log.define_log(log,logfile,modules)
    def __get_observations(self):
        return np.array([self.environment.get_current_liquidity(),
                         self.environment.get_current_interest_rate()])

    def reset(self, seed=40579):
        """
        Set to the initial state the Interbank.model
        """
        super().reset(seed=seed)
        self.environment.initialize()
        self.steps = 0
        self.done = False
        return self.__get_observations(), dict()

    def step(self, action):
        self.environment.forward()
        self.steps += 1
        self.last_action = action
        truncated = False
        info = {}
        for i in range(self.episodes_per_step):
            self.environment.set_policy_recommendation(self.actions_translation[action])
            self.environment.forward()
            observation = self.__get_observations()
            reward = self.environment.get_current_fitness()
            self.done = self.environment.t >= self.environment.config.T
            if self.done:
                truncated = i < self.episodes_per_step
                info = {"Info": "Truncated"} if truncated else {"Info": "Finish (t=T)"}
                break
        return observation, reward, self.done, truncated, info

    def close(self):
        self.environment.finish()

    def render(self, mode='human'):
        liquidity, interest_rate = self.__get_observations()
        fitness = self.environment.get_current_fitness()
        self.environment.set_policy_recommendation()
        print(f"t={self.environment.t}: ŋ={self.last_action} fitness.μ={fitness:5.2f} " +
              f"ƩC={liquidity:5.2f} interest=%{interest_rate:5.2}")
