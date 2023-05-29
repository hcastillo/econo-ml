# -*- coding: utf-8 -*-
"""
Agent to use for reinforce learning using Pytorch + stable baselines 3
The agent uses the mathematical model implemented in interbank.py

@author: hector@bith.net
@date:   05/2023
"""

import numpy as np
import gymnasium
from   gymnasium import spaces
import random
import interbank

class InterbankPPO(gymnasium.Env):

    def __init__(self, columns=3, rows=3):

        self.interbank = interbank.Model()
        self.columns = columns
        self.rows = rows
        self.tablero = np.zeros(9, np.uint8)
        self.done = False
        self.turno = 0

        # Allowed actions will be: ŋ = [0,0.5,1]
        self.action_space = spaces.Discrete(3)

        # Vamos a definir el entorno
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3,),
            dtype=np.uint8)

    def __next_interbank_step(self):
        colocada = 0
        #TODO
        return self.tablero

    def __determine_reward(self):
        # TODO
        return 0.5

    def reset(self, seed=0):
        """
        Importante: the observation must be a numpy array
        :return: (np.array)
        """
        self.tablero = np.zeros(9, np.uint8)
        observation = np.zeros(3, np.uint8)
        self.turno = 0
        self.done = False
        return observation, dict()

    def step(self, action):
        if self.tablero[action] != 0:
            observation = self.tablero
            self.done = True
            reward = -1
            info = {"Error": "Intento hacer trampa"}
            return observation, reward, self.done, False, info
        self.tablero[action] = 1
        self.turno += 1
        observation = self.tablero
        reward = self.__determine_reward() ##TODO
        if self.turno == 5:
            self.done = True
        info = {}
        if self.done == False:
            self.tablero = self.__next_interbank_step()
            observation = self.tablero
        return observation, reward, self.done, False, info

    def render(self, mode='human'):
        print("")
        print("turno: " + str(self.turno))
        for i in range(9):
            if self.tablero[i] == 0:
                print("   |", end="")
            elif self.tablero[i] == 1:
                print(" O |", end="")
            elif self.tablero[i] == 2:
                print(" X |", end="")
            if (i + 1) % 3 == 0:
                print("")
        print("")

