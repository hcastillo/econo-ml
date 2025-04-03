#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank_lenderchange import Boltzmann
import exp_runner


class BoltzmannRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 5

    OUTPUT_DIRECTORY = "../experiments/boltzmann"
    ALGORITHM = Boltzmann
    COMPARING_DATA = None

    parameters = {  # items should be iterable:
        "m": np.linspace(1, 1, num=1),
    }

    LENGTH_FILENAME_PARAMETER = 2
    LENGTH_FILENAME_CONFIG = 0


if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(BoltzmannRun)
