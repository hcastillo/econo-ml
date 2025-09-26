#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank_lenderchange import SmallWorld
import exp_runner


class SmallWorldRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 50

    OUTPUT_DIRECTORY = "c:\\experiments\\smallworld"
    ALGORITHM = SmallWorld

    parameters = {  # items should be iterable:
        "p": np.linspace(0.001, 0.100, num=200),
    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    SEED_FOR_EXECUTION = 88993

if __name__ == "__main__":
    runner = exp_runner.Runner(SmallWorldRun)
    runner.do()
