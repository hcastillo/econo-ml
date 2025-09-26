#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank_lenderchange import ShockedMarket
import exp_runner


class MarketPowerRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 50

    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "c:\\experiments\\marketpower"
    COMPARING_DATA = "not_valid"
    COMPARING_LABEL = None

    parameters = {
        "p": [0.99]
    }
    
    config = {
        "psi": np.linspace(0.0, 0.99, num=10),
    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner.Runner(MarketPowerRun)
    runner.do()

