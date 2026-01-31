#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank_lenderchange import ShockedMarket3
import exp_runner


class MarketPowerRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 15

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/2/psiendog"

    parameters = {
        "p": [0.0001, 0.001, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
    }

    config = {}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': True, 'normalize_interest_rate_max': -2}
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025



if __name__ == "__main__":
    runner = exp_runner.Runner(MarketPowerRun)
    runner.do()