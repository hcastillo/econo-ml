#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank_lenderchange import ShockedMarket3
import exp_runner
import pandas as pd
from progress.bar import Bar
import warnings
import scipy.stats

class MarketPowerRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 10

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/3_psiendog_nonorm"

    parameters = {
        "p": np.linspace(0.0001, 0.2, num=5),
    }

    config = {}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': True, 'normalize_interest_rate_max': 0}
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025



if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(MarketPowerRun)