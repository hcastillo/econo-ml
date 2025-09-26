#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank_lenderchange import ShockedMarket3
import exp_runner_distributed

class MarketPowerRun(exp_runner_distributed.ExperimentRunDistributed):
    N = 50
    T = 100
    MC = 3

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/tmp/prueba4"

    parameters = {
        "p": np.linspace(0.0001, 0.2, num=5),
    }

    config = { }

    EXTRA_MODEL_CONFIGURATION = { 'psi_endogenous':True }
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025



if __name__ == "__main__":
    runner = exp_runner_distributed.Runner(MarketPowerRun)
    runner.do()