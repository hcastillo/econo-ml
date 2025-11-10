#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
from interbank_lenderchange import ShockedMarket3
import exp_runner


class MarketPowerRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 5

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251110_ir_study_base_psi"
    COLOR = 'black'

    parameters = {
        "p": [0.2]
    }

    config = { "psi": [ 0, 0.25, 0.50, 0.75, 0.99 ]}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2 }
    
    LENGTH_FILENAME_PARAMETER = 0
    LENGTH_FILENAME_CONFIG = 6

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner.Runner(MarketPowerRun)
    runner.do()