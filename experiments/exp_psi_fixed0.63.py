#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket

We determine the average of psi for the endogenous execution c:\\experiments\\psi_endogenous
(psi_fixed_average) and also for each p (psi_fixed_by_p), to compare properly later the results

@author: hector@bith.net
"""
import numpy as np
from interbank_lenderchange import ShockedMarket
import exp_runner


class MarketPowerRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 40

    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "c:\\experiments\\psi_fixed0.63"
    COMPARING_DATA = "not_exists"
    COMPARING_LABEL = None

    EXTRA_MODEL_CONFIGURATION = { 'psi_endogenous':False, 'psi': 0.6374517779471452 }

    parameters = {
        "p": np.linspace(0.0001, 1, num=10)
    }


    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(MarketPowerRun)

