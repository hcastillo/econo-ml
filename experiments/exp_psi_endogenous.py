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
    MC = 10

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "c:\\experiments\\0psi_endogenous"
    COMPARING_DATA = "c:\\experiments\\0psi_fixed" # "c:\\experiments\\psi_fixed0.63"
    COMPARING_LABEL = "psi_fixed"  # "psi_fixed=0.63"

    NAME_OF_X_SERIES = "psi_endogenous"

    parameters = {
        "p": np.linspace(0.0001, 1, num=10)
    }


    EXTRA_MODEL_CONFIGURATION = { 'psi_endogenous':True }
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(MarketPowerRun)

