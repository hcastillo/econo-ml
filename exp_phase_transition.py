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
    MC = 40

    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "c:\\experiments\\5_phi_phase_transition"

    parameters = {
        "p": np.linspace(0.0001, 0.3, num=10)
    }


    EXTRA_MODEL_CONFIGURATION = { 'psi_endogenous':True }
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(MarketPowerRun)

