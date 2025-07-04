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
    MC = 10

    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "c:\\experiments\\4_xi"
    EXTRA_MODEL_CONFIGURATION = { 'psi_endogenous':True }

    parameters = {
        "xi": [0.1, 0.2, 0.3, 0.4, 0.5]
    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(MarketPowerRun)

