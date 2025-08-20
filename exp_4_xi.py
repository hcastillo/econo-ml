#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np

import exp_runner_comparer
from interbank_lenderchange import ShockedMarket3
import exp_runner


class MarketPowerRun(exp_runner_comparer.ExperimentComparerRun):
    N = 50
    T = 1000
    MC = 10

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "c:\\experiments\\04_xi"
    EXTRA_MODEL_CONFIGURATION = { 'psi_endogenous':True }

    config = {
        "xi": [0.1, 0.2, 0.3, 0.4, 0.5]
    }

    parameters = {  # items should be iterable:
        "p": np.linspace(0.001, 1, num=10),
    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 5
    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(MarketPowerRun)

