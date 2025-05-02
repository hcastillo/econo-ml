#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank import Model
from interbank_lenderchange import ShockedMarket
import exp_runner


class ShockedMarketRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 100

    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "c:\\experiments\\shockedmarket"

    parameters = {
        "p": np.linspace(0.001, 0.95, num=10)
    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    seed = 9
    seed_offset = 1

    SEED_FOR_EXECUTION = 98899

if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(ShockedMarketRun)
