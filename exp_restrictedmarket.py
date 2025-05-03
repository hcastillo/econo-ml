#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
from interbank_lenderchange import RestrictedMarket
import exp_runner


class RestrictedMarketRun(exp_runner.ExperimentRun):
    N = 50
    T = 100
    MC = 10

    OUTPUT_DIRECTORY = "c:\\experiments\\restrictedmarket"
    ALGORITHM = RestrictedMarket

    parameters = {  # items should be iterable:
        "p": {0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1}
    }


    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    SEED_FOR_EXECUTION = 988993


if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(RestrictedMarketRun)
