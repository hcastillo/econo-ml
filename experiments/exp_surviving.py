#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc RestrictedMarket
   to determine number of surviving banks (no replacement of bankrupted)
   depending on p

@author: hector@bith.net
"""
import numpy as np
import exp_runner_surviving


class RestrictedMarketSurvivingRun(exp_runner_surviving.SurvivingRun):
    N = 50
    T = 100
    MC = 2

    OUTPUT_DIRECTORY = "c:\\experiments\\surviving"

    parameters = {  # items should be iterable:
        "p": np.linspace(0.0001, 0.6, num=4),
    }

    SEED_FOR_EXECUTION = 918994


if __name__ == "__main__":
    runner = exp_runner_surviving.Runner(RestrictedMarketSurvivingRun)
    experiment = runner.do()
    if experiment:
        experiment.generate_data_surviving()
        experiment.plot_surviving()