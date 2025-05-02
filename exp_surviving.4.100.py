#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc RestrictedMarket
   to determine number of survivings banks (no replacement of bankrupted)
   depending on p

@author: hector@bith.net
"""
import numpy as np
import exp_surviving_runner


class RestrictedMarketSurvivingRun(exp_surviving_runner.SurvivingRun):
    N = 100
    T = 1000
    MC = 15

    OUTPUT_DIRECTORY = "c:\\experiments\\surviving.4.100"

    parameters = {  # items should be iterable:
        "p": np.linspace(0.0001, 1, num=4),
    }

    SEED_FOR_EXECUTION = 918994


if __name__ == "__main__":
    runner = exp_surviving_runner.Runner()
    experiment = runner.do(RestrictedMarketSurvivingRun)
    experiment.plot_surviving(experiment)

