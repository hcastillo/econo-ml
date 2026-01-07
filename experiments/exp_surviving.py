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
    MC = 5

    OUTPUT_DIRECTORY = "/experiments/251123.surviving"
    DESCRIPTION_TITLE = "with \\rho=0.3"

    parameters = {
        "p": [0.0001, 0.001, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
    }
    # parameters = {
    #     "p": [0.0001, 0.05, 0.07,
    #                0.08,
    #           0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1] # np.linspace(0.0001, 0.2, num=5),
    # }

    SEED_FOR_EXECUTION = 918994


if __name__ == "__main__":
    runner = exp_runner_surviving.Runner(RestrictedMarketSurvivingRun)
    experiment = runner.do()
    if experiment:
        experiment.generate_data_surviving()
        experiment.plot_surviving()