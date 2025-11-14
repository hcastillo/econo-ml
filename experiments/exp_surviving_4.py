#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc RestrictedMarket
   to determine number of survivings banks (no replacement of bankrupted)
   depending on p

@author: hector@bith.net
"""
import numpy as np
import exp_runner_surviving


class RestrictedMarketSurvivingRun(exp_runner_surviving.SurvivingRun):
    N = 50
    T = 1000
    MC = 15

    OUTPUT_DIRECTORY = "/experiments/251113/surviving.4"
    DESCRIPTION_TITLE = "with \\rho=0.3"

    parameters = {
        "p": np.linspace(0.0001, 1, num=4),
    }

    SEED_FOR_EXECUTION = 918995


if __name__ == "__main__":
    runner = exp_runner_surviving.Runner(RestrictedMarketSurvivingRun)
    experiment = runner.do()
    if experiment:
        experiment.generate_data_surviving()
        experiment.plot_surviving()

