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
    N = 50
    T = 1000
    MC = 15

    OUTPUT_DIRECTORY = "c:\\experiments\\surviving.psi"

    parameters = {
        "p": [1],
    }
    config = {
        "psi": [0, 0.3333, 0.6666, 0.999]
    }

    SEED_FOR_EXECUTION = 318994
    LENGTH_FILENAME_PARAMETER = 3
    LENGTH_FILENAME_CONFIG = 5
    COLORS_VARIABLE = 'psi'

if __name__ == "__main__":
    runner = exp_surviving_runner.Runner()
    experiment = runner.do(RestrictedMarketSurvivingRun)
    if experiment:
        experiment.plot_surviving(experiment)

