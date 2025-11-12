#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
from interbank_lenderchange import ShockedMarket3
import exp_runner_parametrized


class MarketPowerRun(exp_runner_parametrized.ExperimentRunParametrized):
    N = 50
    T = 1000
    MC = 5

    COMPARING_DATA = "/experiments/251110_ir_study_base_p"
    COMPARING_LABEL = "Average"
    NAME_OF_X_SERIES = "Smaller $p$"
    COMPARING_COLOR = 'black'
    STYLE = '--'
    COLOR = 'red'
    MARKER = '+'

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251110_ir_study_base_p_p"

    parameters = {
        # "p": [0.0001, 0.05, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
        "p": [0.0001, 0.001, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
    }
    extra_individual_parameters = [
        {'p_avg_ir': 0.02501},
        {'p_avg_ir': 0.03188},
        {'p_avg_ir': 0.1185850},
        {'p_avg_ir': 0.1750388},
        {'p_avg_ir': 0.1871085},
        {'p_avg_ir': 0.1958568},
        {'p_avg_ir': 0.2028540},
        {'p_avg_ir': 0.2009669},
        {'p_avg_ir': 0.2175480},
        {'p_avg_ir': 0.2142441},
        {'p_avg_ir': 0.2266124},
        {'p_avg_ir': 0.2238496},
        {'p_avg_ir': 0.2283871},
        {'p_avg_ir': 0.2297489},
        {'p_avg_ir': 0.2309701},
        {'p_avg_ir': 0.2376311},
        {'p_avg_ir': 0.2315953},
    ]
    extra_individual_parameters_multiplier = 0.1

    config = {}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2, 'psi': 0.3}
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()