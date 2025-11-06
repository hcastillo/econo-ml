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

    COMPARING_DATA = "/experiments/251105_ir_study_base_p"
    COMPARING_LABEL = "Using $A_i$"
    NAME_OF_X_SERIES = 'Using $\\bar{A_i}$'
    DESCRIPTION_TITLE = 'Montecarlo $\mathcal{M}=5$'
    COMPARING_TICKS = ':'
    COMPARING_COLOR = 'black'
    TICKS = '--'
    COLOR = 'red'

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251105_ir_study_base_asset_i_canary"

    parameters = {
        "p": [0.0001, 0.05, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
    }
    extra_individual_parameters = [
        {'asset_i_avg_ir':     165.2221},
        {'asset_i_avg_ir':     170.8417},
        {'asset_i_avg_ir':     168.4675},
        {'asset_i_avg_ir':     168.3421},
        {'asset_i_avg_ir':     168.4038},
        {'asset_i_avg_ir':     168.6880},
        {'asset_i_avg_ir':     168.1806},
        {'asset_i_avg_ir':     167.0170},
        {'asset_i_avg_ir':     167.5376},
        {'asset_i_avg_ir':     168.0968},
        {'asset_i_avg_ir':     167.5727},
        {'asset_i_avg_ir':     168.1890}
    ]

    config = {}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2, 'psi': 0.3}
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()