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
    COMPARING_LABEL = "Using $A_i$"
    NAME_OF_X_SERIES = 'Using $\\bar{A_i}$'
    DESCRIPTION_TITLE = 'Montecarlo $\mathcal{M}=5$'
    COMPARING_STYLE = ':'
    COMPARING_COLOR = 'black'
    STYLE = '--'
    COLOR = 'red'

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251110_ir_study_base_p_asset_i_canary"

    parameters = {
        # "p": [0.0001, 0.05, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
        "p": [0.0001, 0.001, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
    }
    extra_individual_parameters = [
        {'asset_i_avg_ir': 180.8181},
        {'asset_i_avg_ir': 185.3165},
        {'asset_i_avg_ir': 183.8469},
        {'asset_i_avg_ir': 177.5628},
        {'asset_i_avg_ir': 176.1214},
        {'asset_i_avg_ir': 175.4710},
        {'asset_i_avg_ir': 174.6336},
        {'asset_i_avg_ir': 172.6699},
        {'asset_i_avg_ir': 171.8027},
        {'asset_i_avg_ir': 172.0873},
        {'asset_i_avg_ir': 172.6047},
        {'asset_i_avg_ir': 170.3323},
        {'asset_i_avg_ir': 170.0435},
        {'asset_i_avg_ir': 170.0624},
        {'asset_i_avg_ir': 171.0692},
        {'asset_i_avg_ir': 169.7977},
        {'asset_i_avg_ir': 171.8950},
    ]
    config = {}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2, 'psi': 0.3}
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()