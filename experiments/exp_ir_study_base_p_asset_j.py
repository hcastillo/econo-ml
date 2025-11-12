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
    NAME_OF_X_SERIES = "Smaller $A_j$"
    COMPARING_COLOR = 'black'
    STYLE = '--'
    COLOR = 'red'
    MARKER = '+'

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251110_ir_study_base_p_asset_j"

    parameters = {
        # "p": [0.0001, 0.05, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
        "p": [0.0001, 0.001, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
    }
    extra_individual_parameters = [
        {'asset_j_avg_ir': 188.7844},
        {'asset_j_avg_ir': 188.2250},
        {'asset_j_avg_ir': 186.0505},
        {'asset_j_avg_ir': 179.5493},
        {'asset_j_avg_ir': 176.9311},
        {'asset_j_avg_ir': 176.4252},
        {'asset_j_avg_ir': 175.1078},
        {'asset_j_avg_ir': 173.3365},
        {'asset_j_avg_ir': 171.8820},
        {'asset_j_avg_ir': 172.5247},
        {'asset_j_avg_ir': 172.8819},
        {'asset_j_avg_ir': 170.6922},
        {'asset_j_avg_ir': 170.8796},
        {'asset_j_avg_ir': 171.1837},
        {'asset_j_avg_ir': 172.1175},
        {'asset_j_avg_ir': 171.0100},
        {'asset_j_avg_ir': 172.6347},
    ]
    extra_individual_parameters_multiplier=0.1

    config = {}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2, 'psi': 0.3}
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()