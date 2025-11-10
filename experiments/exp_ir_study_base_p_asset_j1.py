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

    COMPARING_DATA2 = "/experiments/251110_ir_study_base_p_asset_j"
    COMPARING_LABEL = "Average"
    COMPARING_LABEL2 = "Smaller $A_j$"
    NAME_OF_X_SERIES = "Greater $A_j$"
    DESCRIPTION_TITLE = '(Forcing $A_j$)'
    COMPARING_MARKER = "o"
    COMPARING_MARKER2 = "x"
    COMPARING_STYLE2 = ':'
    COMPARING_COLOR2 = 'red'
    COMPARING_STYLE = '-'
    COMPARING_COLOR = 'black'
    STYLE = '--'
    COLOR = 'red'
    MARKER = '+'

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251110_ir_study_base_p_asset_j1"

    parameters = {
        # "p": [0.0001, 0.05, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
        "p": [0.0001, 0.001, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
    }
    extra_individual_parameters = [
        {'asset_j_avg_ir': 182.0357},
        {'asset_j_avg_ir': 182.5018},
        {'asset_j_avg_ir': 179.4384},
        {'asset_j_avg_ir': 173.8898},
        {'asset_j_avg_ir': 173.4964},
        {'asset_j_avg_ir': 172.0575},
        {'asset_j_avg_ir': 171.1850},
        {'asset_j_avg_ir': 169.6096},
        {'asset_j_avg_ir': 168.9811},
        {'asset_j_avg_ir': 169.0410},
        {'asset_j_avg_ir': 168.8644},
        {'asset_j_avg_ir': 168.3623},
        {'asset_j_avg_ir': 167.9528},
        {'asset_j_avg_ir': 168.0805},
        {'asset_j_avg_ir': 168.6741},
        {'asset_j_avg_ir': 168.5763},
        {'asset_j_avg_ir': 169.0580},
    ]
    extra_individual_parameters_multiplier = 10

    config = {}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2, 'psi': 0.3}
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()