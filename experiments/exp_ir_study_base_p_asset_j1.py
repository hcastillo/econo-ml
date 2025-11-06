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

    COMPARING_DATA2 = "/experiments/251105_ir_study_base_p_asset_j"
    COMPARING_LABEL = "Average"
    COMPARING_LABEL2 = "Smaller $A_j$"
    NAME_OF_X_SERIES = "Greater $A_j$"
    DESCRIPTION_TITLE = '(Forcing $A_j$)'
    COMPARING_TICKS2 = ':'
    COMPARING_COLOR2 = 'red'
    COMPARING_TICKS = '-'
    COMPARING_COLOR = 'black'
    TICKS = '--'
    COLOR = 'red'

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251105_ir_study_base_p_asset_j1"

    parameters = {
        "p": [0.0001, 0.05, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
    }
    extra_individual_parameters = [
        {'asset_j_avg_ir': 1817.343},
        {'asset_j_avg_ir': 1717.124},
        {'asset_j_avg_ir': 1696.096},
        {'asset_j_avg_ir': 1689.811},
        {'asset_j_avg_ir': 1690.410},
        {'asset_j_avg_ir': 1688.644},
        {'asset_j_avg_ir': 1683.623},
        {'asset_j_avg_ir': 1679.528},
        {'asset_j_avg_ir': 1680.805},
        {'asset_j_avg_ir': 1686.741},
        {'asset_j_avg_ir': 1685.763},
        {'asset_j_avg_ir': 1690.580},
    ]

    config = {}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2, 'psi': 0.3}
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()