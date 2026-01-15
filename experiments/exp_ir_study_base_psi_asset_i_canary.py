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

    COMPARING_DATA = "/experiments/251110_ir_study_base_psi"
    COMPARING_LABEL = "Using $A_i$"
    NAME_OF_X_SERIES = 'Using $\\bar{A_i}$'
    DESCRIPTION_TITLE = 'Montecarlo $\\mathcal{M}=5$'
    COMPARING_STYLE = ':'
    COMPARING_COLOR = 'black'
    STYLE = '--'
    COLOR = 'red'

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251110_ir_study_base_psi_asset_i_canary"

    parameters = {
            "p": [0.2]
    }

    config = {"psi": [0, 0.25, 0.50, 0.75, 0.99]}

    extra_individual_parameters = [
        {'asset_i_avg_ir': 167.4086},
        {'asset_i_avg_ir': 166.6946},
        {'asset_i_avg_ir': 167.8455},
        {'asset_i_avg_ir': 168.6340},
        {'asset_i_avg_ir': 167.7435},    ]

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': 1}

    LENGTH_FILENAME_PARAMETER = 0
    LENGTH_FILENAME_CONFIG = 6

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()