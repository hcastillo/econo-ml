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
    COMPARING_LABEL = "Average"
    NAME_OF_X_SERIES = "Smaller $p$"
    COMPARING_COLOR = 'black'
    STYLE = '--'
    COLOR = 'red'
    MARKER = '+'

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251110_ir_study_base_psi_p"

    parameters = {
        "p": [0.2]
    }

    config = {"psi": [0, 0.25, 0.50, 0.75, 0.99]}

    extra_individual_parameters = [
        {'p_avg_ir': 0.3176117},
        {'p_avg_ir': 0.3335269},
        {'p_avg_ir': 0.3602519},
        {'p_avg_ir': 0.3881468},
        {'p_avg_ir': 0.3331875},
    ]
    extra_individual_parameters_multiplier = 0.1

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2}

    LENGTH_FILENAME_PARAMETER = 0
    LENGTH_FILENAME_CONFIG = 6

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()