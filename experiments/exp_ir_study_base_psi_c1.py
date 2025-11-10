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
    COMPARING_DATA2 = "/experiments/251110_ir_study_base_psi_c"
    COMPARING_LABEL = "Average"
    COMPARING_LABEL2 = "Smaller c"
    NAME_OF_X_SERIES = "Greater c"
    DESCRIPTION_TITLE = '(Forcing $c$)'

    COMPARING_MARKER = "o"
    COMPARING_MARKER2 = "x"
    COMPARING_STYLE2 = ':'
    COMPARING_COLOR2 = 'red'
    COMPARING_STYLE = '-'
    COMPARING_COLOR = 'black'
    STYLE = '--'
    COLOR = 'red'
    MARKER= '+'

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251110_ir_study_base_psi_c1"

    parameters = {
            "p": [0.2]
    }

    config = {"psi": [0, 0.25, 0.50, 0.75, 0.99]}

    extra_individual_parameters = [
        {'c_avg_ir': 163.5796},
        {'c_avg_ir': 162.6920},
        {'c_avg_ir': 163.9278},
        {'c_avg_ir': 164.6128},
        {'c_avg_ir': 164.5199},
    ]
    extra_individual_parameters_multiplier = 10

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2}

    LENGTH_FILENAME_PARAMETER = 0
    LENGTH_FILENAME_CONFIG = 6

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()