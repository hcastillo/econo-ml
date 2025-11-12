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

    COMPARING_DATA2 = "/experiments/251110_ir_study_base_p_c"
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
    OUTPUT_DIRECTORY = "/experiments/251110_ir_study_base_p_c1"

    parameters = {
        # "p": [0.0001, 0.05, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
        "p": [0.0001, 0.001, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
    }
    extra_individual_parameters = [
        {'c_avg_ir': 184.2818},
        {'c_avg_ir': 183.7230},
        {'c_avg_ir': 181.5590},
        {'c_avg_ir': 175.1106},
        {'c_avg_ir': 172.4889},
        {'c_avg_ir': 171.9848},
        {'c_avg_ir': 170.7092},
        {'c_avg_ir': 168.9634},
        {'c_avg_ir': 167.5176},
        {'c_avg_ir': 168.1224},
        {'c_avg_ir': 168.4239},
        {'c_avg_ir': 166.2980},
        {'c_avg_ir': 166.4121},
        {'c_avg_ir': 166.6523},
        {'c_avg_ir': 167.5630},
        {'c_avg_ir': 166.6562},
        {'c_avg_ir': 168.2223},
    ]
    extra_individual_parameters_multiplier =10
    config = {}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2, 'psi': 0.3}
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()