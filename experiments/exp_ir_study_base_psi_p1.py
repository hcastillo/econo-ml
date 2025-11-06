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

    COMPARING_DATA = "/experiments/251105_ir_study_base_psi"


    COMPARING_DATA2 = "/experiments/251105_ir_study_base_psi_c"
    COMPARING_LABEL = "Average"
    COMPARING_LABEL2 = "Smaller p"
    NAME_OF_X_SERIES = "Greater p"
    DESCRIPTION_TITLE = '(Forcing probability of bankruptcy $p$)'
    COMPARING_TICKS2 = ':'
    COMPARING_COLOR2 = 'red'
    COMPARING_TICKS = '-'
    COMPARING_COLOR = 'black'
    TICKS = '--'
    COLOR = 'red'

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251105_ir_study_base_psi_c1"

    parameters = {
            "p": [0.2]
    }

    config = {"psi": [0, 0.25, 0.50, 0.75, 0.99]}

    extra_individual_parameters = [
        {'p_avg_ir': 03.244191},
        {'p_avg_ir': 03.352819},
        {'p_avg_ir': 03.633483},
        {'p_avg_ir': 03.855890},
        {'p_avg_ir': 03.304037},
    ]

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2}

    LENGTH_FILENAME_PARAMETER = 0
    LENGTH_FILENAME_CONFIG = 6

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()