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
    COMPARING_LABEL = "Average"

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "/experiments/251105_ir_study_base_p_p"

    parameters = {
        "p": [0.0001, 0.05, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
    }
    extra_individual_parameters = [
        {'p_avg_ir': 0.007951},
        {'p_avg_ir': 0.03000788},
        {'p_avg_ir': 0.03137311},
        {'p_avg_ir': 0.03345110},
        {'p_avg_ir': 0.03355167},
        {'p_avg_ir': 0.03271613},
        {'p_avg_ir': 0.03381310},
        {'p_avg_ir': 0.03413424},
        {'p_avg_ir': 0.03387948},
        {'p_avg_ir': 0.03476465},
        {'p_avg_ir': 0.03442159},
        {'p_avg_ir': 0.03297361},
    ]

    config = {}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': False, 'normalize_interest_rate_max': -2, 'psi': 0.3}
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025


if __name__ == "__main__":
    runner = exp_runner_parametrized.Runner(MarketPowerRun)
    runner.do()