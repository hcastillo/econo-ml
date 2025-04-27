#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np

from interbank import Model
from interbank_lenderchange import ShockedMarket
import exp_runner


class MarketPowerRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 100

    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "c:\\experiments\\marketpower"
    COMPARING_DATA = "not_valid"
    COMPARING_LABEL = None

    parameters = {  # items should be iterable:
        "psi": np.linspace(0.0, 0.99, num=10),
        "p" : [0.99]
    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    seed = 2025
    seed_offset = 1

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()
        model.export_datafile = filename
        model.config.lender_change = self.ALGORITHM()
        model.config.lender_change.set_parameter("p", execution_parameters["p"])
        model.configure(T=self.T, N=self.N, **execution_config)
        MarketPowerRun.seed_offset += 1
        model.initialize(seed=(self.seed + MarketPowerRun.seed_offset), save_graphs_instants=None,
                         export_datafile=filename,
                         generate_plots=False,
                         export_description=self.describe_experiment_parameters(model, execution_parameters,
                                                                                seed_random))
        model.simulate_full(interactive=False)
        return model.finish()


if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(MarketPowerRun)

