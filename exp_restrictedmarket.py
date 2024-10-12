#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank import Model
from interbank_lenderchange import RestrictedMarket
import exp_runner


class RestrictedMarketRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 2

    OUTPUT_DIRECTORY = "experiments/restrictedmarket2"
    ALGORITHM = RestrictedMarket

    parameters = {  # items should be iterable:
        "p": np.linspace(0.001, 0.100, num=200),
    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    seed = 988993
    seed_offset = 0

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()
        model.export_datafile = filename
        model.config.lender_change = self.ALGORITHM()
        model.config.lender_change.set_parameter("p", execution_parameters["p"])
        model.configure(T=self.T, N=self.N, **execution_config)
        RestrictedMarketRun.seed_offset += 1
        model.initialize(seed=self.seed + RestrictedMarketRun.seed_offset, save_graphs_instants=None,
                         export_datafile=filename,
                         generate_plots=False,
                         export_description=str(model.config) + str(execution_parameters))
        model.simulate_full(interactive=False)
        return model.finish()


if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(RestrictedMarketRun)
