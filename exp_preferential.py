#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank import Model
from interbank_lenderchange import Preferential
import exp_runner


class PreferentialRun(exp_runner.ExperimentRun):
    N = 50
    T = 100
    MC = 1

    OUTPUT_DIRECTORY = "experiments\\preferential"
    ALGORITHM = Preferential
    COMPARING_DATA = "experiments\\boltzman"
    COMPARING_LABEL = "Boltzman"

    parameters = {  # items should be iterable:
        "m": np.linspace(1, 49, num=49),
    }

    LENGTH_FILENAME_PARAMETER = 2
    LENGTH_FILENAME_CONFIG = 0

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()
        model.export_datafile = filename
        model.config.lender_change = self.ALGORITHM()
        model.config.lender_change.set_parameter("m", int(execution_parameters["m"]))
        model.configure(T=self.T, N=self.N, **execution_config)
        model.initialize(seed=seed_random, save_graphs_instants=None,
                         export_datafile=filename,
                         generate_plots=False,
                         export_description=str(model.config) + str(execution_parameters))
        model.simulate_full(interactive=False)
        return model.finish()


if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(PreferentialRun)

