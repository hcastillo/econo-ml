#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank import Model
import exp_runner_no_concurrent


class PreferentialRun(exp_runner_old.ExperimentRun):
    N = 50
    T = 100
    MC = 10

    OUTPUT_DIRECTORY = "c:\\experiments\\exp_parallel2b"
    parameters = {
        "p": np.linspace(0.0001, 0.2, num=10),
    }

    config = {}

    EXTRA_MODEL_CONFIGURATION = {'psi_endogenous': True}

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()
        model.export_datafile = filename
        model.config.lender_change = self.ALGORITHM()
        model.config.lender_change.set_parameter("p", int(execution_parameters["p"]))
        model.configure(T=self.T, N=self.N, **execution_config)
        model.initialize(seed=seed_random, save_graphs_instants=None,
                         export_datafile=filename,
                         generate_plots=False)
        model.simulate_full(interactive=False)
        return model.finish()


if __name__ == "__main__":
    runner = exp_runner_old.Runner()
    runner.do(PreferentialRun)

