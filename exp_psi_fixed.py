#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket

We determine the average of psi for the endogenous execution c:\\experiments\\psi_endogenous
(psi_fixed_average) and also for each p (psi_fixed_by_p), to compare properly later the results

@author: hector@bith.net
"""
import numpy as np
from interbank_lenderchange import ShockedMarket
import exp_runner
from interbank import Model
import interbank_lenderchange

class MarketPowerRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 40

    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "c:\\experiments\\psi_fixed"
    COMPARING_DATA = "not_exists"
    COMPARING_LABEL = None

    PSI_ENDOGENOUS_DATA = "c:\\experiments\\psi_endogenous"


    EXTRA_MODEL_CONFIGURATION = { 'psi_endogenous':False, 'psi': 0.585919 }

    parameters = {
        "p": np.linspace(0.0001, 1, num=10)
    }


    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    SEED_FOR_EXECUTION = 2025


    psi = {}

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()
        model.export_datafile = filename
        model.config.lender_change = self.ALGORITHM()
        if 'p' in execution_parameters:
            model.config.lender_change.set_parameter("p", execution_parameters["p"])
        model.configure(T=self.T, N=self.N,
                        allow_replacement_of_bankrupted=self.ALLOW_REPLACEMENT_OF_BANKRUPTED, **execution_config)
        model.configure(**self.EXTRA_MODEL_CONFIGURATION)
        model.initialize(seed=seed_random, save_graphs_instants=None,
                         export_datafile=filename,
                         generate_plots=False)
        model.simulate_full(interactive=False)
        return model.finish()

    @staticmethod
    def load_psi_values():
        MarketPowerRun.psi = {}
        print("hola")
        import sys
        sys.exit()


if __name__ == "__main__":
    runner = exp_runner.Runner()
    MarketPowerRun.load_psi_values()
    runner.do(MarketPowerRun)

