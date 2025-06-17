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
import interbank

class MarketPowerRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 40

    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "c:\\experiments\\psi_fixed"
    COMPARING_DATA = "not_exists"
    COMPARING_LABEL = None

    PSI_ENDOGENOUS_DATA = "c:\\experiments\\psi_endogenous"

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
        print(MarketPowerRun.psi[execution_parameters["p"]])
        value_of_p = float(MarketPowerRun.psi[execution_parameters["p"]])
        extra_config = { 'psi_endogenous':False, 'psi': value_of_p  }
        model.configure(**extra_config)
        model.initialize(seed=seed_random, save_graphs_instants=None,
                         export_datafile=filename,
                         generate_plots=False)
        model.simulate_full(interactive=False)
        return model.finish()

    def load_psi_values(self):
        import os.path
        MarketPowerRun.psi = {}
        for model_configuration in self.get_models(self.config):
            for model_parameter in self.get_models(self.parameters):
                for i in range(self.MC):
                    filename_for_iteration = self.get_filename_for_iteration(model_parameter, model_configuration)+f"_{i}.gdt"
                    if os.path.exists(self.PSI_ENDOGENOUS_DATA+"/"+filename_for_iteration):
                        result_mc = interbank.Statistics.read_gdt(self.PSI_ENDOGENOUS_DATA+"/"+filename_for_iteration)
                        value_psi = result_mc['psi'].mean()
                        if model_parameter['p'] in MarketPowerRun.psi:
                            MarketPowerRun.psi[model_parameter['p']].append(value_psi)
                        else:
                            MarketPowerRun.psi[model_parameter['p']]= [value_psi]
                    else:
                        import sys
                        print(f"error: absence of {filename_for_iteration}")
                        sys.exit(-1)

    def show(self):
        total = []
        for value_of_p in MarketPowerRun.psi:
            total.append(np.average(MarketPowerRun.psi[value_of_p]))
            print(f"p={value_of_p:-15} psi={total[-1]:-7}")
        print("average_of_all=",np.average(total))


if __name__ == "__main__":
    runner = exp_runner.Runner()
    load_values = MarketPowerRun()
    load_values.load_psi_values()
    load_values.show()
    runner.do(MarketPowerRun)

