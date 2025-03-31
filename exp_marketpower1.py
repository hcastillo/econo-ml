#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank import Model, Statistics
from interbank_lenderchange import ShockedMarket
import exp_runner
import os


class MarketPowerRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 1


    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "../experiments/marketpower"

    parameters = {  # items should be iterable:
        "psi": np.linspace(0.0, 0.99, num=10),
        "p": np.linspace(0.0001, 1, num=2),
    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    seed = 988994
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
                         export_description=str(model.config) + str(execution_parameters))
        model.simulate_full(interactive=False)
        return model.finish()
    
    

    @staticmethod
    def plot_surviving(experiment):
        print("Plotting marketpower...")
        # now we open all the .gdt files:
        data_of_surviving_banks = {}
        data_of_failures_rationed = {}
        data_of_failures_rationed_acum = {}
        data_of_failures = {}
        data_of_failures_acum = {}

        all_models = []
        for model_configuration in experiment.get_models(MarketPowerRun.parameters):
            all_models.append(model_configuration)
            filename_for_iteration = experiment.get_filename_for_iteration(model_configuration, {})
            data_of_surviving_banks[filename_for_iteration] = []
            data_of_failures[filename_for_iteration] = []
            data_of_failures_acum[filename_for_iteration] = []
            data_of_failures_rationed[filename_for_iteration] = []
            data_of_failures_rationed_acum[filename_for_iteration] = []
            for i in range(MarketPowerRun.MC):
                if os.path.isfile(f"{MarketPowerRun.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt"):
                    result_mc = Statistics.read_gdt(
                        f"{MarketPowerRun.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
                    data_of_surviving_banks[filename_for_iteration].append(result_mc['num_banks'])
                    # bankruptcies and bankruptcies rationed are accumulated data:
                    # [1,2,1,0,1] -> [1,3,4,4,5]
                    data_of_failures[filename_for_iteration].append(result_mc['bankruptcies'])
                    data_of_failures_acum[filename_for_iteration].append(
                        MarketPowerRun.accumulated_data(result_mc['bankruptcies']))
                    data_of_failures_rationed[filename_for_iteration].append(result_mc['bankruptcy_rationed'])
                    data_of_failures_rationed_acum[filename_for_iteration].append(
                        MarketPowerRun.accumulated_data(result_mc['bankruptcy_rationed']))

        # we have now MC different series for each iteration, so let's estimate the average of all the MonteCarlos:
        data_of_surviving_banks_avg = MarketPowerRun.determine_average_of_series(data_of_surviving_banks)
        data_of_failures_avg = MarketPowerRun.determine_average_of_series(data_of_failures)
        data_of_failures_acum_avg = MarketPowerRun.determine_average_of_series(data_of_failures_acum)
        data_of_failures_rationed_avg =\
            MarketPowerRun.determine_average_of_series(data_of_failures_rationed)
        data_of_failures_rationed_acum_avg =\
            MarketPowerRun.determine_average_of_series(data_of_failures_rationed_acum)

        max_t = MarketPowerRun.determine_max_t([data_of_surviving_banks_avg,data_of_failures_avg,
                                                              data_of_failures_rationed_avg])
        MarketPowerRun.generate_plot("Surviving banks", "_surviving.png",
                                                   data_of_surviving_banks_avg, all_models, max_t)
        MarketPowerRun.generate_plot("Surviving banks log", "_surviving_log.png",
                                                   data_of_surviving_banks_avg, all_models, max_t, logarithm=True)
        MarketPowerRun.generate_plot("Failures", "_failures.png",
                                                   data_of_failures_avg, all_models, max_t)
        MarketPowerRun.generate_plot("Failures log", "_failures_log.png",
                                                   data_of_failures_avg, all_models, max_t, logarithm=True)
        MarketPowerRun.generate_plot("Failures acum", "_rationed_acum.png",
                                                   data_of_failures_acum_avg, all_models, max_t)
        MarketPowerRun.generate_plot("Failures acum log", "_rationed_acum_log.png",
                                                   data_of_failures_acum_avg, all_models, max_t, logarithm=True)
        MarketPowerRun.generate_plot("Failures rationed", "_rationed.png",
                                                   data_of_failures_rationed_avg, all_models, max_t)
        MarketPowerRun.generate_plot("Failures rationed", "_rationed_log.png",
                                                   data_of_failures_rationed_avg, all_models, max_t, logarithm=True)
        MarketPowerRun.generate_plot("Failures rationed acum", "_rationed_acum.png",
                                                   data_of_failures_rationed_acum_avg, all_models, max_t)
        MarketPowerRun.generate_plot("Failures rationed acum log", "_rationed_acum_log.png",
                                                   data_of_failures_rationed_acum_avg, all_models, max_t, logarithm=True)
        return [data_of_surviving_banks_avg,
                data_of_failures_rationed_avg,
                data_of_failures_avg], all_models, max_t

    @staticmethod
    def save_surviving_csv(data, max_t):
        data_columns = ['surviving', 'failures_rationed', 'failures']
        for k in range(len(data_columns)):
            with open(f"{MarketPowerRun.OUTPUT_DIRECTORY}/_{data_columns[k]}.csv", "w") as file:
                file.write("t")
                for j in data[k]:
                    file.write(f";{j}")
                file.write("\n")
                for i in range(max_t):
                    file.write(f"{i}")
                    for j in data[k]:
                        if i < len(data[k][j]):
                            file.write(f";{data[k][j][i]}")
                        else:
                            file.write(f";0.0")
                    file.write("\n")


if __name__ == "__main__":
    runner = exp_runner.Runner()
    experiment = runner.do(MarketPowerRun)
    (data, all_models, max_t) = MarketPowerRun.plot_marketpower(experiment)
    MarketPowerRun.save_marketpower_csv(data, max_t)

