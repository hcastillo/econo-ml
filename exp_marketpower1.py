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
    OUTPUT_DIRECTORY = "../experiments/marketpower1"
    COMPARING_DATA = "not_valid"

    parameters = {  # items should be iterable:
        "psi": np.linspace(0.0, 0.99, num=10),
        "p": np.linspace(0.0001, 1, num=10),
    }

    LENGTH_FILENAME_PARAMETER = 8
    LENGTH_FILENAME_CONFIG = 1

    seed = 9999
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
    def plot_marketpower(experiment):
        print("Plotting marketpower...")
        # now we open all the .gdt files:

        all_models = {}
        data = {}

        for model_configuration in experiment.get_models(MarketPowerRun.parameters):
            if not model_configuration['p'] in all_models:
                all_models[model_configuration['p']] = {}

            filename_for_iteration = experiment.get_filename_for_iteration(model_configuration, {})
            all_models[model_configuration['p']][model_configuration['psi']] = filename_for_iteration
            data[filename_for_iteration] = []

            for i in range(MarketPowerRun.MC):
                if os.path.isfile(f"{MarketPowerRun.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt"):
                    result_mc = Statistics.read_gdt(
                        f"{MarketPowerRun.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
                    data[filename_for_iteration].append(result_mc)
        return data, all_models

    @staticmethod
    def save_marketpower_csv(data, all_models):
        # all_models[p][psi] = [filename]
        # data[filename][0]...[MC]
        #
        # we extract the first item in the data[], which is a matrix with arrays (next of next then)
        data_columns = list(next(iter(data[next(iter(data))])).columns)
        for k in data_columns:
            with open(f"{MarketPowerRun.OUTPUT_DIRECTORY}/_{k}.csv", "w") as file:
                f1 = open(f"{MarketPowerRun.OUTPUT_DIRECTORY}/_{k}_mean.csv", "w")
                for j in all_models[next(iter(all_models))]:
                    file.write(f";phi={j}")
                    f1.write(f";phi={j}")
                file.write("\n")
                f1.write("\n")

                for i in all_models:
                    file.write(f"p={i}")
                    f1.write(f"p={i}")
                    for j in all_models[i]:
                        value = data[all_models[i][j]][0][k].median()
                        value_mean = data[all_models[i][j]][0][k].mean()
                        for l in range(1, 1+len(data[all_models[i][j]][1:])):
                            value += data[all_models[i][j]][l][k].median()
                            value_mean += data[all_models[i][j]][l][k].mean()
                        value /= len(data[all_models[i][j]])
                        value_mean /= len(data[all_models[i][j]])
                        file.write(f";{value}".replace(".",","))
                        f1.write(f";{value_mean}".replace(".", ","))
                    file.write("\n")
                    f1.write("\n")
                f1.close()

if __name__ == "__main__":
    runner = exp_runner.Runner()
    experiment = runner.do(MarketPowerRun)
    (data, all_models) = MarketPowerRun.plot_marketpower(experiment)
    MarketPowerRun.save_marketpower_csv(data, all_models)
