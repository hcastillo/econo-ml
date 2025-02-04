#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
   to determine number of survivings banks (no replacement of bankrupted)
   depending on p

@author: hector@bith.net
"""
import numpy as np
from interbank import Model, Statistics
from interbank_lenderchange import ShockedMarket
import exp_runner
import os
import matplotlib.pyplot as plt

class ShockedMarketSurvivingRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 20

    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "../experiments/surviving"

    parameters = {  # items should be iterable:
        # "p": np.linspace(0.05, 0.3, num=4),
        "p": {0.05, 0.3333333, 0.66666666, 0.99999999}
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
        model.configure(T=self.T, allow_replacement_of_bankrupted=False,
                        N=self.N, **execution_config)
        ShockedMarketSurvivingRun.seed_offset += 1
        model.initialize(seed=(self.seed + ShockedMarketSurvivingRun.seed_offset), save_graphs_instants=None,
                         export_datafile=filename,
                         generate_plots=False,
                         export_description=str(model.config) + str(execution_parameters))
        model.simulate_full(interactive=False)
        return model.finish()

    @staticmethod
    def plot_surviving(experiment):
        print("Plotting surviving...")
        # now we open all the .gdt files and we obtain the plot with only the num banks:
        data_of_surviving_banks = {}
        data_of_failures_rationed = {}
        data_of_failures_not_rationed = {}

        all_models = []
        for model_configuration in experiment.get_models(ShockedMarketSurvivingRun.parameters):
            all_models.append(model_configuration)
            filename_for_iteration = experiment.get_filename_for_iteration(model_configuration, {})
            data_of_surviving_banks[filename_for_iteration] = []
            data_of_failures_not_rationed[filename_for_iteration] = []
            data_of_failures_rationed[filename_for_iteration] = []
            for i in range(ShockedMarketSurvivingRun.MC):
                if os.path.isfile(f"{ShockedMarketSurvivingRun.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt"):
                    result_mc = Statistics.read_gdt(
                        f"{ShockedMarketSurvivingRun.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
                    data_of_surviving_banks[filename_for_iteration].append(result_mc['num_banks'])
                    data_of_failures_not_rationed[filename_for_iteration].append(result_mc['bankruptcy_not_rationed'])
                    data_of_failures_rationed[filename_for_iteration].append(
                        result_mc['bankruptcies']-result_mc['bankruptcy_not_rationed'])

        # we have now MC different series for each iteration, so let's estimate the average of all of the diff MC for
        # surviving banks:
        data_of_surviving_banks_avg = {}
        data_of_failures_rationed_avg = {}
        data_of_failures_not_rationed_avg = {}
        for iteration in data_of_surviving_banks:
            # we should do the average of each MC serie (an array):
            total = data_of_surviving_banks[iteration][0]
            for another in data_of_surviving_banks[iteration][1:]:
                total += another
            data_of_surviving_banks_avg[iteration] = total / len(data_of_surviving_banks[iteration])

        for iteration in data_of_failures_rationed:
            # we should do the average of each MC serie (an array):
            total = data_of_failures_rationed[iteration][0]
            for another in data_of_failures_rationed[iteration][1:]:
                total += another
            result_this_iteration = total / len(data_of_failures_rationed[iteration])
            # now we have in result_this_iteration the average for each t, but we need to have the surviving, so we do a
            # substraction starting in t=0 with ShockedMarketSurvivingRun.N
            surviving_minus_failures_rationed = ShockedMarketSurvivingRun.N
            for j in range(len(result_this_iteration)):
                surviving_minus_failures_rationed -= result_this_iteration[j]
                result_this_iteration[j] = surviving_minus_failures_rationed
            data_of_failures_rationed_avg[iteration] = result_this_iteration
        for iteration in data_of_failures_not_rationed:
            # we should do the average of each MC serie (an array):
            total = data_of_failures_not_rationed[iteration][0]
            for another in data_of_failures_not_rationed[iteration][1:]:
                total += another
            result_this_iteration = total / len(data_of_failures_not_rationed[iteration])
            surviving_minus_failures_not_rationed = ShockedMarketSurvivingRun.N
            for j in range(len(result_this_iteration)):
                surviving_minus_failures_not_rationed -= result_this_iteration[j]
                result_this_iteration[j] = surviving_minus_failures_not_rationed
            data_of_failures_not_rationed_avg[iteration] = result_this_iteration
        viridis = plt.colormaps['viridis']
        colors = viridis(np.linspace(0, 1, len(data_of_surviving_banks)))

        plt.clf()
        plt.title(f"Number of surviving banks with ShockedMarket p (MC={ShockedMarketSurvivingRun.MC})")
        max_t = 0
        for i, iteration in enumerate(data_of_surviving_banks_avg):
            plt.plot(data_of_surviving_banks_avg[iteration], "-", color=colors[i],
                     label=str(all_models[i]['p'])[:5])
            if len(data_of_surviving_banks_avg[iteration]) > max_t:
                max_t = len(data_of_surviving_banks_avg[iteration])
        plt.legend(loc='best', title="p")
        plt.xticks(range(0, max_t, 5))
        plt.savefig(f"{ShockedMarketSurvivingRun.OUTPUT_DIRECTORY}/_surviving.svg")

        plt.clf()
        plt.title(f"Number of surviving minus failures rationed - ShockedMarket p (MC={ShockedMarketSurvivingRun.MC})")
        max_t = 0
        for i, iteration in enumerate(data_of_failures_rationed_avg):
            plt.plot(data_of_failures_rationed_avg[iteration], "-", color=colors[i],
                     label=str(all_models[i]['p'])[:5])
            if len(data_of_failures_rationed_avg[iteration]) > max_t:
                max_t = len(data_of_failures_rationed_avg[iteration])
        plt.legend(loc='best', title="p")
        plt.xticks(range(0, max_t, 5))
        plt.savefig(f"{ShockedMarketSurvivingRun.OUTPUT_DIRECTORY}/_failures_rationed.svg")

        plt.clf()
        plt.title(f"Number of surviving minus failures not rationed - ShockedMarket p (MC={ShockedMarketSurvivingRun.MC})")
        max_t = 0
        for i, iteration in enumerate(data_of_failures_not_rationed_avg):
            plt.plot(data_of_failures_not_rationed_avg[iteration], "-", color=colors[i],
                     label=str(all_models[i]['p'])[:5])
            if len(data_of_failures_not_rationed_avg[iteration]) > max_t:
                max_t = len(data_of_failures_not_rationed_avg[iteration])
        plt.legend(loc='best', title="p")
        plt.xticks(range(0, max_t, 5))
        plt.savefig(f"{ShockedMarketSurvivingRun.OUTPUT_DIRECTORY}/_failures_not_rationed.svg")
        return [data_of_surviving_banks_avg,
                data_of_failures_rationed_avg,
                data_of_failures_not_rationed_avg], all_models, max_t

    @staticmethod
    def save_surviving_csv(data, array_with_x_values, max_t):

        data_columns= ['surviving','failures_rationed','failures_not_rationed']
        for k in range(len(data_columns)):
            with open(f"{ShockedMarketSurvivingRun.OUTPUT_DIRECTORY}/_{data_columns[k]}.csv", "w") as file:
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
    experiment = runner.do(ShockedMarketSurvivingRun)
    (data, all_models, max_t) = ShockedMarketSurvivingRun.plot_surviving(experiment)
    ShockedMarketSurvivingRun.save_surviving_csv(data, all_models, max_t)
