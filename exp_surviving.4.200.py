#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc RestrictedMarket
   to determine number of survivings banks (no replacement of bankrupted)
   depending on p

@author: hector@bith.net
"""
import numpy as np
import pandas as pd
from interbank import Model, Statistics
from interbank_lenderchange import ShockedMarket
import exp_runner
import os
import matplotlib.pyplot as plt


class RestrictedMarketSurvivingRun(exp_runner.ExperimentRun):
    N = 100
    T = 1000
    MC = 15


    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "c:\\experiments\\surviving.4.200"

    parameters = {  # items should be iterable:
        "p": np.linspace(0.0001, 1, num=4),
        # "p": {0.0001, 0.3333, 0.6666}

    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    seed = 918994
    seed_offset = 1

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()
        model.export_datafile = filename
        model.config.lender_change = self.ALGORITHM()
        model.config.lender_change.set_parameter("p", execution_parameters["p"])
        model.configure(T=self.T, allow_replacement_of_bankrupted=False,
                        N=self.N, **execution_config)
        RestrictedMarketSurvivingRun.seed_offset += 1
        model.initialize(seed=(self.seed + RestrictedMarketSurvivingRun.seed_offset), save_graphs_instants=None,
                         export_datafile=filename,
                         generate_plots=False,
                         export_description=self.describe_experiment_parameters(model, execution_parameters,
                                                                                seed_random))
        model.simulate_full(interactive=False)
        return model.finish()

    # p = [0..1] and range of viridis colors is 10 options, so if value of p=0.177 we will choose #2
    #                                                                      p=0.301 we will choose #3
    colors = plt.colormaps['viridis'](np.linspace(0, 1, 11))

    @staticmethod
    def get_color(value_of_p):
        idx = int(round(value_of_p*10, 0))
        return RestrictedMarketSurvivingRun.colors[idx]


    @staticmethod
    def determine_average_of_series(array_with_iterations_and_series):
        result_array = {}
        for iteration in array_with_iterations_and_series:  # iterations = p=0.0001, p=0.0002
            result_array[iteration] = pd.concat(array_with_iterations_and_series[iteration],axis=1).agg("mean", 1)
        return result_array

    @staticmethod
    def generate_plot(title, output_file, data_to_plot, all_models, max_t, logarithm=False):
        plt.clf()
        plt.title(f"{title} with RestrictedMarket p (MC={RestrictedMarketSurvivingRun.MC})")
        max_t = 0
        for i, iteration in enumerate(data_to_plot):
            plt.plot(data_to_plot[iteration], "-",
                     color=RestrictedMarketSurvivingRun.get_color(all_models[i]['p']),
                     label=str(all_models[i]['p'])[:5])
            if len(data_to_plot[iteration]) > max_t:
                max_t = len(data_to_plot[iteration])
        plt.legend(loc='best', title="p")
        # max of ticks we want in x range: 10
        plt.xticks(range(0, max_t, divmod(max_t, 10)[0]))
        if logarithm:
            plt.yscale('log')
            plt.xscale('log')
        plt.savefig(f"{RestrictedMarketSurvivingRun.OUTPUT_DIRECTORY}/{output_file}")

    @staticmethod
    def determine_max_t(data_array):
        max_t = 0
        for data in data_array:
            for i, iteration in enumerate(data):
                if len(data[iteration]) > max_t:
                    max_t = len(data[iteration])
        return max_t

    @staticmethod
    def accumulated_data(array):
        result = array.copy()
        total = 0
        for idx,item in enumerate(array):
            total += array[idx]
            result[idx] = total
        return result

    @staticmethod
    def plot_surviving(experiment):
        print("Plotting surviving...")
        # now we open all the .gdt files:
        data_of_surviving_banks = {}
        data_of_failures_rationed = {}
        data_of_failures_rationed_acum = {}
        data_of_failures = {}
        data_of_failures_acum = {}

        all_models = []
        for model_configuration in experiment.get_models(RestrictedMarketSurvivingRun.parameters):
            all_models.append(model_configuration)
            filename_for_iteration = experiment.get_filename_for_iteration(model_configuration, {})
            data_of_surviving_banks[filename_for_iteration] = []
            data_of_failures[filename_for_iteration] = []
            data_of_failures_acum[filename_for_iteration] = []
            data_of_failures_rationed[filename_for_iteration] = []
            data_of_failures_rationed_acum[filename_for_iteration] = []
            for i in range(RestrictedMarketSurvivingRun.MC):
                if os.path.isfile(f"{RestrictedMarketSurvivingRun.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt"):
                    result_mc = Statistics.read_gdt(
                        f"{RestrictedMarketSurvivingRun.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
                    data_of_surviving_banks[filename_for_iteration].append(result_mc['num_banks'])
                    # bankruptcies and bankruptcies rationed are accumulated data:
                    # [1,2,1,0,1] -> [1,3,4,4,5]
                    data_of_failures[filename_for_iteration].append(result_mc['bankruptcies'])
                    data_of_failures_acum[filename_for_iteration].append(
                        RestrictedMarketSurvivingRun.accumulated_data(result_mc['bankruptcies']))
                    data_of_failures_rationed[filename_for_iteration].append(result_mc['bankruptcy_rationed'])
                    data_of_failures_rationed_acum[filename_for_iteration].append(
                        RestrictedMarketSurvivingRun.accumulated_data(result_mc['bankruptcy_rationed']))

        # we have now MC different series for each iteration, so let's estimate the average of all the MonteCarlos:
        data_of_surviving_banks_avg = RestrictedMarketSurvivingRun.determine_average_of_series(data_of_surviving_banks)
        data_of_failures_avg = RestrictedMarketSurvivingRun.determine_average_of_series(data_of_failures)
        data_of_failures_acum_avg = RestrictedMarketSurvivingRun.determine_average_of_series(data_of_failures_acum)
        data_of_failures_rationed_avg =\
            RestrictedMarketSurvivingRun.determine_average_of_series(data_of_failures_rationed)
        data_of_failures_rationed_acum_avg =\
            RestrictedMarketSurvivingRun.determine_average_of_series(data_of_failures_rationed_acum)

        max_t = RestrictedMarketSurvivingRun.determine_max_t([data_of_surviving_banks_avg,data_of_failures_avg,
                                                              data_of_failures_rationed_avg])
        RestrictedMarketSurvivingRun.generate_plot("Surviving banks", "_surviving.png",
                                                   data_of_surviving_banks_avg, all_models, max_t)
        RestrictedMarketSurvivingRun.generate_plot("Surviving banks log", "_surviving_log.png",
                                                   data_of_surviving_banks_avg, all_models, max_t, logarithm=True)
        RestrictedMarketSurvivingRun.generate_plot("Failures", "_failures.png",
                                                   data_of_failures_avg, all_models, max_t)
        RestrictedMarketSurvivingRun.generate_plot("Failures log", "_failures_log.png",
                                                   data_of_failures_avg, all_models, max_t, logarithm=True)
        RestrictedMarketSurvivingRun.generate_plot("Failures acum", "_rationed_acum.png",
                                                   data_of_failures_acum_avg, all_models, max_t)
        RestrictedMarketSurvivingRun.generate_plot("Failures acum log", "_rationed_acum_log.png",
                                                   data_of_failures_acum_avg, all_models, max_t, logarithm=True)
        RestrictedMarketSurvivingRun.generate_plot("Failures rationed", "_rationed.png",
                                                   data_of_failures_rationed_avg, all_models, max_t)
        RestrictedMarketSurvivingRun.generate_plot("Failures rationed", "_rationed_log.png",
                                                   data_of_failures_rationed_avg, all_models, max_t, logarithm=True)
        RestrictedMarketSurvivingRun.generate_plot("Failures rationed acum", "_rationed_acum.png",
                                                   data_of_failures_rationed_acum_avg, all_models, max_t)
        RestrictedMarketSurvivingRun.generate_plot("Failures rationed acum log", "_rationed_acum_log.png",
                                                   data_of_failures_rationed_acum_avg, all_models, max_t, logarithm=True)
        return [data_of_surviving_banks_avg,
                data_of_failures_rationed_avg,
                data_of_failures_avg], all_models, max_t

    @staticmethod
    def save_surviving_csv(data, max_t):
        data_columns = ['surviving', 'failures_rationed', 'failures']
        for k in range(len(data_columns)):
            with open(f"{RestrictedMarketSurvivingRun.OUTPUT_DIRECTORY}/_{data_columns[k]}.csv", "w") as file:
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
    experiment = runner.do(RestrictedMarketSurvivingRun)
    (data, all_models, max_t) = RestrictedMarketSurvivingRun.plot_surviving(experiment)
    RestrictedMarketSurvivingRun.save_surviving_csv(data, max_t)
