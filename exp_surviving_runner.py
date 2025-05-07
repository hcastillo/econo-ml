#!/usr/bin/env python
# coding: utf-8
"""
Executor base class for the interbank model using different values for the lc RestrictedMarket
   to determine number of survivings banks (no replacement of bankrupted)
   depending on p

@author: hector@bith.net
"""
import numpy as np
import pandas as pd
from interbank import Statistics
from interbank_lenderchange import ShockedMarket
import exp_runner
import os
import matplotlib.pyplot as plt


class SurvivingRun(exp_runner.ExperimentRun):
    N = 5
    T = 100
    MC = 1

    ALLOW_REPLACEMENT_OF_BANKRUPTED = False

    ALGORITHM = ShockedMarket
    OUTPUT_DIRECTORY = "not_valid"

    parameters = {  # items should be iterable:
        # "p": np.linspace(0.0001, 1, num=4),

    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    SEED_FOR_EXECUTION = 1

    # which variable is used to determine the color:
    COLORS_VARIABLE = 'p'
    COMPARING_DATA_IN_SURVIVING = False

    # p = [0..1] and range of viridis colors is 10 options, so if value of p=0.177 we will choose #2
    #                                                                      p=0.301 we will choose #3
    _colors = np.array([])

    def get_color(self, all_models, i):
        if self._colors.size == 0:
            self._colors = plt.colormaps['viridis'](np.linspace(0, 1, len(all_models)))
        return self._colors[i]

    @staticmethod
    def determine_average_of_series(array_with_iterations_and_series):
        result_array = {}
        for iteration in array_with_iterations_and_series:  # iterations = p=0.0001, p=0.0002
            if array_with_iterations_and_series[iteration] != []:
                result_array[iteration] = pd.concat(array_with_iterations_and_series[iteration], axis=1).agg("mean", 1)
            else:
                result_array[iteration] = np.nan
        return result_array

    def generate_plot(self, title, output_file, data_to_plot, all_models, max_t,
                      logarithm=False, data_comparing_data_surviving=None):
        plt.clf()
        plt.title(f"{title} with RestrictedMarket p (MC={self.MC})")
        for i, iteration in enumerate(data_to_plot):
            plt.plot(data_to_plot[iteration], "-",
                     color=self.get_color(all_models, i),
                     label=str(all_models[i][self.COLORS_VARIABLE])[:5])
        if data_comparing_data_surviving and self.COMPARING_DATA_IN_SURVIVING:
            for i, iteration in enumerate(data_to_plot):
                plt.plot(data_comparing_data_surviving[iteration], "-",
                         color=self.get_color(all_models, i), alpha=0.3)

        plt.legend(loc='best', title=self.COLORS_VARIABLE)
        # max of ticks we want in x range: 10
        plt.xticks(range(0, max_t, divmod(max_t, 10)[0]))
        if logarithm:
            plt.yscale('log')
            plt.xscale('log')
        plt.savefig(f"{self.OUTPUT_DIRECTORY}/{output_file}")

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
        if isinstance(array, dict):
            result = {}
            for series in array:
                result[series] = SurvivingRun.accumulated_data(array[series])
            return result
        else:
            result = array.copy()
            total = 0
            for idx, item in enumerate(array):
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
        for model_configuration in experiment.get_models(experiment.config):
            for model_parameters in experiment.get_models(experiment.parameters):
                values_of_this_iteration = model_configuration.copy()
                values_of_this_iteration.update(model_parameters)
                all_models.append(values_of_this_iteration)
                filename_for_iteration = experiment.get_filename_for_iteration(model_parameters, model_configuration)
                data_of_surviving_banks[filename_for_iteration] = []
                data_of_failures[filename_for_iteration] = []
                data_of_failures_acum[filename_for_iteration] = []
                data_of_failures_rationed[filename_for_iteration] = []
                data_of_failures_rationed_acum[filename_for_iteration] = []
                for i in range(experiment.MC):
                    if os.path.isfile(f"{experiment.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt"):
                        result_mc = Statistics.read_gdt(
                            f"{experiment.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
                        data_of_surviving_banks[filename_for_iteration].append(result_mc['num_banks'])
                        # bankruptcies and bankruptcies rationed are accumulated data:
                        # [1,2,1,0,1] -> [1,3,4,4,5]
                        data_of_failures[filename_for_iteration].append(result_mc['bankruptcies'])
                        data_of_failures_rationed[filename_for_iteration].append(result_mc['bankruptcy_rationed'])

        # we have now MC different series for each iteration, so let's estimate the average of all the MonteCarlos:
        data_of_surviving_banks_avg = experiment.determine_average_of_series(data_of_surviving_banks)
        data_of_failures_avg = experiment.determine_average_of_series(data_of_failures)
        data_of_failures_acum_avg = experiment.accumulated_data(data_of_failures_avg)
        data_of_failures_rationed_avg = \
            experiment.determine_average_of_series(data_of_failures_rationed)
        data_of_failures_rationed_acum_avg = experiment.accumulated_data(data_of_failures_rationed_avg)

        max_t = experiment.determine_max_t([data_of_surviving_banks_avg, data_of_failures_avg,
                                            data_of_failures_rationed_avg])
        # experiment.generate_plot("Surviving banks", "_surviving.png",
        #                           data_of_surviving_banks_avg, all_models, max_t)
        #experiment.generate_plot("Surviving banks log", "_surviving_log.png",
        #                           data_of_surviving_banks_avg, all_models, max_t, logarithm=True)
        experiment.save_surviving_csv(data_of_surviving_banks_avg, '_surviving', max_t)
        # experiment.generate_plot("Failures", "_failures.png",
        #                            data_of_failures_avg, all_models, max_t)
        # experiment.generate_plot("Failures log", "_failures_log.png",
        #                            data_of_failures_avg, all_models, max_t, logarithm=True)
        # experiment.generate_plot("Failures acum", "_rationed_acum.png",
        #                            data_of_failures_acum_avg, all_models, max_t)
        # experiment.generate_plot("Failures acum log", "_rationed_acum_log.png",
        #                            data_of_failures_acum_avg, all_models, max_t, logarithm=True)
        # experiment.generate_plot("Failures rationed", "_failures_rationed.png",
        #                            data_of_failures_rationed_avg, all_models, max_t)
        experiment.save_surviving_csv(data_of_failures_rationed_avg, '_failures_rationed', max_t)
        # experiment.generate_plot("Failures rationed", "_failures_rationed_log.png",
        #                            data_of_failures_rationed_avg, all_models, max_t, logarithm=True)
        experiment.generate_plot("Failures rationed acum", "_failures_rationed_acum.png",
                                 data_of_failures_rationed_acum_avg, all_models, max_t,
                                 data_comparing_data_surviving=data_of_failures_acum_avg)
        experiment.save_surviving_csv(data_of_failures_rationed_acum_avg, '_failures_rationed_acum', max_t)
        experiment.generate_plot("Failures rationed acum log", "_failures_rationed_acum_log.png",
                                 data_of_failures_rationed_acum_avg, all_models, max_t,
                                 data_comparing_data_surviving=data_of_failures_acum_avg,
                                 logarithm=True)

    def save_surviving_csv(self, data, name, max_t):
        with open(f"{self.OUTPUT_DIRECTORY}/{name}.csv", "w") as file:
            file.write("t")
            for j in data:
                file.write(f";{j}")
            file.write("\n")
            for i in range(max_t):
                file.write(f"{i}")
                for j in data:
                    if i < len(data[j]):
                        file.write(f";{data[j][i]}")
                    else:
                        file.write(f";0.0")
                file.write("\n")


class Runner(exp_runner.Runner):
    pass
