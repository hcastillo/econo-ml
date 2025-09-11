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

    def __init__(self):
        self.data_of_surviving_banks = {}
        self.data_of_failures_rationed = {}
        self.data_of_failures_rationed_accum = {}
        self.data_of_failures = {}
        self.data_of_failures_accum = {}
        self.all_models = []

    def generate_data_surviving(self):
        print("Generating surviving data...")
        for model_configuration in self.get_models(self.config):
            for model_parameters in self.get_models(self.parameters):
                values_of_this_iteration = model_configuration.copy()
                values_of_this_iteration.update(model_parameters)
                self.all_models.append(values_of_this_iteration)
                filename_for_iteration = self.get_filename_for_iteration(model_parameters, model_configuration)
                self.data_of_surviving_banks[filename_for_iteration] = []
                self.data_of_failures[filename_for_iteration] = []
                self.data_of_failures_accum[filename_for_iteration] = []
                self.data_of_failures_rationed[filename_for_iteration] = []
                self.data_of_failures_rationed_accum[filename_for_iteration] = []
                for i in range(self.MC):
                    if os.path.isfile(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt"):
                        result_mc = Statistics.read_gdt(
                            f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
                        self.data_of_surviving_banks[filename_for_iteration].append(result_mc['num_banks'])
                        # bankruptcies and bankruptcies rationed are accumulated data:
                        # [1,2,1,0,1] -> [1,3,4,4,5]
                        self.data_of_failures[filename_for_iteration].append(result_mc['bankruptcies'])
                        self.data_of_failures_rationed[filename_for_iteration].append(result_mc['bankruptcy_rationed'])
        # we have now MC different series for each iteration, so let's estimate the average of all the MonteCarlos:
        self.data_of_surviving_banks_avg = self.determine_average_of_series(self.data_of_surviving_banks)
        self.data_of_failures_avg = self.determine_average_of_series(self.data_of_failures)
        self.data_of_failures_accum_avg = self.accumulated_data(self.data_of_failures_avg)
        self.data_of_failures_rationed_avg = \
            self.determine_average_of_series(self.data_of_failures_rationed)
        self.data_of_failures_rationed_accum_avg = self.accumulated_data(self.data_of_failures_rationed_avg)
        self.max_t = self.determine_max_t([self.data_of_surviving_banks_avg,
                                           self.data_of_failures_avg,
                                           self.data_of_failures_rationed_avg])

    def plot_surviving(self):
        print("Plotting surviving data...")
        self.save_surviving_csv(self.data_of_surviving_banks_avg, '_surviving', self.max_t)
        self.save_surviving_csv(self.data_of_failures_rationed_avg, '_failures_rationed', self.max_t)
        self.generate_plot("Failures rationed accum", "_failures_rationed_accum.png",
                           self.data_of_failures_rationed_accum_avg, self.all_models, self.max_t,
                           data_comparing_data_surviving=self.data_of_failures_accum_avg)
        self.save_surviving_csv(self.data_of_failures_rationed_accum_avg, '_failures_rationed_accum', self.max_t)
        self.generate_plot("Failures rationed accum log", "_failures_rationed_accum_log.png",
                           self.data_of_failures_rationed_accum_avg, self.all_models, self.max_t,
                           data_comparing_data_surviving=self.data_of_failures_accum_avg,
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
