#!/usr/bin/env python
# coding: utf-8
"""
Executor base class for the interbank model
Parametrized : array of different values for each value in

@author: hector@bith.net
"""
import argparse
import concurrent.futures
import exp_runner
import time
from progress.bar import Bar
import pandas as pd
import numpy as np
import scipy
import warnings

class ExperimentRunParametrized(exp_runner.ExperimentRun):


    # same length as combinations we have in config, for instance config = { 'p':[0,1,2,], 'w':[3,4] }
    # should have 6 items:
    extra_individual_config = [ ]

    extra_individual_parameters = [ ]
    extra_individual_parameters_multiplier = None

    def do(self, clear_previous_results=False, reverse_execution=False):
        self.log_replaced_data = ""
        initial_time = time.perf_counter()
        if clear_previous_results:
            results_to_plot = {}
            results_x_axis = []
        else:
            results_to_plot, results_x_axis = self.load(f"{self.OUTPUT_DIRECTORY}/")
        if not results_to_plot:
            self.verify_directories()
            seeds_for_random = self.generate_random_seeds_for_this_execution()
            progress_bar = Bar(
                "Executing models", max=self.get_num_models()
            )
            progress_bar.update()
            correlation_file = open(f"{self.OUTPUT_DIRECTORY}/results.txt", "w")
            montecarlo_iteration_perfect_correlations = {}
            position_inside_seeds_for_random = 0

            array_of_configs = self.get_models(self.config)
            if reverse_execution:
                array_of_configs = reversed(list(array_of_configs))
            for model_configuration_i, model_configuration in enumerate(array_of_configs):
                array_of_parameters = self.get_models(self.parameters)
                if reverse_execution:
                    array_of_parameters = reversed(list(array_of_parameters))
                if self.extra_individual_config:
                    model_configuration_parametrized = model_configuration | \
                                                   self.extra_individual_config[model_configuration_i]
                else:
                    model_configuration_parametrized = model_configuration

                for model_parameters_j, model_parameters in enumerate(array_of_parameters):
                    result_iteration_to_check = pd.DataFrame()
                    graphs_iteration = []
                    filename_for_iteration = self.get_filename_for_iteration(model_parameters, model_configuration)

                    if self.extra_individual_parameters:
                        if self.extra_individual_parameters_multiplier:
                            for element in self.extra_individual_parameters[model_parameters_j]:
                                self.extra_individual_parameters[model_parameters_j][element] *= (
                                    self.extra_individual_parameters_multiplier)
                        model_configuration_parametrized1 = model_configuration_parametrized | \
                                                       self.extra_individual_parameters[model_parameters_j]
                    else:
                        model_configuration_parametrized1 = model_parameters

                    # first round to load all the self.MC and estimate mean and standard deviation of the series
                    # inside result_iteration:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        results_mc = {executor.submit(self.load_or_execute_model,
                                                      model_configuration_parametrized1,
                                                      model_parameters,
                                                      filename_for_iteration, i,
                                                      clear_previous_results,
                                                      seeds_for_random[i + position_inside_seeds_for_random]):
                                          i for i in range(self.MC)}
                        for future in concurrent.futures.as_completed(results_mc):
                            i = results_mc[future]
                            graphs_iteration.append(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}")
                            result_iteration_to_check = pd.concat([result_iteration_to_check, future.result()])

                    # second round to verify if one of the models should be replaced because it presents abnormal
                    # values comparing to the other (self.MC-1) stored in result_iteration_to_check:
                    result_iteration = pd.DataFrame()
                    position_inside_seeds_for_random -= self.MC
                    montecarlo_iteration_perfect_correlation = True
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        results_mc = {executor.submit(self.load_model_and_rerun_till_ok,
                                                      model_configuration_parametrized1,
                                                      model_parameters,
                                                      filename_for_iteration,
                                                      i, clear_previous_results, seeds_for_random,
                                                      position_inside_seeds_for_random, result_iteration_to_check):
                                          i for i in range(self.MC)}

                        for i, future in enumerate(concurrent.futures.as_completed(results_mc)):
                            result_mc = future.result()
                            # correlation of interest_rate -> bankruptcies
                            if 'bankruptcies' in result_mc and (
                                    not ((np.all(result_mc['bankruptcies'] == 0) or
                                          np.all(result_mc['bankruptcies'] == result_mc['bankruptcies'][0]) or
                                          np.all(result_mc['interest_rate'] == 0) or
                                          np.all(result_mc['interest_rate'] == result_mc['interest_rate'][0])))):
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", scipy.stats.NearConstantInputWarning)
                                    correlation_coefficient, p_value = scipy.stats.pearsonr(result_mc['interest_rate'],
                                                                                            result_mc['bankruptcies'])
                                    correlation_coefficient1, p_value1 = scipy.stats.pearsonr(
                                        result_mc['interest_rate'][1:],
                                        result_mc['bankruptcies'][:-1])
                                    correlation_file.write(
                                        f"{filename_for_iteration}_{i} = {model_configuration} {model_parameters}\n")
                                    correlation_file.write(
                                        exp_runner.format_correlation_values(0, correlation_coefficient, p_value))
                                    correlation_file.write(
                                        exp_runner.format_correlation_values(1, correlation_coefficient1, p_value1))
                                    montecarlo_iteration_perfect_correlation = (
                                            montecarlo_iteration_perfect_correlation and (
                                            (correlation_coefficient1 > 0 and p_value1 <= 0.10) or
                                            (correlation_coefficient > 0 and p_value <= 0.10)))
                            result_iteration = pd.concat([result_iteration, result_mc])

                    if montecarlo_iteration_perfect_correlation:
                        montecarlo_iteration_perfect_correlations[
                            str(model_configuration) + ' ' + str(model_parameters)] = \
                                exp_runner.format_correlation_values(1, correlation_coefficient, p_value)

                    # When it arrives here, all the results are correct and inside the self.MC executions, if one
                    # it is outside the limits of LIMIT_MEAN we have replaced the file and the execution by a new
                    # file and execution using a different seed.
                    # We save now mean and std inside results_to_plot to create later the results:
                    for k in result_iteration.keys():
                        if k.strip() == "t":
                            continue
                        mean_estimated = result_iteration[k].mean()
                        warnings.filterwarnings(
                            "ignore"
                        )  # it generates RuntimeWarning: overflow encountered in multiply
                        std_estimated = result_iteration[k].std()
                        if k in results_to_plot:
                            results_to_plot[k].append([mean_estimated, std_estimated])
                        else:
                            results_to_plot[k] = [[mean_estimated, std_estimated]]
                    results_x_axis.append(self.get_title_for(model_configuration, model_parameters))
                    progress_bar.next()

            progress_bar.finish()
            if montecarlo_iteration_perfect_correlations:
                for perfect_correlation in montecarlo_iteration_perfect_correlations:
                    correlation_file.write(f"{perfect_correlation} : "
                                           f"{montecarlo_iteration_perfect_correlations[perfect_correlation]}\n")
            correlation_file.close()
            print(f"Saving results in {self.OUTPUT_DIRECTORY}...")
            self.save_csv(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
            self.save_gdt(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
        else:
            print(f"Loaded data from previous work from {self.OUTPUT_DIRECTORY}")
        results_comparing, results_comparing2 = self.load_comparing(results_x_axis)
        if self.log_replaced_data:
            print(self.log_replaced_data)
        print("Plotting...")
        self.plot(results_to_plot, results_x_axis, self.get_title_for(self.config, self.parameters),
                  f"{self.OUTPUT_DIRECTORY}/", results_comparing, results_comparing2)
        self.results_to_plot = results_to_plot
        final_time = time.perf_counter()
        print('execution_time: %2.5f secs' % (final_time - initial_time))
        return results_to_plot, results_x_axis


class Runner:
    def __init__(self, experiment_runner: ExperimentRunParametrized):
        self.experiment_runner = experiment_runner
        self.parser = argparse.ArgumentParser(description="Executes MC experiments using interbank model")
        self.parser.add_argument(
            "--do",
            default=False,
            action=argparse.BooleanOptionalAction,
            help=f"Execute the experiment and saves the results in {self.experiment_runner.OUTPUT_DIRECTORY}",
        )
        self.parser.add_argument(
            "--listnames",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Print combinations to generate",
        )
        self.parser.add_argument(
            "--clear_results",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Ignore generated results.csv and create it again",
        )
        self.parser.add_argument(
            "--clear",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Ignore generated models and create them again",
        )
        self.parser.add_argument(
            "--errorbar",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Plot also the errorbar (deviation error)",
        )
        self.parser.add_argument(
            "--reverse",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Execute the experiment in opposite order",
        )

    def do(self):
        args = self.parser.parse_args()
        experiment = self.experiment_runner()
        if args.clear_results:
            experiment.clear_results()
        experiment.error_bar = args.errorbar
        if args.listnames:
            experiment.listnames()
        elif args.do:
            experiment.do(clear_previous_results=args.clear, reverse_execution=args.reverse)
            return experiment
        else:
            self.parser.print_help()
