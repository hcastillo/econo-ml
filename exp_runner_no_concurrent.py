#!/usr/bin/env python
# coding: utf-8
"""
Executor base class for the interbank model
@author: hector@bith.net
"""
import warnings
import exp_runner
import pandas as pd
from progress.bar import Bar
import time

class ExperimentRun(exp_runner.ExperimentRun):
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
            position_inside_seeds_for_random = 0
            array_of_configs = self.get_models(self.config)
            if reverse_execution:
                array_of_configs = reversed(list(array_of_configs))
            for model_configuration in array_of_configs:
                array_of_parameters = self.get_models(self.parameters)
                if reverse_execution:
                    array_of_parameters = reversed(list(array_of_parameters))
                for model_parameters in array_of_parameters:
                    result_iteration_to_check = pd.DataFrame()
                    graphs_iteration = []
                    filename_for_iteration = self.get_filename_for_iteration(model_parameters, model_configuration)
                    # first round to load all the self.MC and estimate mean and standard deviation of the series inside
                    # result_iteration:
                    for i in range(self.MC):
                        result_mc = self.load_or_execute_model(model_configuration, model_parameters,
                                                               filename_for_iteration, i, clear_previous_results,
                                                               seeds_for_random[i + position_inside_seeds_for_random])
                        graphs_iteration.append(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}")
                        result_iteration_to_check = pd.concat([result_iteration_to_check, result_mc])

                    # second round to verify if one of the models should be replaced because it presents abnormal
                    # values comparing to the other (self.MC-1):
                    result_iteration = pd.DataFrame()
                    for i in range(self.MC):
                        result_mc = self.load_or_execute_model(model_configuration, model_parameters,
                                                               filename_for_iteration, i, clear_previous_results,
                                                               seeds_for_random[i + position_inside_seeds_for_random])
                        offset = 1
                        while not self.data_seems_ok(filename_for_iteration, i, result_mc, result_iteration_to_check) \
                                and offset <= 10:
                            self.discard_execution_of_iteration(filename_for_iteration, i)
                            result_mc = self.load_or_execute_model(model_configuration, model_parameters,
                                                                   filename_for_iteration, i, clear_previous_results,
                                                                   (seeds_for_random[
                                                                        i + position_inside_seeds_for_random]
                                                                    + offset))
                            offset += 1
                        result_iteration = pd.concat([result_iteration, result_mc])

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
            print(f"Saving results in {self.OUTPUT_DIRECTORY}...")
            self.save_csv(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
            self.save_gdt(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
        else:
            print(f"Loaded data from previous work from {self.OUTPUT_DIRECTORY}")
        results_comparing = self.load_comparing(results_x_axis)
        if self.log_replaced_data:
            print(self.log_replaced_data)
        print("Plotting...")
        self.plot(results_to_plot, results_x_axis, self.get_title_for(self.config, self.parameters),
                  f"{self.OUTPUT_DIRECTORY}/", results_comparing)
        self.results_to_plot = results_to_plot
        final_time = time.perf_counter()
        print('execution_time: %2.5f secs' % (final_time - initial_time))
        return results_to_plot, results_x_axis


class Runner(exp_runner.Runner):
    pass
