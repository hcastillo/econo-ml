#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
from interbank_lenderchange import ShockedMarket3
import exp_runner
import pandas as pd
from progress.bar import Bar
import warnings
import scipy.stats

class MarketPowerRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 10

    ALGORITHM = ShockedMarket3
    OUTPUT_DIRECTORY = "c:\\experiments\\2_psivar"

    parameters = {
        "p": np.linspace(0.0001, 0.2, num=5),
    }

    config = { 'psi': np.linspace(0.0, 1, num=5) }

    EXTRA_MODEL_CONFIGURATION = { 'psi_endogenous':False }
    
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 6

    SEED_FOR_EXECUTION = 2025


    def do(self, clear_previous_results=False):
        self.log_replaced_data = ""
        results_to_plot = results_comparing = {}
        results_x_axis = []

        if not results_to_plot:
            self.__verify_directories__()
            seeds_for_random = self.generate_random_seeds_for_this_execution()
            progress_bar = Bar(
                "Executing models", max=self.get_num_models()
            )
            progress_bar.update()
            correlation_file = open(f"{self.OUTPUT_DIRECTORY}/results.txt", "w")
            montecarlo_iteration_perfect_correlations = {}
            position_inside_seeds_for_random = 0
            for model_configuration in self.get_models(self.config):
                for model_parameters in self.get_models(self.parameters):
                    result_iteration_to_check = pd.DataFrame()
                    graphs_iteration = []
                    filename_for_iteration = self.get_filename_for_iteration(model_parameters, model_configuration)
                    # first round to load all the self.MC and estimate mean and standard deviation of the series inside
                    # result_iteration:
                    for i in range(self.MC):
                        result_mc = self.load_or_execute_model(model_configuration, model_parameters,
                                                               filename_for_iteration, i, clear_previous_results,
                                                               seeds_for_random[position_inside_seeds_for_random])
                        graphs_iteration.append(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}")
                        result_iteration_to_check = pd.concat([result_iteration_to_check, result_mc])
                        position_inside_seeds_for_random += 1

                    # second round to verify if one of the models should be replaced because it presents abnormal
                    # values comparing to the other (self.MC-1):
                    result_iteration = pd.DataFrame()
                    position_inside_seeds_for_random -= self.MC
                    montecarlo_iteration_perfect_correlation = True
                    for i in range(self.MC):
                        result_mc = self.load_or_execute_model(model_configuration, model_parameters,
                                                               filename_for_iteration, i, clear_previous_results,
                                                               seeds_for_random[position_inside_seeds_for_random])
                        offset = 1
                        while not self.data_seems_ok(filename_for_iteration, i, result_mc, result_iteration_to_check)\
                                and offset <= 10:
                            self.discard_execution_of_iteration(filename_for_iteration, i)
                            result_mc = self.load_or_execute_model(model_configuration, model_parameters,
                                                                   filename_for_iteration, i, clear_previous_results,
                                                                   (seeds_for_random[position_inside_seeds_for_random]
                                                                    + offset))
                            offset += 1

                        # correlation of interest_rate -> bankruptcies
                        if not ((np.all(result_mc['bankruptcies'] == 0) or
                                np.all(result_mc['bankruptcies'] == result_mc['bankruptcies'][0]) or
                                np.all(result_mc['interest_rate'] == 0) or
                                np.all(result_mc['interest_rate'] == result_mc['interest_rate'][0]))):
                            correlation_coefficient, p_value = scipy.stats.pearsonr(result_mc['interest_rate'],
                                                                                    result_mc['bankruptcies'])
                            correlation_coefficient1, p_value1 = scipy.stats.pearsonr(
                                result_mc['interest_rate'][1:],
                                result_mc['bankruptcies'][:-1])

                            correlation_file.write(
                                f"{filename_for_iteration}_{i} = {model_configuration} {model_parameters}\n")
                            correlation_file.write(self.format_correlation_values(0, correlation_coefficient, p_value))
                            correlation_file.write(
                                self.format_correlation_values(1, correlation_coefficient1, p_value1))
                            montecarlo_iteration_perfect_correlation = ( montecarlo_iteration_perfect_correlation and
                                ((correlation_coefficient1>0 and p_value1<=0.10) or
                                 (correlation_coefficient>0 and p_value<=0.10)))
                        position_inside_seeds_for_random += 1
                        result_iteration = pd.concat([result_iteration, result_mc])
                    if montecarlo_iteration_perfect_correlation:
                        montecarlo_iteration_perfect_correlations[
                            str(model_configuration)+ ' '+str(model_parameters) ] = \
                                self.format_correlation_values(1, correlation_coefficient, p_value)

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
                    if self.ALGORITHM.GRAPH_NAME:
                        self.get_statistics_of_graphs(graphs_iteration, results_to_plot, model_parameters)
                    results_x_axis.append(self.__get_title_for(model_configuration, model_parameters))
                    progress_bar.next()
            if montecarlo_iteration_perfect_correlations:
                correlation_file.write('\n\n\nGood models:')
                for good_model in montecarlo_iteration_perfect_correlations:
                    correlation_file.write('%s: %s\n' % (good_model,
                                                         montecarlo_iteration_perfect_correlations[good_model]))
            correlation_file.close()
            progress_bar.finish()
            print(f"Saving results in {self.OUTPUT_DIRECTORY}...")
            self.save_csv(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
            self.save_gdt(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
        else:
            print(f"Loaded data from previous work from {self.OUTPUT_DIRECTORY}")
        results_comparing = self.load_comparing(results_to_plot, results_x_axis)
        if self.log_replaced_data:
            print(self.log_replaced_data)
        print("Plotting...")
        self.plot(results_to_plot, results_x_axis, self.__get_title_for(self.config, self.parameters),
                  f"{self.OUTPUT_DIRECTORY}/", results_comparing)
        self.results_to_plot = results_to_plot
        return results_to_plot, results_x_axis

    def format_correlation_values(self, delay, correlation_coefficient, p_value):
        result = f"\tt={delay} "
        result += f"pearson={correlation_coefficient} p_value={p_value}"
        result += " !!!\n" if  p_value < 0.1 and correlation_coefficient > 0 else "\n"
        return result

    def __get_value_for(self, param):
        result = ''
        if param:
            for i in param.keys():
                if hasattr(param[i], "__len__"):
                    result += i + " "
                else:
                    result += i + '=' + str(param[i]) + " "
        return result

    def __get_title_for(self, param1, param2):
        result = self.__get_value_for(param1) + " " + self.__get_value_for(param2)
        return result.strip()

if __name__ == "__main__":
    runner = exp_runner.Runner()
    runner.do(MarketPowerRun)