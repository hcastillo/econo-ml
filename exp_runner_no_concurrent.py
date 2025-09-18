#!/usr/bin/env python
# coding: utf-8
"""
Executor base class for the interbank model
@author: hector@bith.net
"""
import random
import warnings

import numpy as np
import interbank
from interbank import Model
import interbank_lenderchange
import pandas as pd
from progress.bar import Bar
import os
import matplotlib.pyplot as plt
from itertools import product
import lxml.etree
import lxml.builder
import argparse
import time

class ExperimentRun:
    N = 1
    T = 1
    MC = 1

    LIMIT_OUTLIER = 6

    COMPARING_DATA = ""
    COMPARING_LABEL = "Comparing"

    XTICKS_DIVISOR = 1

    LABEL = "Invalid"
    OUTPUT_DIRECTORY = "Invalid"

    ALGORITHM = interbank_lenderchange.ShockedMarket

    NAME_OF_X_SERIES = None

    EXTRA_MODEL_CONFIGURATION = {}

    ALLOW_REPLACEMENT_OF_BANKRUPTED = True

    config = {  # items should be iterable:
        # "µ": np.linspace(0.7,0.8,num=2),
        # "ω": [0.55,0.6,0.7]
    }

    SEED_FOR_EXECUTION = 2025

    parameters = {  # items should be iterable:
        "p": np.linspace(0.001, 0.100, num=40),
    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    log_replaced_data = ""

    def plot(self, array_with_data, array_with_x_values, title_x, directory, array_comparing=None):
        # we plot only x labels 1 of each 10:
        plot_x_values = []
        for j in range(len(array_with_x_values)):
            plot_x_values.append(array_with_x_values[j] if (j % ExperimentRun.XTICKS_DIVISOR == 0) else " ")
        plot_x_values[-1] = array_with_x_values[-1]
        for i in array_with_data:
            i = i.strip()
            if i != "t":
                mean = []
                mean_comparing = []
                for j in range(len(array_with_data[i])):
                    # mean is 0, std is 1:
                    mean.append(array_with_data[i][j][0])
                    if array_comparing and i in array_comparing and j<len(array_comparing[i]):
                        mean_comparing.append(array_comparing[i][j][0])
                plt.clf()
                title = f"{i}"
                title += f" x={title_x} MC={self.MC}"

                plt.plot(array_with_x_values, mean, "b-",
                         label=self.NAME_OF_X_SERIES if self.NAME_OF_X_SERIES else self.ALGORITHM.__name__ if array_comparing else "")
                logarithm_plot = False
                if array_comparing and i in array_comparing:
                    ax = plt.gca()
                    if len(mean_comparing)==1:
                        ax.plot(0, mean_comparing, "or", label=self.COMPARING_LABEL)
                    else:
                        ax.plot(array_with_x_values, mean_comparing, "r-", label=self.COMPARING_LABEL)
                    if abs(mean[0]-mean_comparing[0])>1e6 and abs(mean[-1]-mean_comparing[-1])>1e6:
                        ax.set_yscale('log')
                        logarithm_plot = True

                plt.title(title + (' (log)' if logarithm_plot else ''))
                plt.xticks(plot_x_values, rotation=270, fontsize=5)
                if array_comparing:
                    plt.legend(loc='best')
                plt.savefig(f"{directory}{i}.png", dpi=300)

    def load(self, directory):
        if os.path.exists(f"{directory}results.csv"):
            dataframe = pd.read_csv(
                f"{directory}results.csv", header=1, delimiter=";")
            array_with_data = {}
            array_with_x_values = []
            name_for_x_column = dataframe.columns[0]
            for i in dataframe.columns[1:]:
                if not i.startswith('std_'):
                    array_with_data[i] = []
            for i in array_with_data.keys():
                if i!='psi.1':
                    for j in range(len(dataframe[i])):
                        array_with_data[i].append([dataframe[i][j], dataframe['std_' + i][j]])
            for j in dataframe[name_for_x_column]:
                array_with_x_values.append(f"{name_for_x_column}={j}")
            return array_with_data, array_with_x_values
        else:
            return {}, []

    def save_csv(self, array_with_data, array_with_x_values, directory):
        with open(f"{directory}results.csv", "w") as file:
            file.write(
                f"# MC={self.MC} N={self.N} T={self.T} {self.ALGORITHM.__name__}\n"
            )
            file.write(array_with_x_values[0].split("=")[0])
            for j in array_with_data:
                file.write(f";{j};std_{j}")
            file.write("\n")
            for i in range(len(array_with_x_values)):
                value_for_line = f"{array_with_x_values[i].split('=')[1]}"
                if ' ' in value_for_line:
                    value_for_line = value_for_line.split(' ')[0]
                file.write(f"{value_for_line}")
                for j in array_with_data:
                    file.write(
                        f";{array_with_data[j][i][0]};{array_with_data[j][i][1]}"
                    )
                file.write("\n")

    def save_gdt(self, array_with_data, array_with_x_values, directory):
        E = lxml.builder.ElementMaker()
        GRETLDATA = E.gretldata
        DESCRIPTION = E.description
        VARIABLES = E.variables
        VARIABLE = E.variable
        OBSERVATIONS = E.observations
        OBS = E.obs

        # used for description of the model inside the gdt:
        model = Model()
        model.config.lender_change = self.ALGORITHM()
        description1 = str(array_with_x_values)
        for item_config in self.config:
            if item_config in model.config:
                setattr(model.config, item_config, None)
        # model.config = values of the default Model without the values that
        # are changed in Runner.config and printed as self.config:
        description2 = str(model.config) + str(self.config) + str(model.config.lender_change)

        variables = VARIABLES(count=f"{2 * len(array_with_data) + 1}")
        variables.append(VARIABLE(name=f"{array_with_x_values[0].split('=')[0]}",
                                  label=f"{description1}"))
        first = True
        for j in array_with_data:
            if j == "leverage":
                j = "leverage_"
            if first:
                variables.append(VARIABLE(name=f"{j}", label=f"{description2}"))
            else:
                variables.append(VARIABLE(name=f"{j}"))
            first = False
            variables.append(VARIABLE(name=f"std_{j}"))

        observations = OBSERVATIONS(count=f"{len(array_with_x_values)}", labels="false")
        for i in range(len(array_with_x_values)):
            value_for_line = f"{array_with_x_values[i].split('=')[1]}"
            if ' ' in value_for_line:
                value_for_line = value_for_line.split(' ')[0]
            string_obs = f"{value_for_line}  "
            for j in array_with_data:
                string_obs += f"{array_with_data[j][i][0]}  {array_with_data[j][i][1]}  "
            observations.append(OBS(string_obs))
        header_text = f"MC={self.MC} N={self.N} T={self.T} {self.ALGORITHM.__name__ if not self.NAME_OF_X_SERIES else self.NAME_OF_X_SERIES}"
        gdt_result = GRETLDATA(
            DESCRIPTION(header_text),
            variables,
            observations,
            version="1.4", name='prueba', frequency="special:1", startobs="1",
            endobs=f"{len(array_with_x_values)}", type="cross-section"
        )
        with open(f"{directory}results.gdt", 'b+w') as output_file:
            output_file.write(
                b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
            output_file.write(
                lxml.etree.tostring(gdt_result, pretty_print=True, encoding=str).encode('ascii'))

    def set_lender_change(self, execution_parameters):
        p = execution_parameters['p'] if 'p' in execution_parameters else interbank.LENDER_CHANGE_DEFAULT_P
        m = execution_parameters['m'] if 'm' in execution_parameters else None
        return interbank_lenderchange.determine_algorithm(self.ALGORITHM(), p, m)

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()
        model.export_datafile = filename
        print("pasa")
        model.config.lender_change = self.set_lender_change(execution_parameters)

        model.configure(T=self.T, N=self.N,
                        allow_replacement_of_bankrupted=self.ALLOW_REPLACEMENT_OF_BANKRUPTED, **execution_config)
        model.configure(**self.EXTRA_MODEL_CONFIGURATION)
        model.initialize(seed=seed_random, save_graphs_instants=None,
                         export_datafile=filename,
                         generate_plots=False)
        model.simulate_full(interactive=False)
        return model.finish()

    def get_num_models(self):
        models_for_parameters = len(list(self.get_models(self.parameters)))
        models_for_config = len(list(self.get_models(self.config)))
        return models_for_config * models_for_parameters

    def get_models(self, parameters):
        return (dict(zip(parameters.keys(), values)) for values in sorted(product(*parameters.values())))

    def __filename_clean(self, value, max_length):
        value = str(value).replace("np.float64(","").replace("np.float(","")
        for r in "{}[]()',: .":
            value = value.replace(r, "")
        if value.endswith(".0"):
            # integer: 0 at left
            value = value[:-2]
            last_digit = len(value) - 1
            while last_digit > 0 and value[last_digit].isdigit():
                last_digit -= 1
            while len(value) <= max_length:
                value = value[:last_digit + 1] + '0' + value[last_digit + 1:]
        else:
            # float: 0 at right
            value = value.replace(".", "")
            while len(value) <= max_length:
                value += "0"
            if len(value) > max_length:
                value = value[:max_length]
        return value

    def get_filename_for_iteration(self, parameters, config):
        return self.get_filename_for_parameters( parameters) + self.get_filename_for_config(config)

    def get_filename_for_config(self, config):
        return self.__filename_clean(config, self.LENGTH_FILENAME_CONFIG)

    def get_filename_for_parameters(self, parameters):
        return self.__filename_clean(parameters, self.LENGTH_FILENAME_PARAMETER)

    def __verify_directories__(self):
        if not os.path.isdir(self.OUTPUT_DIRECTORY):
            os.mkdir(self.OUTPUT_DIRECTORY)

    def listnames(self):
        num = 0
        for config in self.get_models(self.config):
            for parameter in self.get_models(self.parameters):
                model_name = self.get_filename_for_iteration(parameter, config)
                print(model_name)
                num += 1
        print("total: ", num)

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

    def _get_statistics_of_individual_graph(self, filename, communities_not_alone, gcs, communities, lengths):
        graph = interbank_lenderchange.load_graph_json(filename)
        graph_communities = interbank_lenderchange.GraphStatistics.communities(graph)
        communities_not_alone.append(interbank_lenderchange.GraphStatistics.communities_not_alone(graph))
        gcs.append(interbank_lenderchange.GraphStatistics.giant_component_size(graph))
        communities.append(len(graph_communities))
        lengths += [len(i) for i in graph_communities]

    def get_statistics_of_graphs(self, graph_files, results, model_parameters):
        communities_not_alone = []
        communities = []
        lengths = []
        gcs = []
        # if results has not yet an array with graph statistics, we incorporate it:
        if not 'grade_avg' in results:
            results['grade_avg'] = []
            results['communities'] = []
            results['communities_not_alone'] = []
            results['gcs'] = []
        for graph_file in graph_files:
            filename = f"{graph_file}_{self.ALGORITHM.GRAPH_NAME}.json"
            # we need to obtain the stats of all graph_files, but maybe we have not only a graph for each model
            # analyzed, also a graph for each step t, so a "_0.json" will be present:
            if not os.path.exists(filename) and os.path.exists(filename.replace(".json", "_0.json")):
                for i in range(0, self.T):
                    filename = f"{graph_file}_{self.ALGORITHM.GRAPH_NAME}_{i}.json"
                    try:
                        self._get_statistics_of_individual_graph(filename,
                                                                 communities_not_alone, gcs, communities, lengths)
                    except FileNotFoundError:
                        break
            else:
                self._get_statistics_of_individual_graph(filename, communities_not_alone, gcs, communities, lengths)

        results['grade_avg'].append([0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths)), 0])
        results['communities'].append([0 if len(communities) == 0 else (sum(communities)) / len(communities), 0])
        results['communities_not_alone'].append([0 if len(communities_not_alone) == 0 else
                                                 (sum(communities_not_alone)) / len(communities_not_alone), 0])
        results['gcs'].append([0 if len(gcs) == 0 else sum(gcs) / len(gcs), 0])

    def load_comparing(self, results_to_plot, results_x_axis):
        results_comparing = None
        if self.COMPARING_DATA:
            results_comparing, results_x_comparing = self.load(f"{self.COMPARING_DATA}/")
            if len(results_x_comparing) not in (len(results_x_axis),1):
                results_comparing = None
        return results_comparing

    def data_seems_ok(self, filename_for_iteration: str, i: int,
                      individual_execution: pd.core.frame.DataFrame, array_all_data: pd.core.frame.DataFrame):
        # if 'interest_rate' not in array, means that it's the first execution, so nothing to compare:
        for k in array_all_data.keys():
            if k.strip() == "t":
                continue
            mean_individual_execution = individual_execution[k].mean()

            # mean_estimated = array_all_data[k].mean()
            # warnings.filterwarnings(
            #     "ignore"
            # )  # it generates RuntimeWarning: overflow encountered in multiply
            # std_estimated = array_all_data[k].std()
            # we discard outliers: whatever is over μ±3σ or under μ±3σ:
            # if (not np.isnan(mean_estimated) and not np.isnan(std_estimated) and
            #     not np.isnan(mean_individual_execution) and
            #     not ((mean_estimated - self.LIMIT_OUTLIER * std_estimated) <=
            #     mean_individual_execution <=
            #     (mean_estimated + self.LIMIT_OUTLIER * std_estimated))):
            #     return False
            means = []
            for i in range(self.MC):
                means.append(array_all_data[k][i * self.T:(i * self.T) + (self.T-1)].mean())
            #Z-Score Method
            #mean = np.mean(means)
            #std = np.std(means)
            #if ( mean_individual_execution - mean ) > 7:
            #    print(f"{k} {filename_for_iteration}:{i} {mean_individual_execution} mean:{mean} std:{std}")
            q1 = np.percentile(means, 25)
            q3 = np.percentile(means, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.LIMIT_OUTLIER * iqr
            upper_bound = q3 + self.LIMIT_OUTLIER * iqr
            # IQR Method (Non-parametric, more robust)
            # How it works:
            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            # Compute IQR = Q3 - Q1
            # Outliers are values outside [Q1 - 1.5IQR, Q3 + 1.5IQR]
            if (not np.isnan(mean_individual_execution) and not np.isnan(iqr) and
                not (lower_bound <= mean_individual_execution <= upper_bound) and
                not (lower_bound==upper_bound)):
                print(f"{k} {filename_for_iteration}:{i} {lower_bound:.3f} <= {mean_individual_execution:.3f} <= {upper_bound:.3f}")
                return False
        return True

    def load_or_execute_model(self, model_configuration, model_parameters, filename_for_iteration,
                              i, clear_previous_results, seed_for_this_model):
        if (os.path.isfile(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.csv")
                and not clear_previous_results):
            result_mc = pd.read_csv(
                f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.csv", header=2)
        elif (os.path.isfile(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
              and not clear_previous_results):
            result_mc = interbank.Statistics.read_gdt(
                f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
        else:
            result_mc = self.run_model(
                f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}",
                model_configuration, model_parameters, seed_for_this_model)
        return result_mc

    def do(self, clear_previous_results=False):
        self.log_replaced_data = ""
        initial_time = time.perf_counter()
        if clear_previous_results:
            results_to_plot = {}
            results_x_axis = []
        else:
            results_to_plot, results_x_axis = self.load(f"{self.OUTPUT_DIRECTORY}/")
        if not results_to_plot:
            self.__verify_directories__()
            seeds_for_random = self.generate_random_seeds_for_this_execution()
            progress_bar = Bar(
                "Executing models", max=self.get_num_models()
            )
            progress_bar.update()
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
                                                               seeds_for_random[i+position_inside_seeds_for_random])
                        graphs_iteration.append(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}")
                        result_iteration_to_check = pd.concat([result_iteration_to_check, result_mc])


                    # second round to verify if one of the models should be replaced because it presents abnormal
                    # values comparing to the other (self.MC-1):
                    result_iteration = pd.DataFrame()
                    for i in range(self.MC):
                        result_mc = self.load_or_execute_model(model_configuration, model_parameters,
                                                               filename_for_iteration, i, clear_previous_results,
                                                               seeds_for_random[i+position_inside_seeds_for_random])
                        offset = 1
                        while not self.data_seems_ok(filename_for_iteration, i, result_mc, result_iteration_to_check)\
                                and offset <= 10:
                            self.discard_execution_of_iteration(filename_for_iteration, i)
                            result_mc = self.load_or_execute_model(model_configuration, model_parameters,
                                                                   filename_for_iteration, i, clear_previous_results,
                                                                   (seeds_for_random[i+position_inside_seeds_for_random]
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
                    if self.ALGORITHM.GRAPH_NAME:
                        self.get_statistics_of_graphs(graphs_iteration, results_to_plot, model_parameters)
                    results_x_axis.append(self.__get_title_for(model_configuration, model_parameters))
                    progress_bar.next()

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
        final_time = time.perf_counter()
        print('execution_time: %2.5f secs' % (final_time - initial_time))
        return results_to_plot, results_x_axis

    def generate_random_seeds_for_this_execution(self):
        seeds_for_random = []
        random.seed(self.SEED_FOR_EXECUTION)
        for _ in self.get_models(self.config):
            for _ in self.get_models(self.parameters):
                for i in range(self.MC):
                    seeds_for_random.append(random.randint(1000, 99999))
        return seeds_for_random

    def discard_execution_of_iteration(self, filename_for_iteration, i):
        # we should erase or remove the file and then we will generate a new execution with a different seed:
        if os.path.exists(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.csv"):
            base, ext = os.path.splitext(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.csv")
        elif os.path.exists(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt"):
            base, ext = os.path.splitext(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
        offset = 1
        while True:
            new_name = f"{base}_discarded{offset}{ext}"
            if not os.path.exists(new_name):
                os.rename(f"{base}{ext}", new_name)
                break
            offset += 1

    def clear_results(self):
        try:
            os.remove( self.OUTPUT_DIRECTORY + '/results.csv')
        except FileNotFoundError:
            pass
        try:
            os.remove( self.OUTPUT_DIRECTORY + '/results.gdt')
        except FileNotFoundError:
            pass


class Runner:
    def do(self, experiment_runner):
        parser = argparse.ArgumentParser(description="Executes interbank model using " +
                                                     experiment_runner.__name__)
        parser.add_argument(
            "--do",
            default=False,
            action=argparse.BooleanOptionalAction,
            help=f"Execute the experiment and saves the results in {experiment_runner.OUTPUT_DIRECTORY}",
        )
        parser.add_argument(
            "--listnames",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Print combinations to generate",
        )
        parser.add_argument(
            "--clear_results",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Ignore generated results.csv and create it again",
        )
        parser.add_argument(
            "--clear",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Ignore generated models and create them again",
        )
        args = parser.parse_args()
        experiment = experiment_runner()
        if args.clear_results:
            experiment.clear_results()
        if args.listnames:
            experiment.listnames()
        elif args.do:
            experiment.do(args.clear)
            return experiment
        else:
            parser.print_help()
