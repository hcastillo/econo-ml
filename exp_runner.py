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
import gzip
import argparse

class ExperimentRun:
    N = 1
    T = 1
    MC = 1

    LIMIT_MEAN = 3
    LIMIT_STD = 20
    LIMIT_VARIABLE_TO_CHECK = 'interest_rate'

    COMPARING_DATA = ""
    COMPARING_LABEL = "Comparing"

    XTICKS_DIVISOR = 1

    LABEL = "Invalid"
    OUTPUT_DIRECTORY = "Invalid"

    ALGORITHM = interbank_lenderchange.ShockedMarket

    config = {  # items should be iterable:
        # "µ": np.linspace(0.7,0.8,num=2),
        # "ω": [0.55,0.6,0.7]
    }

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
                for j in array_with_data[i]:
                    # mean is 0, std is 1:
                    mean.append(j[0])
                plt.clf()
                title = f"{i}"
                title += f" x={title_x} MC={self.MC}"
                plt.title(title)
                plt.plot(array_with_x_values, mean, "b-",
                         label=self.ALGORITHM.__name__ if array_comparing else "")
                if array_comparing and i in array_comparing:
                    ax = plt.gca()
                    ax.plot(0, array_comparing[i][0][0], "or", label=self.COMPARING_LABEL)
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
        description2 = str(model.config) + str(model.config.lender_change)

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
        header_text = f"MC={self.MC} N={self.N} T={self.T} {self.ALGORITHM.__name__}"
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

    def describe_experiment_parameters(self, model, execution_parameters, seed_random):
        export_description = ""
        for item in dir(model.config):
            if item in execution_parameters:
                export_description += f"{item}={execution_parameters[item]} "
            elif item == 'seed':
                export_description += f"seed={seed_random} "
            elif isinstance(getattr(model.config, item), int) or isinstance(getattr(model.config, item), float):
                export_description += f"{item}={getattr(model.config, item)} "
        return export_description

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()
        model.export_datafile = filename
        model.config.lender_change = self.ALGORITHM()
        model.configure(T=self.T, N=self.N, **execution_config)

        model.initialize(seed=seed_random, save_graphs_instants=None,
                         export_datafile=filename,
                         generate_plots=False,
                         export_description=self.describe_experiment_parameters(model, execution_parameters,
                                                                                seed_random))
        model.simulate_full(interactive=False)
        return model.finish()

    def get_num_models(self):
        models_for_parameters = len(list(self.get_models(self.parameters)))
        models_for_config = len(list(self.get_models(self.config)))
        return models_for_config * models_for_parameters

    def get_models(self, parameters):
        return (dict(zip(parameters.keys(), values)) for values in sorted(product(*parameters.values())))

    def __filename_clean(self, value, max_length):
        value = str(value)
        for r in "{}',: .":
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
        return self.__filename_clean(parameters, self.LENGTH_FILENAME_PARAMETER) + \
            self.__filename_clean(config, self.LENGTH_FILENAME_CONFIG)

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

    def _get_statistics_of_individual_graph(self,filename, communities_not_alone, gcs, communities, lengths):
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
            if len(results_x_comparing) != len(results_x_axis) and len(results_x_comparing)!=1:
                results_comparing = None
        return results_comparing

    def data_seems_ok(self, filename_for_iteration:str, i:int,
                      new_data:pd.core.frame.DataFrame, array_all_data:pd.core.frame.DataFrame):
        # if 'interest_rate' not in array, means that it's the first execution, so nothing to compare:
        if self.LIMIT_VARIABLE_TO_CHECK in array_all_data.keys():
            # we obtain the average and std:
            mean_new_data = new_data.interest_rate.mean()
            std_new_data = new_data.interest_rate.std()
            mean_all_data = array_all_data.interest_rate.mean()
            std_all_data = array_all_data.interest_rate.std()
            if mean_new_data > self.LIMIT_MEAN * mean_all_data:
                self.log_replaced_data += (f"\n discarded {filename_for_iteration}_{i}: mean of "
                                           f"{self.LIMIT_VARIABLE_TO_CHECK} {mean_new_data} >"
                                           f" {self.LIMIT_MEAN}*{mean_all_data}")
                return False
            elif std_new_data > self.LIMIT_STD * std_all_data:
                self.log_replaced_data += (f"\n discarded {filename_for_iteration}_{i}: std of "
                                           f"{self.LIMIT_VARIABLE_TO_CHECK} "
                                           f"{std_new_data} > {self.LIMIT_STD}*{std_all_data}")
                return False
        return True

    def load_or_execute_model(self, model_configuration, model_parameters, filename_for_iteration,
                              i, clear_previous_results):
        mc_iteration = random.randint(9999, 20000)
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
                model_configuration, model_parameters, mc_iteration)
        return result_mc

    def do(self, clear_previous_results=False):
        self.log_replaced_data = ""
        if clear_previous_results:
            results_to_plot = results_comparing = {}
            results_x_axis = []
        else:
            results_to_plot, results_x_axis = self.load(f"{self.OUTPUT_DIRECTORY}/")
            results_comparing = self.load_comparing(results_to_plot, results_x_axis)
        if not results_to_plot:
            self.__verify_directories__()
            progress_bar = Bar(
                "Executing models", max=self.get_num_models()
            )
            progress_bar.update()
            for model_configuration in self.get_models(self.config):
                for model_parameters in self.get_models(self.parameters):
                    result_iteration_to_check = pd.DataFrame()
                    graphs_iteration = []
                    filename_for_iteration = self.get_filename_for_iteration(model_parameters, model_configuration)
                    # first round to load all the self.MC and estimate mean and standard deviation of the series inside
                    # result_iteration:
                    for i in range(self.MC):
                        result_mc = self.load_or_execute_model(model_configuration, model_parameters,
                                                               filename_for_iteration, i, clear_previous_results)
                        graphs_iteration.append(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}")
                        result_iteration_to_check = pd.concat([result_iteration_to_check, result_mc])

                    # second round to verify if one of the models should be replaced because it presents abnormal
                    # values comparing to the other (self.MC-1):
                    result_iteration = pd.DataFrame()
                    for i in range(self.MC):
                        result_mc = self.load_or_execute_model(model_configuration, model_parameters,
                                                               filename_for_iteration, i, clear_previous_results)
                        while not self.data_seems_ok(filename_for_iteration, i, result_mc, result_iteration_to_check):
                            # we should erase the file now and replace by a new execution:
                            try:
                                os.remove(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.csv")
                            except FileNotFoundError:
                                pass
                            try:
                                os.remove(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
                            except FileNotFoundError:
                                pass
                            # new execution with different seed:
                            result_mc = self.load_or_execute_model(model_configuration, model_parameters,
                                                                   filename_for_iteration, i, clear_previous_results)
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
        if self.log_replaced_data:
            print(self.log_replaced_data)
        print("Plotting...")
        self.plot(results_to_plot, results_x_axis, self.__get_title_for(self.config, self.parameters),
                  f"{self.OUTPUT_DIRECTORY}/", results_comparing)
        self.results_to_plot = results_to_plot
        return results_to_plot, results_x_axis


class Runner:
    def do(self, experiment_runner):
        parser = argparse.ArgumentParser(description="Executes interbank model using "+
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
            "--clear",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Ignore generated files and create them again",
        )
        args = parser.parse_args()
        experiment = experiment_runner()
        if args.listnames:
            experiment.listnames()
        elif args.do:
            experiment.do(args.clear)
            return experiment
        else:
            parser.print_help()
