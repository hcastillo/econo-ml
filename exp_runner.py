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
                file.write(f"{array_with_x_values[i].split('=')[1]}")
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
        variables = VARIABLES(count=f"{2 * len(array_with_data) + 1}")
        variables.append(VARIABLE(name=f"{array_with_x_values[0].split('=')[0]}"))
        for j in array_with_data:
            if j == "leverage":
                j = "leverage_"
            variables.append(VARIABLE(name=f"{j}"))
            variables.append(VARIABLE(name=f"std_{j}"))

        observations = OBSERVATIONS(count=f"{len(array_with_x_values)}", labels="false")
        for i in range(len(array_with_x_values)):
            string_obs = f"{array_with_x_values[i].split('=')[1]}  "
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
        with gzip.open(f"{directory}results.gdt", 'w') as output_file:
            output_file.write(
                b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
            output_file.write(
                lxml.etree.tostring(gdt_result, pretty_print=True, encoding=str).encode('ascii'))

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()
        model.export_datafile = filename
        model.config.lender_change = self.ALGORITHM()
        model.configure(T=self.T, N=self.N, **execution_config)
        model.initialize(seed=seed_random, save_graphs_instants=None,
                         export_datafile=filename,
                         generate_plots=False,
                         export_description=str(model.config) + str(execution_parameters))
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

    def get_statistics_of_graphs(self, graph_files, results):
        communities_not_alone = []
        communities = []
        lengths = []
        gcs = []
        for graph_file in graph_files:
            graph = interbank_lenderchange.load_graph_json(f"{graph_file}_{self.ALGORITHM.GRAPH_NAME}.json")
            graph_communities = interbank_lenderchange.GraphStatistics.communities(graph)
            communities_not_alone.append(interbank_lenderchange.GraphStatistics.communities_not_alone(graph))
            gcs.append(interbank_lenderchange.GraphStatistics.giant_component_size(graph))
            communities.append(len(graph_communities))
            lengths += [len(i) for i in graph_communities]
        if not 'grade_avg' in results:
            results['grade_avg'] = []
            results['communities'] = []
            results['communities_not_alone'] = []
            results['gcs'] = []
        results['grade_avg'].append([0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths)), 0])
        results['communities'].append([(sum(communities)) / len(communities), 0])
        results['communities_not_alone'].append([(sum(communities_not_alone)) / len(communities_not_alone), 0])
        results['gcs'].append([sum(gcs) / len(gcs), 0])

    def load_comparing(self, results_to_plot, results_x_axis):
        results_comparing = None
        if self.COMPARING_DATA:
            results_comparing, results_x_comparing = self.load(f"{self.COMPARING_DATA}/")
            if len(results_x_comparing) != len(results_x_axis) and len(results_x_comparing)!=1:
                results_comparing = None
        return results_comparing

    def data_seems_ok(self, iteration_name:str,
                      new_data:pd.core.frame.DataFrame, array_all_data:pd.core.frame.DataFrame):
        # if 'interest_rate' not in array, means that it's the first execution, so nothing to compare:
        if 'interest_rate' in array_all_data.keys():
            # we obtain the average and std:
            mean_new_data = new_data.interest_rate.mean()
            std_new_data = new_data.interest_rate.std()
            mean_all_data = array_all_data.interest_rate.mean()
            std_all_data = array_all_data.interest_rate.std()
            # if mean_new_data > self.LIMIT_MEAN * mean_all_data:
            #    print(f"\n discarded {iteration_name}: mean {mean_new_data} > {self.LIMIT_MEAN}*{mean_all_data}")
            #    return False
            #elif std_new_data > self.LIMIT_STD * std_all_data:
            #    print(f"\n discarded {iteration_name}: std {std_new_data} > {self.LIMIT_STD}*{std_all_data}")
            #    return False
        return True

    def do(self, clear_previous_results=False):
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
                    result_iteration = pd.DataFrame()
                    graphs_iteration = []
                    filename_for_iteration = self.get_filename_for_iteration(model_parameters, model_configuration)
                    for i in range(self.MC):
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

                        # time to decide what to do with result_mc: let's see the mean() and std():
                        if self.data_seems_ok(f"{filename_for_iteration}_{i}", result_mc, result_iteration):
                            graphs_iteration.append(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}")
                            result_iteration = pd.concat([result_iteration, result_mc])
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
                        self.get_statistics_of_graphs(graphs_iteration, results_to_plot)
                    results_x_axis.append(self.__get_title_for(model_configuration, model_parameters))
                    progress_bar.next()
            progress_bar.finish()
            print(f"Saving results in {self.OUTPUT_DIRECTORY}...")
            self.save_csv(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
            self.save_gdt(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
        else:
            print(f"Loaded data from previous work from {self.OUTPUT_DIRECTORY}")
        print("Plotting...")
        self.plot(results_to_plot, results_x_axis, self.__get_title_for(self.config, self.parameters),
                  f"{self.OUTPUT_DIRECTORY}/", results_comparing)
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
