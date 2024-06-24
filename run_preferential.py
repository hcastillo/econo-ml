#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import random
import warnings

import numpy as np

import interbank
from interbank import Model
from interbank_lenderchange import Preferential
import pandas as pd
from progress.bar import Bar
import os
import matplotlib.pyplot as plt
import argparse
from itertools import product


class Experiment:
    N = 50
    T = 1000
    MC = 10

    OUTPUT_DIRECTORY = "preferential"
    ALGORITHM = Preferential

    config = {  # items should be iterable:
        # "µ": np.linspace(0.7,0.8,num=2),
        # "ω": [0.55,0.6,0.7]
    }

    parameters = {  # items should be iterable:
        "m": np.linspace(1, 50, num=50),
    }

    LENGTH_FILENAME_PARAMETER = 2
    LENGTH_FILENAME_CONFIG = 0

    def plot(self, array_with_data, array_with_x_values, title_x, directory):
        # we plot only x labels 1 of each 10:
        plot_x_values = []
        for j in range(len(array_with_x_values)):
            plot_x_values.append(array_with_x_values[j] if (j % 10 == 0) else " ")
        plot_x_values[-1] = array_with_x_values[-1]
        for i in array_with_data:
            i = i.strip()
            if i != "t":
                mean = []
                standard_deviation = []
                for j in array_with_data[i]:
                    # mean is 0, std is 1:
                    mean.append(j[0])
                    ##standard_deviation.append(
                    ##    abs(np.log(j[1]) / 2 if use_logarithm else j[1] / 2)
                    ##)
                plt.clf()
                # plt.xlabel(title_x)
                title = f"{i}"
                title += f" x={title_x} MC={self.MC}"
                plt.title(title)
                plt.plot(array_with_x_values, mean)
                plt.xticks(plot_x_values, rotation=270, fontsize=5)
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
                    array_with_data[i].append([dataframe[i][j], dataframe['std_'+i][j]])
            for j in dataframe[name_for_x_column]:
                array_with_x_values.append(f"{name_for_x_column}={j}")
            return array_with_data, array_with_x_values
        else:
            return {}, []

    def save(self, array_with_data, array_with_x_values, directory):
        with open(f"{directory}results.csv", "w") as file:
            file.write(
                f"# MC={self.MC} N={self.N} T={self.T}\n"
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

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()
        model.export_datafile = filename
        model.config.lender_change = self.ALGORITHM()
        model.config.lender_change.set_parameter("m", int(execution_parameters["m"]))
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
        return (dict(zip(parameters.keys(), values)) for values in product(*parameters.values()))

    def __filename_clean(self, value, max_length):
        value = str(value)
        for r in "{}',: ":
            value = value.replace(r, "")
        if value.endswith(".0"):
            # integer: 0 at left
            value = value[:-2]
            last_digit = len(value)-1
            while last_digit > 0 and value[last_digit].isdigit():
                last_digit -= 1
            while len(value) <= max_length:
                value = value[:last_digit+1]+'0'+value[last_digit+1:]
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

    def do(self):
        results_to_plot, results_x_axis = self.load(f"{self.OUTPUT_DIRECTORY}/")
        if not results_to_plot:
            self.__verify_directories__()
            progress_bar = Bar(
                "Executing models", max=self.get_num_models()
            )
            progress_bar.update()
            for model_configuration in self.get_models(self.config):
                for model_parameters in self.get_models(self.parameters):
                    result_iteration = pd.DataFrame()
                    filename_for_iteration = self.get_filename_for_iteration(model_parameters, model_configuration)
                    for i in range(self.MC):
                        mc_iteration = random.randint(9999, 20000)
                        if os.path.isfile(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.csv"):
                            result_mc = pd.read_csv(
                                f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.csv", header=2)
                        elif os.path.isfile(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt"):
                            result_mc = interbank.Statistics.read_gdt(
                                f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
                        else:
                            result_mc = self.run_model(
                                f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}",
                                model_configuration, model_parameters, mc_iteration)
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
                    results_x_axis.append(self.__get_title_for(model_configuration, model_parameters))
                    progress_bar.next()
            self.save(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
            progress_bar.finish()
        else:
            print("Loaded data from previous work")
        print("Plotting...")
        self.plot(results_to_plot, results_x_axis, self.__get_title_for(self.config, self.parameters),
                  f"{self.OUTPUT_DIRECTORY}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executes the interbank model")
    parser.add_argument(
        "--do",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=f"Execute the experiment and saves the results in {Experiment.OUTPUT_DIRECTORY}",
    )
    parser.add_argument(
        "--listnames",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Print combinations to generate",
    )
    args = parser.parse_args()

    experiment = Experiment()

    if args.listnames:
        experiment.listnames()
    elif args.do:
        experiment.do()
    else:
        parser.print_help()
