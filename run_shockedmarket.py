#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import random
import numpy as np
from interbank import Model
from interbank_lenderchange import ShockedMarket
import warnings
import pandas as pd
from progress.bar import Bar
import matplotlib.pyplot as plt
import os, sys
import math
import argparse
import scipy
import glob, shutil
import nbformat as nbf
from IPython.display import Markdown
from itertools import product


class Experiment:
    N = 10
    T = 100
    MC = 10

    OUTPUT_DIRECTORY = "shocked_market"
    ALGORITHM = ShockedMarket

    config = {  # items should be iterable:
        # "µ": np.linspace(0.7,0.8,num=2),
        # "ω": [0.55,0.6,0.7]
    }

    parameters = {  # items should be iterable:
        "p": np.linspace(0.001, 0.1, num=10),
    }
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()
        model.configure(T=self.T, N=self.N, **execution_config)
        model.initialize(seed=seed_random, save_graphs_instants=None,
                         export_datafile=filename, export_description=str(execution_parameters))
        model.config.lender_change = self.ALGORITHM()
        model.config.lender_change.set_parameter("p", execution_parameters["p"])
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
        for r in "{}',:. ":
            value = value.replace(r, "")
        while len(value) < max_length:
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

    def do(self):
        self.__verify_directories__()
        progress_bar = Bar(
            "Executing models", max=self.get_num_models()
        )
        progress_bar.update()
        for model_configuration in self.get_models(self.config):
            for model_parameters in self.get_models(self.parameters):
                filename_for_iteration = self.get_filename_for_iteration(model_parameters, model_configuration)
                for i in range(self.MC):
                    mc_iteration = random.randint(9999, 20000)
                    if not os.path.isfile(
                            f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.csv"):
                        a = self.run_model(
                            f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}",
                            model_configuration, model_parameters, mc_iteration)
                        print(a)
                    else:
                        pass
                progress_bar.next()
        progress_bar.finish()


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
