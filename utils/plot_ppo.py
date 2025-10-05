#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plots:
 - top frequency of policy recommendation comparing mc.txt to ppo.txt (pag21 paper)
 - cumulative fitness

@author: hector@bith.net
@date:   06/2023, 09/2025
"""

import matplotlib.pyplot as plt
import interbank
import argparse
import numpy as np
import pandas as pd
import math


class PlotFrequency:
    data = []
    legend = []
    colors_and_styles = ['black', 'red', 'green', 'blue', 'pink']
    default_interbank = interbank.Model()

    def get_color(self, i):
        return self.colors_and_styles[i % len(self.colors_and_styles)]

    def load_data(self, datafiles):
        for datafile in datafiles.split(","):
            self.data.append([])
            self.legend.append(datafile.replace("_policy", "").upper())
            lines = 0
            ignored = 0
            with open(self.default_interbank.statistics.get_export_path(datafile, '.txt'),
                      'r', encoding="utf-8") as loadfile:
                for line in loadfile.readlines():
                    if not line.strip().startswith("#"):
                        elements = line.split("\t")
                        self.data[-1].append(elements)
                        lines += 1
                    else:
                        ignored += 1
            print(f"{ignored} lines in {datafile}, {lines} incorporated")

    def plot(self, save, file_format):
        destination = self.default_interbank.statistics.get_export_path(save, file_format)
        if len(self.data) == 0:
            print("no data loaded to create the plot")
            return False
        else:
            x_legend = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
            x = np.arange(len(x_legend))
            yy = []
            for data in self.data:
                yy.append([0, 0, 0, 0, 0, 0])
                for values in data:
                    # colum#0 is the time
                    for value in values[1:]:
                        value = value.strip()
                        if value == "0.0":
                            yy[-1][0] += 1
                        else:
                            if value == "1.0":
                                yy[-1][-1] += 1
                            else:
                                print("error no_expected_value: ", value)
            plt.clf()
            plt.xlabel("ŋ")
            plt.xticks(x, x_legend)
            plt.ylabel("Frequency")
            # plt.figure(figsize=(12, 8))
            for i in range(len(yy)):
                plt.bar(x + (0.1 * i), yy[i], color=self.get_color(i), width=0.1, label=self.legend[i])
            plt.legend()
            plt.savefig(destination)
            print("plot saved in ", destination)


class PlotCumulativeFitness:
    t = []
    legend = []
    stdev = []
    num_of_simulations = 0
    confidence_interval = []
    z_confidence_interval = 1.96
    mean = []
    colors = ['black', 'red', 'green', 'blue', 'purple']
    fills = ['gray', 'mistyrose', 'honeydew', 'lavender', 'plum']
    default_interbank = interbank.Model()

    def get_color(self, i):
        return self.colors[i % len(self.colors)]

    def get_fill(self, i):
        return self.fills[i % len(self.fills)]

    @staticmethod
    def convert_to_array_of_numbers(strings, n):
        result = []
        i = 0
        for item in strings:
            result += [float(item) * n]
            i += 1
        if i > PlotCumulativeFitness.num_of_simulations:
            PlotCumulativeFitness.num_of_simulations = i
        return np.array(result)

    def load_data(self, n, datafiles):
        for datafile in datafiles.split(","):
            t = []
            mean = []
            stdev = []
            confidence_interval = []
            self.legend.append(datafile.replace("_fitness", "").upper())
            lines = 0
            ignored = 0
            with open(self.default_interbank.statistics.get_export_path(datafile, '.txt'),
                      'r', encoding="utf-8") as loadfile:
                for line in loadfile.readlines():
                    if not line.strip().startswith("#"):
                        line_strings = line.split("\t")
                        if line_strings[0].strip() != "0":
                            elements = PlotCumulativeFitness.convert_to_array_of_numbers(line_strings[1:], n)
                            t.append(int(line_strings[0]))
                            mean.append(elements.mean())
                            stdev.append(elements.std())
                            confidence_interval.append(
                                self.z_confidence_interval * stdev[-1] / math.sqrt(elements.size))
                            lines += 1
                        else:
                            ignored += 1
                    else:
                        ignored += 1
            self.mean.append(np.array(mean))
            self.confidence_interval.append(np.array(confidence_interval))
            self.stdev.append(np.array(stdev))
            self.t.append(np.array(t))
            print(f"{ignored} lines in {datafile}, {lines} incorporated")

    def plot(self, save, file_format, type):
        destination = self.default_interbank.statistics.get_export_path(save, file_format)
        if len(self.t) == 0:
            print("no data loaded to create the plot")
            return False
        else:
            plt.clf()
            fig, ax = plt.subplots(figsize=(12, 8))
            for i in range(len(self.mean)):
                if type == 'none':
                    mean = self.mean[i]
                    confidence_interval = self.confidence_interval[i]
                else:
                    # we estimate the exponentially weighted moving average (EWMA) and simple
                    # moving average (SMA) using DataFrames:
                    if type == 'sma':
                        mean_aux = pd.DataFrame(self.mean[i]).rolling(20).mean()
                        confidence_interval_aux = pd.DataFrame(self.confidence_interval[i]).rolling(20).mean()
                    else:
                        mean_aux = pd.DataFrame(self.mean[i]).ewm(span=8).mean()
                        confidence_interval_aux = pd.DataFrame(self.confidence_interval[i]).ewm(span=10).mean()
                    mean = mean_aux[0].to_numpy()
                    confidence_interval = confidence_interval_aux[0].to_numpy()

                min_fill = mean - confidence_interval
                max_fill = mean + confidence_interval
                ax.fill_between(self.t[i], min_fill, max_fill,
                                where=(min_fill < max_fill),
                                color=self.get_fill(i), alpha=0.80)
                ax.plot(self.t[i], mean, color=self.get_color(i), label=self.legend[i])
            plt.xlabel("t")
            plt.ylabel(f"Ʃ μ (#{PlotCumulativeFitness.num_of_simulations} simulations)")
            plt.legend()
            plt.savefig(destination)
            print("plot saved in ", destination)


def run_interactive():
    """
        Run the Plot class
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="result", help=f"Saves the plot")
    parser.add_argument("--extension", default='svg', help="Saves as svg/pdf/jpg/png")
    parser.add_argument("--load", default='ppo_policy,mc_policy',
                        help="Loads the file(s) with the data (sep by comma)")
    parser.add_argument("--plot", default="fitness",
                        help="Two options: fitness or frequency. If fitness, remember to put also --n and --type")
    parser.add_argument("--n", type=int, default=50, help=f"Number of banks")
    parser.add_argument("--type", default='sma', help=f"sma or ewma (moving average) or none (raw data)")

    args = parser.parse_args()
    if not args.plot in ("fitness", "frequency"):
        print("bad usage: fitness or frequency are the only two options for --plot")
    elif not args.load or not args.save:
        print("bad usage: check --load mc,ppo --save result")
    else:
        if args.plot == "fitness":
            plot = PlotCumulativeFitness()
            plot.load_data(args.n, args.load)
            plot.plot(args.save, "." + args.extension, args.type)
        else:
            plot = PlotFrequency()
            plot.load_data(args.load)
            plot.plot(args.save, "." + args.extension)


if __name__ == "__main__":
    run_interactive()
