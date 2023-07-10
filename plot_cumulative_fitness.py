# -*- coding: utf-8 -*-
"""
Plots pag21 top frequency of policy recommendation comparing mc.txt to ppo.txt

@author: hector@bith.net
@date:   07/2023
"""

import matplotlib.pyplot as plt
import interbank
import typer
import numpy as np
import pandas as pd
import math


class Plot:
    t = []
    legend = []
    stdev = []
    num_of_simulations = 0
    confidence_interval = []
    z_confidence_interval= 1.96
    mean = []
    colors = ['black', 'red', 'green', 'blue', 'purple']
    fills = ['gray', 'mistyrose', 'honeydew', 'lavender', 'plum']

    def get_color(self, i):
        return self.colors[i % len(self.colors)]

    def get_fill(self, i):
        return self.fills[i % len(self.fills)]

    @staticmethod
    def convert_to_array_of_numbers(strings, n):
        result = []
        i = 0
        for item in strings:
            result += [float(item)*n]
            i += 1
        if i > Plot.num_of_simulations:
            Plot.num_of_simulations = i
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
            with open(interbank.Statistics.get_export_path(datafile), 'r', encoding="utf-8") as loadfile:
                for line in loadfile.readlines():
                    if not line.strip().startswith("#"):
                        line_strings = line.split("\t")
                        if line_strings[0].strip() != "0":
                            elements = Plot.convert_to_array_of_numbers(line_strings[1:], n)
                            t.append(int(line_strings[0]))
                            mean.append(elements.mean())
                            stdev.append(elements.std())
                            confidence_interval.append(
                               self.z_confidence_interval*stdev[-1] / math.sqrt(elements.size))
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
        destination = interbank.Statistics.get_export_path(save).replace(".txt", "." + file_format)
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

                min_fill = mean-confidence_interval
                max_fill = mean+confidence_interval
                ax.fill_between(self.t[i], min_fill, max_fill,
                                where=(min_fill < max_fill),
                                color=self.get_fill(i), alpha=0.80)
                ax.plot(self.t[i], mean, color=self.get_color(i), label=self.legend[i])
            plt.xlabel("t")
            plt.ylabel(f"Ʃ μ (#{Plot.num_of_simulations} simulations)")
            plt.legend()
            plt.savefig(destination)
            print("plot saved in ", destination)


def run_interactive(save: str = typer.Option("cum_fitness1", help=f"Saves the plot"),
                    n: int = typer.Option(50, help="Number of banks"),
                    extension: str = typer.Option("svg", help=f"Saves as svg/pdf/jpg/png"),
                    type: str = typer.Option("sma", help="sma or ewma (moving average) or none (raw data)"),
                    load: str = typer.Option("ppo_fitness,mc_fitness",
                                             help=f"Loads the file(s) with the data (sep by comma)")):
    """
        Run the Plot class
    """
    global plot
    plot = Plot()
    if load and save:
        plot.load_data(n, load)
        plot.plot(save, extension, type)
    else:
        print("bad usage: check --load mc,ppo --save freq_mc_ppo")


app = typer.Typer()
if __name__ == "__main__":
    typer.run(run_interactive)
