# -*- coding: utf-8 -*-
"""
Plots pag21 top frequency of policy recommendation comparing mc.txt to ppo.txt

@author: hector@bith.net
@date:   06/2023
"""

import matplotlib.pyplot as plt
import interbank
import argparse
import numpy as np

class Plot:
    data = []
    legend = []
    colors_and_styles = ['black', 'red', 'green', 'blue', 'pink']

    def get_color(self, i):
        return self.colors_and_styles[i % len(self.colors_and_styles)]

    def load_data(self, datafiles):
        for datafile in datafiles.split(","):
            self.data.append([])
            self.legend.append(datafile.replace("_policy","").upper())
            lines = 0
            ignored = 0
            with open(interbank.Statistics.get_export_path(datafile), 'r', encoding="utf-8") as loadfile:
                for line in loadfile.readlines():
                    if not line.strip().startswith("#"):
                        elements = line.split("\t")
                        self.data[-1].append(elements)
                        lines += 1
                    else:
                        ignored += 1
            print(f"{ignored} lines in {datafile}, {lines} incorporated")

    def plot(self, save, file_format):
        #description = "Frequency of ŋ between PPO/MonteCarlo"
        destination = interbank.Statistics.get_export_path(save).replace(".txt", "." + file_format)
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
                        value=value.strip()
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
                plt.bar(x+(0.1*i), yy[i], color=self.get_color(i), width=0.1, label=self.legend[i])
            plt.legend()
            plt.savefig(destination)
            print("plot saved in ", destination)


def run_interactive():
    """
        Run the Plot class
    """
    plot = Plot()
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="cum_fitness1", help=f"Saves the plot")
    parser.add_argument("--extension", default='svg', help="Saves as svg/pdf/jpg/png")
    parser.add_argument("--load", default='ppo_policy,mc_policy',
                        help="Loads the file(s) with the data (sep by comma)")
    args = parser.parse_args()
    if args.load and args.save:
        plot.load_data(args.load)
        plot.plot(args.save, args.extension)
    else:
        print("bad usage: check --load mc,ppo --save freq_mc_ppo")


if __name__ == "__main__":
    run_interactive()

