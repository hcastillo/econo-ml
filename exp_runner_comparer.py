#!/usr/bin/env python
# coding: utf-8
"""
Executor base class for the interbank model
@author: hector@bith.net
"""
import exp_runner
import numpy as np
import matplotlib.pyplot as plt

class ExperimentComparerRun(exp_runner.ExperimentRun):
    _colors = np.array([])

    def get_color(self, diff_lines, i):
        if self._colors.size == 0:
            self._colors = plt.colormaps['viridis'](np.linspace(0, 1, len(diff_lines)))
        return self._colors[i]

    def plot(self, array_with_data, array_with_x_values, title_x, directory, array_comparing=None):
        x_values = []
        x_title = '-'
        legend_title = '-'
        for i in self.config:
            x_title = i
            for ii in self.config[i]:
                x_values.append(ii)
        x_values_numbers = list(range(len(x_values)))
        #diff_lines = list(self.get_models(self.parameters))
        diff_lines = []
        for i in self.parameters:
            legend_title = i
            for ii in self.parameters[i]:
                diff_lines.append(ii)

        for i in array_with_data:
            i = i.strip()
            mean = []
            if i != "t" and array_with_data[i] != []:
                plt.clf()
                title = f"{i}"
                title += f" x={title_x} MC={self.MC}"

                for j, line in enumerate(diff_lines):
                    mean = []
                    for jj in range(len(x_values)):
                        mean.append(array_with_data[i][jj * len(diff_lines) + j][0])
                        # if j == 0 and array_comparing and i in array_comparing and j<len(array_comparing[i]):
                        #    mean_comparing.append(array_comparing[i][j][0])

                    plt.plot(x_values_numbers, mean, "-",
                             color=self.get_color(diff_lines, j),
                             label=line)

                if i=='interest_rate':
                    ax = plt.gca()
                    ax.set_yscale('log')
                    plt.title(title + ' (log)')
                else:
                    plt.title(title)
                plt.legend(loc='best', title=legend_title)
                plt.xticks(x_values_numbers, labels=x_values, rotation=270, fontsize=5)
                plt.savefig(f"{directory}{i}.png", dpi=300)

