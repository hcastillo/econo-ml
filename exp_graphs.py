#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc ShockedMarket
@author: hector@bith.net
"""
import numpy as np
import pandas as pd
import interbank_lenderchange
import exp_runner
import random
import networkx as nx
import os
import matplotlib.pyplot as plt

class GraphsRun(exp_runner.ExperimentRun):
    N = 100
    T = 1000
    MC = 1

    COMPARING_DATA = "none"
    COMPARING_LABEL = "none"
    OUTPUT_DIRECTORY = "../experiments/shockedmarket.graphs"

    parameters = {
        "p": np.linspace(0.001, 0.99, num=10),
        "outgoings": [1, 20, 99, 100]
    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    SEED_FOR_EXECUTION = 9

    def generate_banks_graph(self, p, outgoings):
        result = nx.DiGraph()
        result.add_nodes_from(list(range(self.N)))
        if outgoings == 1:
            for i in range(self.N):
                if random.random() < p:
                    j = random.randrange(self.N)
                    while j == i:
                        j = random.randrange(self.N)
                    result.add_edge(i, j)
        elif outgoings == GraphsRun.N:
            result = nx.erdos_renyi_graph(GraphsRun.N, p)
        else:
            num_items = 0
            while num_items < self.N:
                i = random.randrange(self.N)
                while len(result.out_edges(i))>outgoings:
                    i = random.randrange(self.N)
                if random.random() < p:
                    j = random.randrange(self.N)
                    while j == i:
                        j = random.randrange(self.N)
                    result.add_edge(i, j)
                num_items += 1
        return result, f"erdos_renyi p={p:5.3} out={outgoings} {interbank_lenderchange.GraphStatistics.describe(result)}"

    def get_statistics_of_graph(self, graph, communities_not_alone, gcs, communities, lengths):
        graph_communities = interbank_lenderchange.GraphStatistics.communities(graph)
        communities_not_alone.append(interbank_lenderchange.GraphStatistics.communities_not_alone(graph))
        gcs.append(interbank_lenderchange.GraphStatistics.giant_component_size(graph))
        communities.append(len(graph_communities))
        lengths += [len(i) for i in graph_communities]

    def get_statistics_of_graphs(self, graph_files, results, model_parameters):
        # model_parameters = {'outgoings': 1, 'p': 0.001}
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
        graph, _ = self.generate_banks_graph( model_parameters['p'], model_parameters['outgoings'])
        self.get_statistics_of_graph(graph, communities_not_alone, gcs, communities, lengths)
        results['grade_avg'].append([0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths)), 0])
        results['communities'].append([(sum(communities)) / len(communities), 0])
        results['communities_not_alone'].append([(sum(communities_not_alone)) / len(communities_not_alone), 0])
        results['gcs'].append([sum(gcs) / len(gcs), 0])

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        return pd.DataFrame()


    @staticmethod
    def generate_plot(output_file, data_to_plot):
        colors = plt.colormaps['viridis'](np.linspace(0, 1, len(GraphsRun.parameters['outgoings'])))
        plt.clf()
        plt.title(f"{output_file}")
        idx = 0

        for j in GraphsRun.parameters['outgoings']:
            plot_this = []
            for i in GraphsRun.parameters['p']:
                plot_this.append(data_to_plot[j][i][0])
            plt.plot(plot_this, "-", color=colors[idx], label=str(f'out={j}'))
            idx += 1

        plt.legend(loc='best', title="output_file")
        plt.savefig(f"{GraphsRun.OUTPUT_DIRECTORY}/{output_file}.png")

    @staticmethod
    def plot_surviving(experiment):
        print("Plotting...")
        # now we open all the .gdt files:

        grade_avg = {}
        communities = {}
        communities_not_alone = {}
        gcs = {}
        for i in GraphsRun.parameters['outgoings']:
            grade_avg[i] = {}
            communities[i] = {}
            communities_not_alone[i] = {}
            gcs[i] = {}

        idx = 0
        for item in experiment.get_models(GraphsRun.parameters):
            grade_avg[item['outgoings']][item['p']] = experiment.results_to_plot['grade_avg'][idx]
            communities[item['outgoings']][item['p']] = experiment.results_to_plot['communities'][idx]
            communities_not_alone[item['outgoings']][item['p']] = experiment.results_to_plot['communities_not_alone'][idx]
            gcs[item['outgoings']][item['p']] = experiment.results_to_plot['gcs'][idx]
            idx += 1

        for item in (grade_avg, communities, communities_not_alone, gcs):
            filename = 'grade_avg' if item==grade_avg else 'communities_not_alone' if item==communities_not_alone else 'gcs' if item==gcs else 'communities'
            with open(f"{GraphsRun.OUTPUT_DIRECTORY}/{filename}.csv", 'w') as file:
                file.write("outgoings")
                for i in GraphsRun.parameters['p']:
                    file.write(f";p={i}")
                file.write("\n")
                for i in GraphsRun.parameters['outgoings']:
                    file.write(f"{i}")
                    for j in GraphsRun.parameters['p']:
                        file.write(f";{item[i][j][0]}")
                    file.write("\n")
            GraphsRun.generate_plot(filename, item)


if __name__ == "__main__":
    runner = exp_runner.Runner()
    experiment = runner.do(GraphsRun)
    os.remove(f"{GraphsRun.OUTPUT_DIRECTORY}/results.gdt")
    GraphsRun.plot_surviving(experiment)
