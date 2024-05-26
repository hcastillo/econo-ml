# -*- coding: utf-8 -*-
"""
LenderChange is a class used from interbank.py to control the change of lender in a bank.
   It contains different
    - Boltzman           using Boltzman probability to change
    - InitialStability   using a Barabási–Albert graph to initially assign relationships between banks
    - ShockedMarkt       using a Barabási–Albert graph with at least two edges for each node and restrict to it during
                               all simulation
@author: hector@bith.net
@date:   04/2023
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import itertools
import inspect
import statistics


def determine_algorithm(given_name: str):
    DEFAULT = "Boltzman"

    if given_name == "default":
        given_name = DEFAULT

    if given_name == '?':
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj) and obj.__doc__:
                print("\t" + obj.__name__ + (" (default)" if name == DEFAULT else '') + ':\n\t', obj.__doc__)
        sys.exit(0)
    else:
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if name.lower() == given_name.lower():
                if inspect.isclass(obj) and obj.__doc__:
                    return obj()
        print(f"not found lenderchange algorithm with name '{given_name}'")
        sys.exit(-1)


node_position = None
node_colors = None


def __len_edges(graph, node):
    if hasattr(graph, "in_edges"):
        return len(graph.in_edges(node))
    else:
        return len(graph.edges(node))

def draw(graph, force_colors=False):
    """ Draws the graph using a spring layout that reuses the previous one """
    plt.clf()
    global node_position, node_colors
    if not node_position:
        node_position = nx.spring_layout(graph, pos=node_position)
    if not node_colors or force_colors:
        node_colors = []
        guru_node, guru_node_edges = find_guru(graph)
        for node in graph.nodes():
            if node == guru_node:
                node_colors.append('darkorange')
            elif __len_edges(graph, node) == 0:
                node_colors.append('lightblue')
            elif __len_edges(graph, node) == 1:
                node_colors.append('steelblue')
            else:
                node_colors.append('royalblue')
    nx.draw(graph, pos=node_position, node_color=node_colors, with_labels=True)
    plt.show()


def find_guru(graph):
    """It returns the guru ID and also a color_map with red for the guru, lightblue if weight<max/2 and blue others """
    guru_node = None
    guru_node_edges = 0
    for node in graph.nodes():
        edges_node = __len_edges(graph, node)
        if guru_node_edges < edges_node:
            guru_node_edges = edges_node
            guru_node = node
    return guru_node, guru_node_edges


class LenderChange:
    """ Call from Model.initialization()"""

    def initialize_bank_relationships(self, thismodel):
        """ Call at the end of each step before going to the next """
        pass

    def change_lender(self, thismodel, bank, t):
        """ Call at the end of each step before going to the next"""
        pass

    def new_lender(self, thismodel, bank):
        """ Describes the mechanism of change"""
        pass

    def describe(self):
        return "-"


class Boltzman(LenderChange):
    """It chooses randomly a lender for each bank and in each step changes using Boltzman's probability
         Also it has parameter γ to control the change [0..1], 0=Boltzmann only, 1 everyone will move randomly
    """
    # parameter to control the change of guru: 0 then Boltzmann only, 1 everyone will move randomly
    γ: float = 0.5  # [0..1] gamma
    CHANGE_LENDER_IF_HIGHER = 0.5

    def change_lender(self, thismodel, bank, t):
        """ It uses γ but only after t=20, at the beginning only Boltzmann"""
        possible_lender = self.new_lender(thismodel, bank)
        possible_lender_μ = thismodel.banks[possible_lender].μ
        current_lender_μ = bank.getLender().μ

        # we can now break old links and set up new lenders, using probability P
        # (equation 8)
        boltzmann = 1 / (1 + math.exp(-thismodel.config.β * (possible_lender_μ - current_lender_μ)))

        if t < 20:
            # bank.P = 0.35
            # bank.P = random.random()
            bank.P = boltzmann
            # option a) bank.P initially 0.35
            # option b) bank.P randomly
            # option c) with t<=20 boltzmann, and later, stabilize it
        else:
            bank.P_yesterday = bank.P
            # gamma is not sticky/loyalty, persistence of the previous attitude
            bank.P = self.γ * bank.P_yesterday + (1 - self.γ) * boltzmann

        if bank.P >= self.CHANGE_LENDER_IF_HIGHER:
            text_to_return = f"{bank.getId()} new lender is #{possible_lender} from #{bank.lender} with %{bank.P:.3f}"
            bank.lender = possible_lender
        else:
            text_to_return = f"{bank.getId()} maintains lender #{bank.lender} with %{1 - bank.P:.3f}"
        return text_to_return

    def new_lender(self, thismodel, bank):
        """ It gives to the bank a random new lender. It's used initially and from change_lender() """
        # r_i0 is used the first time the bank is created:
        if bank.lender is None:
            bank.rij = np.full(thismodel.config.N, thismodel.config.r_i0, dtype=float)
            bank.rij[bank.id] = 0
            bank.r = thismodel.config.r_i0
            bank.μ = 0
            # if it's just created, only not to be ourselves is enough
            new_value = random.randrange(thismodel.config.N - 1)
        else:
            # if we have a previous lender, new should not be the same
            new_value = random.randrange(thismodel.config.N - 2 if thismodel.config.N > 2 else 1)

        if thismodel.config.N == 2:
            new_value = 1 if bank.id == 0 else 0
        else:
            if new_value >= bank.id:
                new_value += 1
                if bank.lender is not None and new_value >= bank.lender:
                    new_value += 1
            else:
                if bank.lender is not None and new_value >= bank.lender:
                    new_value += 1
                    if new_value >= bank.id:
                        new_value += 1
        return new_value

    def describe(self):
        return f"($\\gamma={self.γ} and change if >{self.CHANGE_LENDER_IF_HIGHER})$"


class InitialStability(Boltzman):
    """ We define a Barabási–Albert graph with 1 edges for each node and we convert to a directed graph """

    def __create_directed_graph_from_barabasi_albert(self, barabasi_albert, result, current_node, previous_node):
        if len(barabasi_albert.edges(current_node)) > 1:
            for (_, destination) in barabasi_albert.edges(current_node):
                if destination != previous_node:
                    self.__create_directed_graph_from_barabasi_albert(barabasi_albert, result,
                                                                      destination, current_node)
                # if not previous_node:
                #    result.add_edge(destination, current_node)
        if previous_node is not None:
            result.add_edge(current_node, previous_node)

    def initialize_bank_relationships(self, model):
        g = nx.barabasi_albert_graph(model.config.N, 1)
        G = nx.DiGraph()
        guru, _ = find_guru(g)
        print(guru)
        self.__create_directed_graph_from_barabasi_albert(g, G, guru, None)
        return g, G


class ShockedMarket(LenderChange):
    def initialize_bank_relationships(self, model, p_step):
        p = p_step
        g = nx.DiGraph()
        g.add_nodes_from([i for i in range(model.config.N)])
        while len(g.edges()) < (model.config.N - 1):
            for u, v in itertools.combinations(g, 2):
                if not g.out_edges(u):
                    if random.random() < p:
                        g.add_edge(u, v)
            p += p_step
        return g


if __name__ == "__main__":
    import interbank

    model = interbank.Model()
    intento = InitialStability()
    b, B = intento.initialize_bank_relationships(model)
    draw(B)
