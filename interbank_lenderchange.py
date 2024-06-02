# -*- coding: utf-8 -*-
"""
LenderChange is a class used from interbank.py to control the change of lender in a bank.
   It contains different
    - Boltzman           using Boltzman probability to change
    - InitialStability   using a Barabási–Albert graph to initially assign relationships between banks
    - ShockedMarket      using a Erdos Renyi graph with p=parameter['p']. This method does not allow the evolution in
                            lender for each bank, it replicates the situation after a crisis when banks do not credit
    - Preferential       using a Barabási-Albert with degree m to set up only a set o possible links between
                            banks
@author: hector@bith.net
@date:   05/2023
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import inspect


def determine_algorithm(given_name: str = "default"):
    DEFAULT_METHOD = "Boltzman"

    if given_name == "default":
        given_name = DEFAULT_METHOD

    if given_name == '?':
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj) and obj.__doc__:
                print("\t" + obj.__name__ + (" (default)" if name == DEFAULT_METHOD else '') + ':\n\t', obj.__doc__)
        sys.exit(0)
    else:
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if name.lower() == given_name.lower():
                if inspect.isclass(obj) and obj.__doc__:
                    return obj()
        print(f"not found lenderchange algorithm with name '{given_name}'")
        sys.exit(-1)


node_positions = None
node_colors = None


def draw(graph, new_guru_look_for=False, title=None, show=False):
    """ Draws the graph using a spring layout that reuses the previous one """
    plt.clf()
    if title:
        plt.title(title)
    global node_positions, node_colors
    # if not node_positions:
    guru = None
    if node_positions is None:
        node_positions = nx.spring_layout(graph, pos=node_positions)
    if not hasattr(graph, "type"):
        # guru should not have out edges, and surely by random graphs it has:
        guru, _ = find_guru(graph)
        for (i, j) in list(graph.out_edges(guru)):
            graph.remove_edge(i, j)
    if hasattr(graph, "type") and graph.type == "barabasi_albert_graph":
        graph, guru = get_graph_from_guru(graph.to_undirected())
    if hasattr(graph, "type") and graph.type == "erdos_renyi_graph":
        for node in list(graph.nodes()):
            if not graph.edges(node) and not graph.in_edges(node):
                graph.remove_node(node)
        new_guru_look_for = True
    if not node_colors or new_guru_look_for:
        node_colors = []
        guru, guru_node_edges = find_guru(graph)
        for node in graph.nodes():
            if node == guru:
                node_colors.append('darkorange')
            elif __len_edges(graph, node) == 0:
                node_colors.append('lightblue')
            elif __len_edges(graph, node) == 1:
                node_colors.append('steelblue')
            else:
                node_colors.append('royalblue')
    if hasattr(graph, "type") and graph.type == "barabasi_preferential":
        nx.draw(graph, pos=node_positions, node_color=node_colors, with_labels=True)
    else:
        nx.draw(graph, pos=node_positions, node_color=node_colors, arrowstyle='->', with_labels=True)

    if show:
        plt.show()
    return guru


def __len_edges(graph, node):
    if hasattr(graph, "in_edges"):
        return len(graph.in_edges(node))
    else:
        return len(graph.edges(node))


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


def __get_graph_from_guru(input_graph, output_graph, current_node, previous_node):
    """ It generates a new graph starting from the guru"""
    if __len_edges(input_graph, current_node) > 1:
        for (_, destination) in input_graph.edges(current_node):
            if destination != previous_node:
                __get_graph_from_guru(input_graph, output_graph, destination, current_node)
    if previous_node is not None:
        output_graph.add_edge(current_node, previous_node)


def get_graph_from_guru(input_graph):
    guru, _ = find_guru(input_graph)
    output_graph = nx.DiGraph()
    __get_graph_from_guru(input_graph, output_graph, guru, None)
    return output_graph, guru


def from_graph_to_array_banks(banks_graph, this_model):
    """ From the graph to a lender for each possible bank (or None if no links in the graph)"""
    for node in banks_graph:
        if banks_graph.out_edges(node):
            this_model.banks[node].lender = list(banks_graph.out_edges(node))[0][1]
        else:
            this_model.banks[node].lender = None


# ---------------------------------------------------------
# prototype
class LenderChange:
    def __init__(self):
        self.parameter = {}

    def initialize_bank_relationships(self, this_model):
        """ Call once at initilize() model """
        pass

    def initialize_step(self, this_model):
        """ Call at the beginning of each step """
        pass

    def change_lender(self, this_model, bank, t):
        """ Call at the end of each step before going to the next"""
        pass

    def new_lender(self, this_model, bank):
        """ Describes the mechanism of change"""
        pass

    def describe(self):
        return ""

    def check_parameter(self, name, value):
        """ Called after set_parameter() to verify that the necessary parameters are set """
        return False

    def set_parameter(self, name, value):
        if not value is None:
            if self.check_parameter(name, value):
                self.parameter[name] = value
            else:
                print(f"error with parameter '{name}' for {self.__class__.__name__}")
                sys.exit(-1)


# ---------------------------------------------------------


class Boltzman(LenderChange):
    """It chooses randomly a lender for each bank and in each step changes using Boltzman's probability
         Also it has parameter γ to control the change [0..1], 0=Boltzmann only, 1 everyone will move randomly
    """
    # parameter to control the change of guru: 0 then Boltzmann only, 1 everyone will move randomly
    γ: float = 0.5  # [0..1] gamma
    CHANGE_LENDER_IF_HIGHER = 0.5

    def initialize_bank_relationships(self, this_model):
        this_model.statistics.get_graph(0)

    def change_lender(self, this_model, bank, t):
        """ It uses γ but only after t=20, at the beginning only Boltzmann"""
        possible_lender = self.new_lender(this_model, bank)
        possible_lender_μ = this_model.banks[possible_lender].μ
        if not bank.getLender() is None:
            current_lender_μ = bank.getLender().μ
        else:
            current_lender_μ = 0

        # we can now break old links and set up new lenders, using probability P
        # (equation 8)
        boltzmann = 1 / (1 + math.exp(-this_model.config.β * (possible_lender_μ - current_lender_μ)))

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

    def new_lender(self, this_model, bank):
        """ It gives to the bank a random new lender. It's used initially and from change_lender() """
        # r_i0 is used the first time the bank is created:
        if bank.lender is None:
            bank.rij = np.full(this_model.config.N, this_model.config.r_i0, dtype=float)
            bank.rij[bank.id] = 0
            bank.r = this_model.config.r_i0
            bank.μ = 0
            # if it's just created, only not to be ourselves is enough
            new_value = random.randrange(this_model.config.N - 1)
        else:
            # if we have a previous lender, new should not be the same
            new_value = random.randrange(this_model.config.N - 2 if this_model.config.N > 2 else 1)

        if this_model.config.N == 2:
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
    """ We define a Barabási–Albert graph with 1 edges for each node, and we convert it to a directed graph.
          It is used for initially have more stable links between banks
    """

    CHANGE_LENDER_IF_HIGHER = 0.8

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

    def initialize_bank_relationships(self, this_model):
        self.banks_graph, _ = get_graph_from_guru(nx.barabasi_albert_graph(this_model.config.N, 1))
        self.banks_graph.type = "barabasi_albert_graph"
        if this_model.export_datafile:
            draw(self.banks_graph, new_guru_look_for=True, title=f"barabasi_albert m=1")
            filename = this_model.statistics.get_export_path(this_model.export_datafile).replace('.txt',
                                                                                                 f"_barabasi.png")
            plt.savefig(filename)
        from_graph_to_array_banks(self.banks_graph, this_model)
        return self.banks_graph


class ShockedMarket(LenderChange):
    """ Using an Erdos Renyi graph with p=parameter['p']. This method does not allow the evolution in
          lender for each bank, it replicates the situation after a crisis when banks do not credit

    """

    def check_parameter(self, name, value):
        if name == 'p':
            if isinstance(value, float) and 0 < value < 1:
                return True
            else:
                print("value for 'p' should be a float number > 0 and < 1")
                return False
        else:
            return False

    def initialize_bank_relationships(self, this_model):
        """ It creates a Erdos Renyi graph with p defined in parameter['p']. No changes in relationsships till the end"""
        self.banks_graph = nx.erdos_renyi_graph(this_model.config.N, self.parameter['p'], directed=True)
        self.banks_graph.type = "erdos_renyi_graph"
        if this_model.export_datafile:
            draw(self.banks_graph, new_guru_look_for=True, title=f"erdos_renyi p={self.parameter['p']}")
            filename = this_model.statistics.get_export_path(this_model.export_datafile).replace('.txt',
                                                                                                 f"_erdos_renyi.png")
            plt.savefig(filename)
        from_graph_to_array_banks(self.banks_graph, this_model)
        return self.banks_graph

    def new_lender(self, this_model, bank):
        """ In this LenderChange we never change of lender """
        bank.rij = np.full(this_model.config.N, this_model.config.r_i0, dtype=float)
        bank.rij[bank.id] = 0
        bank.r = this_model.config.r_i0
        bank.μ = 0
        return bank.lender

    def change_lender(self, this_model, bank, t):
        """ In this LenderChange we never change of lender """
        bank.P = 0
        return f"{bank.getId()} maintains lender #{bank.lender} with %1 (ShockedMarket)"


class Preferential(Boltzman):
    """ Using a Barabasi with grade m we restrict to those relations the possibilities to obtain an outgoing link
          (a lender). To improve the specialization of banks, granting 3*C0 to the guru, 2*C0 to its neighbours
          and C to the others. To balance those with 2C0 and 3C0, we will reduce D:
    """
    banks_graph = None
    guru = None

    def check_parameter(self, name, value):
        if name == 'm':
            if isinstance(value, int) and 1 < value:
                return True
            else:
                print("value for 'm' should be an integer > 1")
                return False
        else:
            return False

    def initialize_bank_relationships(self, this_model):
        self.banks_graph_full = nx.barabasi_albert_graph(this_model.config.N, self.parameter['m'])
        self.banks_graph_full.type = "barabasi_preferential"
        if this_model.export_datafile:
            self.guru = draw(self.banks_graph_full, new_guru_look_for=True,
                             title=f"barabasi_pref m={self.parameter['m']}")
            filename = this_model.statistics.get_export_path(this_model.export_datafile).replace('.txt',
                                                                                                 f"_barabasi_pref.png")
            plt.savefig(filename)
        else:
            self.guru = find_guru(self.banks_graph_full)
        self.full_barabasi_extract_random_directed(this_model)
        self.prize_for_good_banks(this_model)
        return self.banks_graph

    def prize_for_good_banks(self, this_model):
        this_model.banks[self.guru].D -= this_model.banks[self.guru].C * 2
        this_model.banks[self.guru].C *= 3
        for (_, node) in self.banks_graph_full.edges(self.guru):
            this_model.banks[node].D -= this_model.banks[node].C
            this_model.banks[node].C *= 2

    def initialize_step(self, this_model):
        self.full_barabasi_extract_random_directed(this_model)

    def full_barabasi_extract_random_directed(self, this_model, current_node=None):
        if current_node is None:
            self.banks_graph = nx.DiGraph()
            self.banks_graph.type = 'barabasi_albert_graph'
            current_node = self.guru
        edges = list(self.banks_graph_full.edges(current_node))
        self.banks_graph.add_node(current_node)
        for (_, destination) in edges[:]:
            if destination not in self.banks_graph.nodes():
                self.full_barabasi_extract_random_directed(this_model, destination)
        try:
            edges.remove((current_node, this_model.banks[current_node].lender))
        except ValueError:
            pass
        candidate = None
        while candidate is None and edges:
            candidate = random.choice(edges)[1]
            if (candidate, current_node) in self.banks_graph.edges():
                edges.remove((current_node, candidate))
                candidate = None
        if not candidate is None:
            self.banks_graph.add_edge(current_node, candidate)
        else:
            self.banks_graph.add_edge(current_node, this_model.banks[current_node].lender)
        return current_node

    def new_lender(self, this_model, bank):
        if bank.lender is None:
            bank.rij = np.full(this_model.config.N, this_model.config.r_i0, dtype=float)
            bank.rij[bank.id] = 0
            bank.r = this_model.config.r_i0
            bank.μ = 0
        if self.banks_graph and self.banks_graph.out_edges():
            return list(self.banks_graph.out_edges(bank.id))[0][1]
        else:
            return None

    def describe(self):
        return f"($\\gamma={self.γ} and change if >{self.CHANGE_LENDER_IF_HIGHER})$"
