# -*- coding: utf-8 -*-
"""
LenderChange is a class used from interbank.py to control the change of lender in a bank.
   It contains different
    - Boltzman           using Boltzman probability to change
    - InitialStability   using a Barabási–Albert graph to initially assign relationships between banks
    - Preferential       using a Barabási-Albert with m edges for each new nde to set up only a set o links between
                            banks. In each step, new links between are set up based on the initial graph
    - RestrictedMarket   using an Erdos Renyi graph with p=parameter['p']. This method does not allow the evolution in
                            lender for each bank, it replicates the situation after a crisis when banks do not credit
    - ShockedMarket      using an Erdos Renyi graph with p=parameter['p']. In each step a new random Erdos Renyi is
                            used
    - Small World        using a Watts and Strogatz algorithm with parameter['p'] and k=5
                            (Each node is joined with its k nearest neighbors in a ring topology)
@author: hector@bith.net
@date:   05/2023
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import sys
import inspect
import json
import warnings


def determine_algorithm(given_name: str = "default"):
    DEFAULT_METHOD = "Boltzman"

    if given_name == "default":
        warnings.warn(f"selected default method {DEFAULT_METHOD}")
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


def draw(original_graph, new_guru_look_for=False, title=None, show=False):
    """ Draws the graph using a spring layout that reuses the previous one layout, to show similar position for
        the same ids of nodes along time. If the graph is undirected (Barabasi) then no """
    graph_to_draw = original_graph.copy()
    plt.clf()
    if title:
        plt.title(title)
    global node_positions, node_colors
    # if not node_positions:
    guru = None
    if node_positions is None:
        node_positions = nx.spring_layout(graph_to_draw, pos=node_positions)
    if not hasattr(original_graph, "type") and original_graph.is_directed():
        # guru should not have out edges, and surely by random graphs it has:
        guru, _ = find_guru(graph_to_draw)
        for (i, j) in list(graph_to_draw.out_edges(guru)):
            graph_to_draw.remove_edge(i, j)
    if hasattr(original_graph, "type") and original_graph.type == "barabasi_albert":
        graph_to_draw, guru = get_graph_from_guru(graph_to_draw.to_undirected())
    if hasattr(original_graph, "type") and original_graph.type == "erdos_renyi":
        for node in list(graph_to_draw.nodes()):
            if not graph_to_draw.edges(node) and not graph_to_draw.in_edges(node):
                graph_to_draw.remove_node(node)
        new_guru_look_for = True
    if not node_colors or new_guru_look_for:
        node_colors = []
        guru, guru_node_edges = find_guru(graph_to_draw)
        for node in graph_to_draw.nodes():
            if node == guru:
                node_colors.append('darkorange')
            elif __len_edges(graph_to_draw, node) == 0:
                node_colors.append('lightblue')
            elif __len_edges(graph_to_draw, node) == 1:
                node_colors.append('steelblue')
            else:
                node_colors.append('royalblue')
    if hasattr(original_graph, "type") and original_graph.type == "barabasi_pref":
        nx.draw(graph_to_draw, pos=node_positions, node_color=node_colors, with_labels=True)
    else:
        nx.draw(graph_to_draw, pos=node_positions, node_color=node_colors, arrowstyle='->',
                arrows=True, with_labels=True)

    if show:
        plt.show()
    return guru


# def plot_degree_histogram(g, normalized=True):
#     aux_y = nx.degree_histogram(g)
#     aux_x = np.arange(0, len(aux_y)).tolist()
#     n_nodes = g.number_of_nodes()
#     if normalized:
#         for i in range(len(aux_y)):
#             aux_y[i] = aux_y[i] / n_nodes
#     return aux_x, aux_y

def save_graph_png(graph, description, filename, add_info=False):
    if add_info:
        if not description:
            description = ""
        description += " "+GraphStatistics.describe(graph)

    fig = plt.figure(layout="constrained")
    gs = gridspec.GridSpec(4, 4, figure=fig)
    fig.add_subplot(gs[:, :])
    guru = draw(graph, new_guru_look_for=True, title=description)
    plt.rcParams.update({'font.size': 6})

    # ax4 = fig.add_subplot(gs[-1, 0])
    # aux_x, aux_y = plot_degree_histogram(graph, False)
    # ax4.plot(aux_x, aux_y, 'o')
    # warnings.simplefilter("ignore")
    # ax4.set_xscale("log")
    # ax4.set_yscale("log")
    # warnings.resetwarnings()
    # ax4.set_xlabel('')
    # ax4.set_ylabel('')

    ax5 = fig.add_subplot(gs[-1, 0])
    aux_y = nx.degree_histogram(graph)
    aux_y.sort(reverse=True)
    aux_x = np.arange(0, len(aux_y)).tolist()
    ax5.loglog(aux_x, aux_y, 'o')
    ax5.set_xlabel('')
    ax5.set_ylabel('')

    plt.rcParams.update(plt.rcParamsDefault)
    plt.savefig(filename)
    plt.close('all')
    return guru


def save_graph_json(graph, filename):
    if graph:
        graph_json = nx.node_link_data(graph)
        with open(filename, 'w') as f:
            json.dump(graph_json, f)


def load_graph_json(filename):
    with open(filename, 'r') as f:
        graph_json = json.load(f)
        return nx.node_link_graph(graph_json)


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


class GraphStatistics:
    @staticmethod
    def giant_component_size(graph):
        """weakly connected componentes of the directed graph using Tarjan's algorithm:
           https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm"""
        if graph.is_directed():
            return len(max(nx.weakly_connected_components(graph), key=len))
        else:
            return len(max(nx.connected_components(graph), key=len))

    @staticmethod
    def clustering_coeff(graph):
        """clustering coefficient 0..1, 1 for totally connected graphs, and 0 for totally isolated
           if ~0 then a small world"""
        try:
            return nx.average_clustering(graph, count_zeros=True)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def communities(graph):
        """number of communities using greedy modularity maximization"""
        return nx.community.greedy_modularity_communities(graph)

    @staticmethod
    def describe(graph):
        clustering = GraphStatistics.clustering_coeff(graph)
        if clustering > 0 and clustering < 1:
            clustering = f"clus_coef={clustering:5.3f}"
        else:
            clustering = f"clus_coef={clustering}"
        return (f"giant={GraphStatistics.giant_component_size(graph)} " +
                clustering +
                f" comm={len(GraphStatistics.communities(graph))}")


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
            edges = list(banks_graph.out_edges(node))
            edge_selected = random.randrange(len(edges))
            this_model.banks[node].lender = edges[edge_selected][1]
        else:
            this_model.banks[node].lender = None


# ---------------------------------------------------------
# prototype
class LenderChange:
    GRAPH_NAME = ""

    def __init__(self):
        self.parameter = {}
        self.initial_graph_file = None

    def initialize_bank_relationships(self, this_model):
        """ Call once at initilize() model """
        pass

    def step_setup_links(self, this_model):
        """ Call at the end of each step """
        pass

    def change_lender(self, this_model, bank, t):
        """ Call at the end of each step before going to the next"""
        pass

    def new_lender(self, this_model, bank):
        """ Describes the mechanism of change"""
        pass

    def set_initial_graph_file(self, lc_ini_graph_file):
        if lc_ini_graph_file:
            self.initial_graph_file = lc_ini_graph_file

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
    gamma: float = 0.5  # [0..1] gamma
    CHANGE_LENDER_IF_HIGHER = 0.5

    def initialize_bank_relationships(self, this_model):
        if self.initial_graph_file:
            graph = load_graph_json(self.initial_graph_file)
            from_graph_to_array_banks(graph, this_model)
        if this_model.export_datafile:
            this_model.statistics.get_graph(0)

    def change_lender(self, this_model, bank, t):
        """ It uses γ but only after t=20, at the beginning only Boltzmann"""
        possible_lender = self.new_lender(this_model, bank)
        if possible_lender is None:
            possible_lender_mi = 0
        else:
            possible_lender_mi = this_model.banks[possible_lender].mu
        if bank.getLender() is None:
            current_lender_mi = 0
        else:
            current_lender_mi = bank.getLender().mu

        # we can now break old links and set up new lenders, using probability P
        # (equation 8)
        boltzmann = 1 / (1 + math.exp(-this_model.config.beta * (possible_lender_mi - current_lender_mi)))

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
            bank.P = self.gamma * bank.P_yesterday + (1 - self.gamma) * boltzmann

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
            bank.mu = 0
            bank.asset_i = 0
            bank.asset_j = 0
            bank.asset_equity = 0
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
        return f"($\\gamma={self.gamma} and change if >{self.CHANGE_LENDER_IF_HIGHER})$"


class InitialStability(Boltzman):
    """ We define a Barabási–Albert graph with 1 edges for each node, and we convert it to a directed graph.
          It is used for initially have more stable links between banks
    """

    CHANGE_LENDER_IF_HIGHER = 0.8
    GRAPH_NAME = 'barabasi_albert'

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
        if self.initial_graph_file:
            self.banks_graph = load_graph_json(self.initial_graph_file)
            description = f"{self.GRAPH_NAME} from file {self.initial_graph_file}"
        else:
            self.banks_graph, _ = get_graph_from_guru(nx.barabasi_albert_graph(this_model.config.N, 1))
            description = f"{self.GRAPH_NAME} m=1 {GraphStatistics.describe(self.banks_graph)}"
        self.banks_graph.type = self.GRAPH_NAME
        if this_model.export_datafile:
            save_graph_png(self.banks_graph, description,
                           this_model.statistics.get_export_path(this_model.export_datafile, f"_{self.GRAPH_NAME}.png"))
            save_graph_json(self.banks_graph,
                            this_model.statistics.get_export_path(this_model.export_datafile,
                                                                  f"_{self.GRAPH_NAME}.json"))
        from_graph_to_array_banks(self.banks_graph, this_model)
        return self.banks_graph


class RestrictedMarket(LenderChange):
    """ Using an Erdos Renyi graph with p=parameter['p']. This method does not allow the evolution in
          lender for each bank, it replicates the situation after a crisis when banks do not credit

    """

    GRAPH_NAME = "erdos_renyi"

    def check_parameter(self, name, value):
        if name == 'p':
            if isinstance(value, float) and 0 <= value <= 1:
                return True
            else:
                print("value for 'p' should be a float number >= 0 and <= 1")
                return False
        else:
            return False

    def initialize_bank_relationships(self, this_model):
        """ It creates a Erdos Renyi graph with p defined in parameter['p']. No changes in relationships before end"""
        if self.initial_graph_file:
            self.banks_graph = load_graph_json(self.initial_graph_file)
            description = f"erdos_renyi from file {self.initial_graph_file}"
        else:
            self.banks_graph = nx.erdos_renyi_graph(this_model.config.N, self.parameter['p'], directed=True)
            description = f"erdos_renyi p={self.parameter['p']:5.3} {GraphStatistics.describe(self.banks_graph)}"
        self.banks_graph.type = self.GRAPH_NAME
        if this_model.export_datafile:
            save_graph_png(self.banks_graph, description,
                           this_model.statistics.get_export_path(this_model.export_datafile, f"_{self.GRAPH_NAME}.png"))
            save_graph_json(self.banks_graph,
                            this_model.statistics.get_export_path(this_model.export_datafile,
                                                                  f"_{self.GRAPH_NAME}.json"))

        from_graph_to_array_banks(self.banks_graph, this_model)
        if this_model.export_datafile:
            this_model.statistics.get_graph(0)
        return self.banks_graph

    def new_lender(self, this_model, bank):
        """ In this LenderChange we never change of lender """
        bank.rij = np.full(this_model.config.N, this_model.config.r_i0, dtype=float)
        bank.rij[bank.id] = 0
        bank.r = this_model.config.r_i0
        bank.mu = 0
        return bank.lender

    def change_lender(self, this_model, bank, t):
        """ In this LenderChange we never change of lender """
        bank.P = 0
        return f"{bank.getId()} maintains lender #{bank.lender} with %1 (ShockedMarket)"


class Preferential(Boltzman):
    """ Using a Barabasi with grade m we restrict to those relations the possibilities to obtain an outgoing link
          (a lender). To improve the specialization of banks, granting 3*C0 to the guru, 2*C0 to its neighbours
          and C to the others. To balance those with 2C0 and 3C0, we will reduce D
        In each step, we change the lender using the base self.banks_graph_full
    """
    banks_graph = None
    guru = None
    GRAPH_NAME = "barabasi_pref"

    def check_parameter(self, name, value):
        if name == 'm':
            if isinstance(value, int) and 1 <= value:
                return True
            else:
                print("value for 'm' should be an integer >= 1")
                return False
        else:
            return False

    def initialize_bank_relationships(self, this_model):

        if self.initial_graph_file:
            self.banks_graph_full = load_graph_json(self.initial_graph_file)
            description = f"{self.GRAPH_NAME} from file {self.initial_graph_file}"
        else:
            self.banks_graph_full = nx.barabasi_albert_graph(this_model.config.N, self.parameter['m'])
            description = (f"{self.GRAPH_NAME} m={self.parameter['m']:5.3f} "
                           f"{GraphStatistics.describe(self.banks_graph_full)}")
        self.banks_graph_full.type = self.GRAPH_NAME
        if this_model.export_datafile:
            self.guru = save_graph_png(self.banks_graph_full, description,
                                       this_model.statistics.get_export_path(this_model.export_datafile,
                                                                             f"_{self.GRAPH_NAME}.png"))
        else:
            self.guru, _ = find_guru(self.banks_graph_full)
        self.full_barabasi_extract_random_directed(this_model)
        self.prize_for_good_banks(this_model)
        if this_model.export_datafile:
            save_graph_json(self.banks_graph,
                            this_model.statistics.get_export_path(this_model.export_datafile,
                                                                  f"_{self.GRAPH_NAME}.json"))
        return self.banks_graph

    def prize_for_good_banks(self, this_model):
        this_model.banks[self.guru].D -= this_model.banks[self.guru].C * 2
        this_model.banks[self.guru].C *= 3
        for (_, node) in self.banks_graph_full.edges(self.guru):
            this_model.banks[node].D -= this_model.banks[node].C
            this_model.banks[node].C *= 2

    def step_setup_links(self, this_model):
        self.full_barabasi_extract_random_directed(this_model)


    def full_barabasi_extract_random_directed(self, this_model, current_node=None):
        if current_node is None:
            self.banks_graph = nx.DiGraph()
            self.banks_graph.type = self.GRAPH_NAME
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
        if candidate is None:
            if this_model.banks[current_node].lender is not None:
                self.banks_graph.add_edge(current_node, this_model.banks[current_node].lender)
        else:
            self.banks_graph.add_edge(current_node, candidate)
        return current_node

    def new_lender(self, this_model, bank):
        if bank.lender is None:
            bank.rij = np.full(this_model.config.N, this_model.config.r_i0, dtype=float)
            bank.rij[bank.id] = 0
            bank.r = this_model.config.r_i0
            bank.mu = 0
        if self.banks_graph and self.banks_graph.out_edges() and self.banks_graph.out_edges(bank.id):
            return list(self.banks_graph.out_edges(bank.id))[0][1]
        else:
            return None

    def describe(self):
        return f"($\\gamma={self.gamma} and change if >{self.CHANGE_LENDER_IF_HIGHER})$"


class ShockedMarket(LenderChange):
    """ Using an Erdos Renyi graph with p=parameter['p']. This method replicate RestrictedMarket
        but using a new network relationship between banks, but always with same p. So the links
        in t=i are destroyed and new aleatory links in t=i+1 are created using a new Erdos Renyi
        graph.
    """

    GRAPH_NAME = "erdos_renyi"

    def check_parameter(self, name, value):
        if name == 'p':
            if isinstance(value, float) and 0 <= value <= 1:
                return True
            else:
                print(f"value for 'p' should be a float number >= 0 and <= 1: {value}")
                return False
        else:
            return False

    def step_setup_links(self, this_model):
        self.banks_graph = nx.erdos_renyi_graph(this_model.config.N, self.parameter['p'], directed=True)
        self.banks_graph.type = self.GRAPH_NAME
        from_graph_to_array_banks(self.banks_graph, this_model)

    def initialize_bank_relationships(self, this_model):
        """ It creates a Erdos Renyi graph with p defined in parameter['p']. No changes in relationships till the end"""
        if self.initial_graph_file:
            self.banks_graph = load_graph_json(self.initial_graph_file)
            description = f"{self.GRAPH_NAME} from file {self.initial_graph_file}"
        else:
            self.banks_graph = nx.erdos_renyi_graph(this_model.config.N, self.parameter['p'], directed=True)
            description = f"{self.GRAPH_NAME} p={self.parameter['p']:5.3} {GraphStatistics.describe(self.banks_graph)}"
        self.banks_graph.type = self.GRAPH_NAME
        if this_model.export_datafile:
            save_graph_png(self.banks_graph, description,
                           this_model.statistics.get_export_path(this_model.export_datafile, f"_{self.GRAPH_NAME}.png"))
            save_graph_json(self.banks_graph,
                            this_model.statistics.get_export_path(this_model.export_datafile,
                                                                  f"_{self.GRAPH_NAME}.json"))

        from_graph_to_array_banks(self.banks_graph, this_model)
        if this_model.export_datafile:
            this_model.statistics.get_graph(0)
        return self.banks_graph

    def new_lender(self, this_model, bank):
        """ In this LenderChange we never change of lender """
        bank.rij = np.full(this_model.config.N, this_model.config.r_i0, dtype=float)
        bank.rij[bank.id] = 0
        bank.r = this_model.config.r_i0
        bank.asset_i = 0
        bank.asset_j = 0
        bank.asset_equity = 0
        bank.mu = 0
        return bank.lender

    def change_lender(self, this_model, bank, t):
        """ In this LenderChange we never change of lender """
        bank.P = 0
        return f"{bank.getId()} maintains lender #{bank.lender} with %1 (ShockedMarket)"


class SmallWorld(ShockedMarket):
    """ SmallWorld implementation using Watts Strogatz
    """

    GRAPH_NAME = 'small_world'
    # Each node is joined with its k nearest neighbors in a ring topology (Watts Strogatz parameter):
    K = 5

    @staticmethod
    def create_directed_graph_from_watts_strogatz(graph, result=None, pending=None, current_node=None):
        # a) first time: look for extremes of the graph and use current_node to create links from them:
        if result is None:
            result = nx.DiGraph()
            pending = list(graph.nodes())
            # in each node with only one link, we start from it:
            for node in graph.nodes():
                edges = list(graph.edges(node))
                if len(edges) == 1:
                    SmallWorld.create_directed_graph_from_watts_strogatz(graph, result, pending, node)
            # no nodes with only one option: we choose an arbitrary node to start:
            if result.edges() == 0:
                random_node = random.choice(graph.nodes())
                SmallWorld.create_directed_graph_from_watts_strogatz(graph, result, pending, random_node)
            SmallWorld.create_directed_graph_from_watts_strogatz(graph, result, pending)
        elif current_node:
            # b) nodes with an incoming link that we should follow (there's only that option)
            source = current_node
            destination = list(graph.edges(source))[0][1]
            while not destination is None:
                if destination not in result or len(result.in_edges(destination)) == 0:
                    result.add_edge(source, destination)
                    pending.remove(source)
                    edges = list(graph.edges(destination))
                    edges.remove((destination, source))
                    source = destination
                    if len(edges) == 1:
                        destination = edges[0][1]
                    else:
                        destination = None
                else:
                    destination = None
        elif pending:
            # c) items with >1 node
            for source in pending:
                edges = list(graph.edges(source))
                if result.has_node(source):
                    for in_edges in result.in_edges(source):
                        if (in_edges[1], in_edges[0]) in edges:
                            edges.remove((in_edges[1], in_edges[0]))
                if edges:
                    new_link = random.choice(edges)
                    result.add_edge(new_link[0], new_link[1])
                else:
                    result.add_node(source)
        return result

    def check_parameter(self, name, value):
        if name == 'p':
            if isinstance(value, float) and 0 <= value <= 1:
                return True
            else:
                print("value for 'p' should be a float number >= 0 and <= 1")
                return False
        else:
            return False

    def step_setup_links(self, this_model):
        self.banks_graph = SmallWorld.create_directed_graph_from_watts_strogatz(self.banks_smallworld)
        self.banks_graph.type = self.GRAPH_NAME
        from_graph_to_array_banks(self.banks_graph, this_model)

    def initialize_bank_relationships(self, this_model):
        """ It creates a small world graph using Watts Strogatz. It's indirected and we directed it using an own
                algorithm which guarantees 1 output link only for each node (only a lender for each bank)
            """
        if self.initial_graph_file:
            self.banks_smallworld = load_graph_json(self.initial_graph_file)
            self.description = f"{self.GRAPH_NAME} from file {self.initial_graph_file}"
        else:
            self.banks_smallworld = nx.watts_strogatz_graph(this_model.config.N, self.K, self.parameter['p'])
            self.description = f"{self.GRAPH_NAME} p={self.parameter['p']:5.3}"
        self.banks_graph = SmallWorld.create_directed_graph_from_watts_strogatz(self.banks_smallworld)
        self.description += f"{GraphStatistics.describe(self.banks_graph)}"
        self.banks_graph.type = self.GRAPH_NAME
        if this_model.export_datafile:
            save_graph_png(self.banks_graph, self.description,
                           this_model.statistics.get_export_path(this_model.export_datafile, f"_{self.GRAPH_NAME}.png"))
            save_graph_json(self.banks_graph,
                            this_model.statistics.get_export_path(this_model.export_datafile,
                                                                  f"_{self.GRAPH_NAME}.json"))
        from_graph_to_array_banks(self.banks_graph, this_model)
        return self.banks_graph
