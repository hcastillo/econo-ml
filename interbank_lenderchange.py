# -*- coding: utf-8 -*-
"""
LenderChange is a class used from interbank.py to control the change of lender in a bank.
   It contains different
    - Boltzmann          using Boltzman probability to change
    - InitialStability   using a Barabási–Albert graph to initially assign relationships between banks
    - Preferential       using a Barabási-Albert with m edges for each new nde to set up only a set o links between
                            banks. In each step, new links between are set up based on the initial graph
    - RestrictedMarket   using an Erdos Renyi graph with p=parameter['p']. This method does not allow the evolution in
                            lender for each bank, it replicates the situation after a crisis when banks do not credit
    - ShockedMarket      using an Erdos Renyi graph with p=parameter['p']. In each step a new random Erdos Renyi is
                            used
    - ShockedMarket2     Erdos Renyi graph not directed and in each step, we randomly choose a direction

    - SmallWorld         using a Watts and Strogatz algorithm with parameter['p'] and k=5
                            (Each node is joined with its k nearest neighbors in a ring topology)
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
import json
import warnings


def determine_algorithm(given_name: str = "default"):
    default_method = "Boltzmann"

    if given_name == "default":
        print(f"selected default method {default_method}")
        given_name = default_method
    if given_name == '?':
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj) and obj.__doc__:
                print("\t" + obj.__name__ + (" (default)" if name == default_method else '') + ':\n\t', obj.__doc__)
        sys.exit(0)
    else:
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if name.lower() == given_name.lower():
                if inspect.isclass(obj) and obj.__doc__:
                    return obj()
        print(f"not found LenderChange algorithm with name '{given_name}'")
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
    if hasattr(original_graph, "type") and original_graph.type == "erdos_renyi" and graph_to_draw.is_directed():
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
        description += " " + GraphStatistics.describe(graph)

    #fig = plt.figure(layout="constrained")
    #gs = gridspec.GridSpec(4, 4, figure=fig)
    #fig.add_subplot(gs[:, :])
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

    # ax5 = fig.add_subplot(gs[-1, 0])
    # aux_y = nx.degree_histogram(graph)
    # aux_y.sort(reverse=True)
    # aux_x = np.arange(0, len(aux_y)).tolist()
    # ax5.loglog(aux_x, aux_y, 'o')
    # ax5.set_xlabel('')
    # ax5.set_ylabel('')

    plt.rcParams.update(plt.rcParamsDefault)
    warnings.filterwarnings("ignore", category=UserWarning)
    plt.savefig(filename)
    plt.close('all')
    return guru


def save_graph_json(graph, filename):
    if graph:
        graph_json = nx.node_link_data(graph, edges="links")
        with open(filename, 'w') as f:
            json.dump(graph_json, f)


def load_graph_json(filename):
    with open(filename, 'r') as f:
        graph_json = json.load(f)
        return nx.node_link_graph(graph_json, edges="links")


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


class WattsStrogatzGraph:
    def ring_lattice_edges(self, nodes, k):
        half = k // 2
        n = len(nodes)
        for i, v in enumerate(nodes):
            for j in range(i + 1, i + half + 1):
                w = nodes[j % n]
                yield v, w

    def make_ring_lattice(self, n, k):
        G = nx.DiGraph()
        nodes = range(n)
        G.add_nodes_from(nodes)
        edges = self.ring_lattice_edges(nodes, k)
        G.add_edges_from(edges)
        return G

    def rewire(self, G, p):
        for v, w in G.copy().edges():
            if random.random() < p:
                G.remove_edge(v, w)
                choices = set(G) - {v} - set(G[v])
                new_w = random.choice(list(choices))
                G.add_edge(v, new_w)

    def new(self, n, p):
        G = self.make_ring_lattice(n=n, k=2)
        self.rewire(G, p)
        return G


class GraphStatistics:
    @staticmethod
    def giant_component_size(graph):
        """weakly connected componentes of the directed graph using Tarjan's algorithm:
           https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm"""
        if graph.is_directed():
            #return len(max(nx.weakly_connected_components(graph), key=len))
            # return len(max(nx.strongly_connected_components(graph), key=len))
            #TODO
            return nx.average_clustering(graph) #len(max(nx.strongly_connected_components(graph), key=len))
        else:
            return len(max(nx.connected_components(graph), key=len))

    @staticmethod
    def get_all_credit_channels(graph):
        return graph.number_of_edges()

    @staticmethod
    def avg_clustering_coef(graph):
        """clustering coefficient 0..1, 1 for totally connected graphs, and 0 for totally isolated
           if ~0 then a small world"""
        try:
            return nx.average_clustering(graph, count_zeros=True)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def communities(graph):
        """Communities using greedy modularity maximization"""
        return list(nx.weakly_connected_components(graph if graph.is_directed() else graph.to_directed()))
        #return list(nx.strongly_connected_components(graph if graph.is_directed() else graph.to_directed()))
    @staticmethod
    def grade_avg(graph):
        communities = GraphStatistics.communities(graph)
        total = 0
        for community in communities:
            total += len(community)
        return total / len(communities)

    @staticmethod
    def communities_not_alone(graph):
        """Number of communities that are not formed only with an isolated node"""
        total = 0
        for community in GraphStatistics.communities(graph):
            total += len(community) > 1
        return total

    @staticmethod
    def describe(graph, interact=False):
        if isinstance(graph, str):
            graph_name = graph
            try:
                graph = load_graph_json(graph_name)
            except FileNotFoundError:
                print("json file does not exist: %s" % graph)
                sys.exit(0)
            except (UnicodeDecodeError, json.decoder.JSONDecodeError) as e:
                print("json file does not contain a valid graph: %s" % graph)
                sys.exit(0)
        else:
            graph_name = '?'
        communities = GraphStatistics.communities(graph)
        string_result = f"giant={GraphStatistics.giant_component_size(graph)} " + \
                        f" comm_not_alone={GraphStatistics.communities_not_alone(graph)}" + \
                        f" comm={len(communities)}" + \
                        f" gcs={GraphStatistics.giant_component_size(graph)}"
        if interact:
            import code
            print(string_result)
            print(f"grade_avg={GraphStatistics.grade_avg(graph)}")
            print("communities=", list(GraphStatistics.communities(graph)))
            print(f"\n{graph_name} loaded into 'graph'\n")
            code.interact(local=locals())
            sys.exit(0)
        else:
            return string_result


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
    if not banks_graph.is_directed():
        banks_graph = banks_graph.to_directed()
    for node in banks_graph:
        this_model.banks[node].lender = get_new_out_edge(banks_graph, node)


def get_new_out_edge(banks_graph, node):
    if banks_graph.out_edges(node):
        edges = list(banks_graph.out_edges(node))
        edge_selected = random.randrange(len(edges))
        return edges[edge_selected][1]
    else:
        return None


# ---------------------------------------------------------
# prototype
class LenderChange:
    GRAPH_NAME = ""

    def __init__(self):
        self.parameter = {}
        self.initial_graph_file = None

    def __str__(self):
        return "LenderChangePrototype"

    def initialize_bank_relationships(self, this_model):
        """ Call once at initialize() model """
        pass

    def extra_relationships_change(self, this_model):
        pass

    def step_setup_links(self, this_model):
        """ Call at the end of each step """
        # if we remove banks from the array of banks, and there is no step_setup_links() code itself defined
        # in the LenderChange() class, we should at least call initialize_bank_relationships() to be sure
        # there are no links to removed banks:
        if not this_model.config.allow_replacement_of_bankrupted:
            self.initialize_bank_relationships(this_model)

    def change_lender(self, this_model, bank, t):
        """ Call at the end of each step, for each bank and before going to the next step"""
        pass

    def new_lender(self, this_model, bank):
        """ Describes the mechanism of change"""
        pass

    def set_initial_graph_file(self, lc_ini_graph_file):
        self.initial_graph_file = lc_ini_graph_file

    def describe(self):
        return ""

    def check_parameter(self, name, value):
        """ Called after set_parameter() to verify that the necessary parameters are set """
        return False

    def set_parameter(self, name, value):
        if not value is None:
            if self.check_parameter(name, value):
                self.parameter[name] = float(value)
            else:
                print(f"error with parameter '{name}' for {self.__class__.__name__}")
                sys.exit(-1)

    def get_credit_channels(self):
        if hasattr(self, "banks_graph"):
            return GraphStatistics.get_all_credit_channels(self.banks_graph)
        else:
            return None


# ---------------------------------------------------------


class Boltzmann(LenderChange):
    """It chooses randomly a lender for each bank and in each step changes using Boltzmann's probability
         Also it has parameter γ to control the change [0..1], 0=Boltzmann only, 1 everyone will move randomly
    """
    # parameter to control the change of guru: 0 then Boltzmann only, 1 everyone will move randomly
    gamma: float = 0.5  # [0..1] gamma
    CHANGE_LENDER_IF_HIGHER = 0.5

    def __str__(self):
        return "BoltzMann"

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
            possible_lender_mu = 0
        else:
            possible_lender_mu = this_model.banks[possible_lender].mu
        if bank.get_lender() is None:
            current_lender_mu = 0
        else:
            current_lender_mu = bank.get_lender().mu

        # we can now break old links and set up new lenders, using probability P
        # (equation 8)
        boltzmann = 1 / (1 + math.exp(-this_model.config.beta * (possible_lender_mu - current_lender_mu)))

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
            text_to_return = f"{bank.get_id()} new lender is #{possible_lender} from #{bank.lender} with %{bank.P:.3f}"
            bank.lender = possible_lender
        else:
            text_to_return = f"{bank.get_id()} maintains lender #{bank.lender} with %{1 - bank.P:.3f}"
        return text_to_return

    def new_lender(self, this_model, bank):
        """ It gives to the bank a random new lender. It's used initially and from change_lender() """
        # r_i0 is used the first time the bank is created:
        if bank.lender is None:
            bank.rij = np.full(this_model.config.N, this_model.config.r_i0, dtype=float)
            bank.rij[bank.id] = 0
            #bank.r = this_model.config.r_i0
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


class InitialStability(Boltzmann):
    """ We define a Barabási–Albert graph with 1 edges for each node, and we convert it to a directed graph.
          It is used for initially have more stable links between banks
    """

    CHANGE_LENDER_IF_HIGHER = 0.8
    GRAPH_NAME = 'barabasi_albert'

    def __str__(self):
        return "InitialStability"

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


class Preferential(Boltzmann):
    """ Using a Barabasi with grade m we restrict to those relations the possibilities to obtain an outgoing link
          (a lender). To improve the specialization of banks, granting 3*C0 to the guru, 2*C0 to its neighbours
          and C to the others. To balance those with 2C0 and 3C0, we will reduce D
        In each step, we change the lender using the base self.banks_graph_full
    """
    banks_graph = None
    guru = None
    GRAPH_NAME = "barabasi_pref"

    def __str__(self):
        if 'm' in self.parameter:
            return f"Preferential.m={self.parameter['m']}"
        else:
            return f"Preferential"

    def check_parameter(self, name, value):
        if name == 'm':
            if isinstance(value, int) and 1 <= value:
                return True
            else:
                print("value for 'm' should be an integer >= 1: %s" % value)
                return False
        else:
            return False

    def set_parameter(self, name, value):
        if not value is None:
            if self.check_parameter(name, value):
                self.parameter[name] = int(value)
            else:
                print(f"error with parameter '{name}' for {self.__class__.__name__}")
                sys.exit(-1)

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
            #bank.r = this_model.config.r_i0
            bank.mu = 0
        if self.banks_graph and self.banks_graph.out_edges() and self.banks_graph.out_edges(bank.id):
            return list(self.banks_graph.out_edges(bank.id))[0][1]
        else:
            return None

    def describe(self):
        return f"($\\gamma={self.gamma} and change if >{self.CHANGE_LENDER_IF_HIGHER})$"


class RestrictedMarket(LenderChange):
    """ Using an Erdos Renyi graph with p=parameter['p']. This method does not allow the evolution in
          lender for each bank, it replicates the situation after a crisis when banks do not credit
    """
    SAVE_THE_DIFFERENT_GRAPH_OF_EACH_STEP = True
    GRAPH_NAME = "erdos_renyi"

    def __str__(self):
        if 'p' in self.parameter:
            return f"Restricted.p={self.parameter['p']}"
        else:
            return f"Restricted"

    def generate_banks_graph(self, this_model):
        result = nx.DiGraph()
        result.add_nodes_from(list(range(this_model.config.N)))
        for i in range(this_model.config.N):
            if random.random() < self.parameter['p']:
                j = random.randrange(this_model.config.N)
                while j == i:
                    j = random.randrange(this_model.config.N)
                result.add_edge(i, j)
        return result, f"erdos_renyi p={self.parameter['p']:5.3} {GraphStatistics.describe(result)}"

    def check_parameter(self, name, value):
        if name == 'p':
            if isinstance(value, int):
                value = float(value)
            if isinstance(value, float) and 0 <= value <= 1:
                return True
            else:
                print("value for 'p' should be a float number >= 0 and <= 1: %s" % value)
                return False
        else:
            return False

    def initialize_bank_relationships(self, this_model, save_graph=True):
        """ It creates a Erdos Renyi graph with p defined in parameter['p']. No changes in relationships before end"""
        if self.initial_graph_file:
            self.banks_graph = load_graph_json(self.initial_graph_file)
            description = f"from file {self.initial_graph_file}"
        else:
            self.banks_graph, description = self.generate_banks_graph(this_model)
        self.banks_graph.type = self.GRAPH_NAME
        if this_model.export_datafile and save_graph:
            filename_for_file = f"_{self.GRAPH_NAME}"
            if self.SAVE_THE_DIFFERENT_GRAPH_OF_EACH_STEP:
                filename_for_file += f"_{this_model.t}"
            save_graph_png(self.banks_graph, description,
                         this_model.statistics.get_export_path(this_model.export_datafile, f"{filename_for_file}.png"))
            save_graph_json(self.banks_graph,
                         this_model.statistics.get_export_path(this_model.export_datafile, f"{filename_for_file}.json"))
        for (borrower, lender_for_borrower) in self.banks_graph.edges():
            this_model.banks[borrower].lender = lender_for_borrower
        return self.banks_graph

    def step_setup_links(self, this_model):
        # if not declared, at each step it will generate a new graph:
        pass

    def new_lender(self, this_model, bank):
        """ We return the same lender we have created in self.banks_graph """
        bank.rij = np.full(this_model.config.N, this_model.config.r_i0, dtype=float)
        bank.rij[bank.id] = 0
        #bank.r = this_model.config.r_i0
        bank.mu = 0
        bank.asset_i = 0
        bank.asset_j = 0
        bank.asset_equity = 0
        return bank.lender

    def change_lender(self, this_model, bank, t):
        """ We return the same lender we have created in self.banks_graph """
        bank.P = 0
        return f"{bank.get_id()} maintains lender #{bank.lender} with %1"


class ShockedMarket(RestrictedMarket):
    """ Using an Erdos Renyi graph with p=parameter['p']. This method replicate RestrictedMarket
        but using a new network relationship between banks, but always with same p. So the links
        in t=i are destroyed and new aleatory links in t=i+1 are created using a new Erdos Renyi
        graph.
    """
    SAVE_THE_DIFFERENT_GRAPH_OF_EACH_STEP = False

    def __str__(self):
        if 'p' in self.parameter:
            return f"ShockedMarket.p={self.parameter['p']}"
        else:
            return f"ShockedMarket"


    def step_setup_links(self, this_model):
        """ At the end of each step, a new graph is generated """
        self.initialize_bank_relationships(this_model, save_graph=self.SAVE_THE_DIFFERENT_GRAPH_OF_EACH_STEP)


class ShockedMarket2(ShockedMarket):
    """ Erdos Renyi graph not directed and in each step, we randomly choose a direction.
    """
    SAVE_THE_DIFFERENT_GRAPH_OF_EACH_STEP = False

    def __str__(self):
        if 'p' in self.parameter:
            return f"ShockedMarket2.p={self.parameter['p']}"
        else:
            return f"ShockedMarket2"

    def extra_relationships_change(self, this_model):
        for bank in this_model.banks:
            if bank.incrD < 0: # borrowers
                # we search all the possible lenders and we randomly choose one with incrD>0:
                possible_lenders = []
                for (_,j) in self.banks_graph.edges(bank.id):
                    if this_model.banks[j].incrD> 0:
                        possible_lenders.append(j)
                if possible_lenders:
                    bank.lender = random.choice(possible_lenders)
                else:
                    bank.lender = None


class ShockedMarket3(ShockedMarket2):
    """ Erdos Renyi graph not directed and in each step with no limits in number of edges,
        and we randomly choose a direction.
    """
    SAVE_THE_DIFFERENT_GRAPH_OF_EACH_STEP = False

    def __str__(self):
        if 'p' in self.parameter:
            return f"ShockedMarket3.p={self.parameter['p']}"
        else:
            return f"ShockedMarket3"

    def generate_banks_graph(self, this_model):
        result = nx.erdos_renyi_graph(n=this_model.config.N, p=self.parameter['p'])
        return result, f"erdos_renyi p={self.parameter['p']:5.3} {GraphStatistics.describe(result)}"

    def extra_relationships_change(self, this_model):
        for bank in this_model.banks:
            if bank.incrD < 0: # borrowers
                # we search all the possible lenders and we randomly choose one with incrD>0:
                possible_lenders = []
                for (_,j) in self.banks_graph.edges(bank.id):
                    if this_model.banks[j].incrD> 0:
                        possible_lenders.append(j)
                if possible_lenders:
                    bank.lender = random.choice(possible_lenders)
                else:
                    bank.lender = None


class SmallWorld(ShockedMarket):
    """ SmallWorld implementation using Watts Strogatz
    """
    GRAPH_NAME = "watts_strogatz"

    def __str__(self):
        if 'p' in self.parameter:
            return f"SmallWorld.p={self.parameter['p']}"
        else:
            return f"SmallWorld"


    def generate_banks_graph(self, this_model):
        generator = WattsStrogatzGraph()
        result = generator.new(n=this_model.config.N, p=self.parameter['p'])
        return result, f"watts_strogatz p={self.parameter['p']:5.3} {GraphStatistics.describe(result)}"

    def initialize_bank_relationships(self, this_model, save_graph=True):
        """ It creates a Erdos Renyi graph with p defined in parameter['p']. No changes in relationships before end"""
        if self.initial_graph_file:
            self.banks_graph = load_graph_json(self.initial_graph_file)
            description = f"from file {self.initial_graph_file}"
        else:
            self.banks_graph, description = self.generate_banks_graph(this_model)
        self.banks_graph.type = self.GRAPH_NAME
        if this_model.export_datafile and save_graph:
            save_graph_png(self.banks_graph, description,
                           this_model.statistics.get_export_path(this_model.export_datafile, f"_{self.GRAPH_NAME}.png"))
            save_graph_json(self.banks_graph,
                            this_model.statistics.get_export_path(this_model.export_datafile,
                                                                  f"_{self.GRAPH_NAME}.json"))
        for (borrower, lender_for_borrower) in self.banks_graph.edges():
            this_model.banks[borrower].lender = lender_for_borrower
        return self.banks_graph

    def new_lender(self, this_model, bank):
        """ Same lender we have created in self.banks_graph """
        bank.rij = np.full(this_model.config.N, this_model.config.r_i0, dtype=float)
        bank.rij[bank.id] = 0
        #bank.r = this_model.config.r_i0
        bank.mu = 0
        bank.asset_i = 0
        bank.asset_j = 0
        bank.asset_equity = 0
        return bank.lender

    def change_lender(self, this_model, bank, t):
        """ Lender is not changing in this model """
        bank.P = 0
        return f"{bank.get_id()} maintains lender #{bank.lender} with %1"
