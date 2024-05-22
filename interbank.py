# -*- coding: utf-8 -*-
"""
Generates a simulation of an interbank network following the rules described in paper
  Reinforcement Learning Policy Recommendation for Interbank Network Stability
  from Gabrielle and Alessio

@author: hector@bith.net
@date:   04/2023
"""
import copy
import enum
import random
import logging
import math
import argparse
import numpy as np
import networkx as nx
import sys
import os
from PIL import Image
#import cv2
import matplotlib.pyplot as plt
import interbank_lenderchange as lc

class Config:
    """
    Configuration parameters for the interbank network
    """
    T: int = 1000  # time (1000)
    N: int = 50    # number of banks (50)

    # not used in this implementation:
    # ȓ: float  = 0.02     # percentage reserves (at the moment, no R is used)
    # đ: int    = 1        # number of outgoing links allowed

    # shocks parameters:
    µ: float = 0.7  # mi
    ω: float = 0.6  # omega

    lenderchange : lc.LenderChange = lc.Boltzman()

    # screening costs
    Φ: float = 0.025  # phi
    Χ: float = 0.015  # ji

    # liquidation cost of collateral
    ξ: float = 0.3  # xi
    ρ: float = 0.3  # ro fire sale cost

    β: float = 5    # intensity of breaking the connection (5)
    α: float = 0.1  # below this level of E or D, we will bankrupt the bank

    # banks initial parameters
    # L + C + (R) = D + E
    L_i0: float = 120   # long term assets
    C_i0: float = 30    # capital
    D_i0: float = 135   # deposits
    E_i0: float = 15    # equity
    r_i0: float = 0.02  # initial rate

    # if enabled and != [] the values of t in the array (for instance [150,350]) will generate
    # a graph with the relations of the firms. If * all the instants will generate a graph, and also an animated gif
    # with the results
    GRAPHS_MOMENTS = []

    def __str__(self):
        description = sys.argv[0]
        for attr in dir(self):
            value = getattr(self, attr)
            if isinstance(value, int) or isinstance(value, float):
                description += f" {attr}={value}"
        return description


class DataColumns(enum.IntEnum):
    POLICY = 0
    FITNESS = 1
    LIQUIDITY = 2
    IR = 3
    BANKRUPTCY = 4
    BEST_LENDER = 5
    BEST_NUM_CLIENTS = 6
    CREDIT_CHANNELS = 7
    RATIONING = 8
    LEVERAGE = 9
    PROB_CHANGE_LENDER = 10
    BAD_DEBT = 11

    def get_name(i):
        for j in DataColumns:
            if j.value == i:
                return j.name.replace("_", " ").lower()
        return None
    

class Statistics:
    bankruptcy = []
    best_lender = []
    best_lender_clients = []
    credit_channels = []
    liquidity = []
    policy = []
    interest_rate = []
    incrementD = []
    fitness = []
    rationing = []
    leverage = []
    P = []
    B = []
    model = None
    graphs = {}

    OUTPUT_DIRECTORY = "output"

    def __init__(self, model):
        self.model = model
        if not os.path.isdir(self.OUTPUT_DIRECTORY):
            os.mkdir(self.OUTPUT_DIRECTORY)

    def reset(self):
        self.bankruptcy = np.zeros(self.model.config.T, dtype=int)
        self.best_lender = np.full(self.model.config.T, -1, dtype=int)
        self.best_lender_clients = np.zeros(self.model.config.T, dtype=int)
        self.credit_channels = np.zeros(self.model.config.T, dtype=int)
        self.fitness = np.zeros(self.model.config.T, dtype=float)
        self.interest_rate = np.zeros(self.model.config.T, dtype=float)
        self.incrementD = np.zeros(self.model.config.T, dtype=float)
        self.liquidity = np.zeros(self.model.config.T, dtype=float)
        self.rationing = np.zeros(self.model.config.T, dtype=float)
        self.leverage = np.zeros(self.model.config.T, dtype=float)
        self.policy = np.zeros(self.model.config.T, dtype=float)
        self.P = np.zeros(self.model.config.T, dtype=float)
        self.P_max = np.zeros(self.model.config.T, dtype=float)
        self.P_min = np.zeros(self.model.config.T, dtype=float)
        self.P_std = np.zeros(self.model.config.T, dtype=float)
        self.B = np.zeros(self.model.config.T, dtype=float)

    def compute_credit_channels_and_best_lender(self):
        lenders = {}
        for bank in self.model.banks:
            if bank.lender in lenders:
                lenders[bank.lender] += 1
            else:
                lenders[bank.lender] = 1
        best = -1
        best_value = -1
        for lender in lenders.keys():
            if lenders[lender] > best_value:
                best = lender
                best_value = lenders[lender]

        self.best_lender[self.model.t] = best
        self.best_lender_clients[self.model.t] = best_value
        # self.credit_channels is updated directly in the moment the credit channel is set up during Model.do_loans()

    def compute_interest(self):
        self.interest_rate[self.model.t] = sum(map(lambda x: x.getLoanInterest(), self.model.banks)) / \
                                           self.model.config.N

    def compute_liquidity(self):
        self.liquidity[self.model.t] = sum(map(lambda x: x.C, self.model.banks))

    def compute_fitness(self):
        self.fitness[self.model.t] = sum(map(lambda x: x.μ, self.model.banks)) / self.model.config.N

    def compute_policy(self):
        self.policy[self.model.t] = self.model.ŋ

    def compute_bad_debt(self):
        self.B[self.model.t] = sum(map(lambda x: x.B, self.model.banks))

    def compute_rationing(self):
        self.rationing[self.model.t] = sum(map(lambda x: x.rationing, self.model.banks))

    def compute_leverage(self):
        self.leverage[self.model.t] = sum(map(lambda x: (x.E / x.L), self.model.banks)) / self.model.config.N

    def compute_probability_of_lender_change(self):
        probabilities = [bank.P for bank in self.model.banks]
        self.P[self.model.t] = sum(probabilities) / self.model.config.N
        self.P_max[self.model.t] = max(probabilities)
        self.P_min[self.model.t] = min(probabilities)
        self.P_std[self.model.t] = np.std(probabilities)


    def export_data(self, export_datafile=None, export_description=None):
        if export_datafile:
            self.save_data(export_datafile, export_description)
            self.plot_credit_channels(export_datafile)
            self.plot_bankruptcies(export_datafile)
            self.plot_liquidity(export_datafile)
            self.plot_best_lender(export_datafile)
            self.plot_interest_rate(export_datafile)
            self.plot_P(export_datafile)
            self.plot_B(export_datafile)
        if Utils.is_notebook() or Utils.is_spyder():
            self.plot_credit_channels()
            self.plot_bankruptcies()
            self.plot_liquidity()
            self.plot_best_lender()
            self.plot_interest_rate()
            self.plot_P()
            self.plot_B()

    def get_graph(self, t):
        """
        Extracts from the model the graph that corresponds to the network in this instant
        """
        self.graphs[t] = nx.DiGraph(directed=True)
        for bank in self.model.banks:
            self.graphs[t].add_edge(bank.id, bank.lender)
        plt.clf()
        plt.title(f"t={t}")
        pos = nx.spring_layout(self.graphs[t])

        # pos = nx.spiral_layout(graph)
        nx.draw(self.graphs[t], pos, with_labels=True, arrowstyle='->')
        filename = sys.argv[0] if self.model.export_datafile is None else self.model.export_datafile

        if Utils.is_spyder():
            plt.show()
            return None
        else:
            filename = Statistics.get_export_path(filename).replace('.txt', f"_{t}.png")
            plt.savefig(filename)
            return filename


    def create_animation_graph(self,list_of_files):
        filename_output = sys.argv[0] if self.model.export_datafile is None else self.model.export_datafile
        filename_output = Statistics.get_export_path(filename_output).replace('.txt','.gif')
        images=[]
        for image_file in list_of_files:
            images.append(Image.open(image_file))
        images[0].save(fp=filename_output, format='GIF', append_images=images[1:],
             save_all=True, duration=100, loop=0)
        # filename_output = sys.argv[0] if self.model.export_datafile is None else self.model.export_datafile
        # filename_output = Statistics.get_export_path(filename_output).replace('.txt', '.avi')
        # frame0 = cv2.imread(list_of_files[0])
        # height, width, layers = frame0.shape
        # video = cv2.VideoWriter(filename_output, 0, 1, (width, height))
        # video.write(frame0)
        # for image in list_of_files[1:]:
        #     video.write(cv2.imread(image))
        # cv2.destroyAllWindows()
        # video.release()

    @staticmethod
    def get_export_path(filename):
        #if not filename.startswith(Statistics.OUTPUT_DIRECTORY):
        if not os.path.dirname(filename):
            filename = f"{Statistics.OUTPUT_DIRECTORY}/{filename}"
        return filename if filename.endswith('.txt') else f"{filename}.txt"

    def save_data(self, export_datafile=None, export_description=None):
        if export_datafile:
            with open(Statistics.get_export_path(export_datafile), 'w', encoding="utf-8") as savefile:
                savefile.write('  t\tpolicy\tfitness           \tliquidity            \tir         \t' +
                               'bankrupts\tbestLenderID\tbestLenderClients\tcreditChannels\trationing\tleverage\tprob_change_lender\tbad_debt\n')
                if export_description:
                    savefile.write(f"# {export_description}\n")
                else:

                    savefile.write(f"# {__name__} T={self.model.config.T} N={self.model.config.N}\n")
                for i in range(self.model.config.T):
                    savefile.write(f"{i:3}\t{self.policy[i]:3}\t{self.fitness[i]:19}\t{self.liquidity[i]:19}" +
                                   f"\t{self.interest_rate[i]:20}\t{self.bankruptcy[i]:3}" +
                                   f"\t{self.best_lender[i]:20}" +
                                   f"\t{self.best_lender_clients[i]:20}" +
                                   f"\t{self.credit_channels[i]:3}" +
                                   f"\t{self.rationing[i]:20}" +
                                   f"\t{self.leverage[i]:20}" +
                                   f"\t{self.P[i]:20}" +
                                   f"\t{self.B[i]:20}" +
                                   "\n")

    def get_data(self):
        return (
            np.array(self.policy),
            np.array(self.fitness),
            np.array(self.liquidity),
            np.array(self.interest_rate),
            np.array(self.bankruptcy),
            np.array(self.best_lender),
            np.array(self.best_lender_clients),
            np.array(self.credit_channels),
            np.array(self.rationing),
            np.array(self.leverage),
            np.array(self.P),
            np.array(self.B))

    def plot_bankruptcies(self, export_datafile=None):
        xx = []
        yy = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.bankruptcy[i])
        plt.clf()
        plt.plot(xx, yy, '-', color="blue")
        plt.xlabel("Time")
        plt.title("Bankruptcies")
        plt.ylabel("num of bankruptcies")
        if export_datafile:
            plt.savefig(Statistics.get_export_path(export_datafile).replace(".txt", "_bankruptcies.svg"))
        else:
            plt.show()

    def plot_interest_rate(self, export_datafile=None):
        xx = []
        yy = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.interest_rate[i])
        plt.clf()
        plt.plot(xx, yy, '-', color="blue")
        plt.xlabel("Time")
        plt.title("Interest")
        plt.ylabel("interest")
        if export_datafile:
            plt.savefig(Statistics.get_export_path(export_datafile).replace(".txt", "_interest_rate.svg"))
        else:
            plt.show()

    def plot_B(self, export_datafile=None):
        xx = []
        yy = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.B[i])
        plt.clf()
        plt.plot(xx, yy, '-', color="blue")
        plt.xlabel("Time")
        plt.title("Bad debt")
        plt.ylabel("Bad debt")
        if export_datafile:
            plt.savefig(Statistics.get_export_path(export_datafile).replace(".txt", "_bad_debt.svg"))
        else:
            plt.show()

    def plot_P(self, export_datafile=None):
        xx = []
        yy = []
        yy_min = []
        yy_max = []
        yy_std = []

        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.P[i])
            yy_min.append(self.P_min[i])
            yy_max.append(self.P_max[i])
            yy_std.append(self.P_std[i])

        plt.clf()
        plt.plot(xx, yy_min, ':', color="cyan", label="Max and min prob")
        plt.plot(xx, yy_max, ':', color="cyan")
        plt.plot(xx, yy_std, '-', color="aquamarine", label="Std")
        plt.plot(xx, yy, '-', color="blue", label="Avg prob with $\gamma$")
        plt.xlabel("Time")
        plt.title(f"Prob of change lender $\\gamma={Config.lenderchange.γ}$")
        plt.ylabel(f"Changes if >{self.model.config.lenderchange.CHANGE_LENDER_IF_HIGHER})")
        plt.legend()
        if export_datafile:
            plt.savefig(Statistics.get_export_path(export_datafile).replace(".txt", "_prob_change_lender.svg"))
        else:
            plt.show()

    def plot_liquidity(self, export_datafile=None):
        xx = []
        yy = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.liquidity[i])
        plt.clf()
        plt.plot(xx, yy, '-', color="blue")
        plt.xlabel("Time")
        plt.title("Liquidity")
        plt.ylabel("Ʃ liquidity")
        if export_datafile:
            plt.savefig(Statistics.get_export_path(export_datafile).replace(".txt", "_liquidity.svg"))
        else:
            plt.show()

    def plot_credit_channels(self, export_datafile=None):
        xx = []
        yy = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.credit_channels[i])
        plt.clf()
        plt.plot(xx, yy, '-', color="blue")
        plt.xlabel("Time")
        plt.title("Credit channels")
        plt.ylabel("Credit channels")
        if export_datafile:
            plt.savefig(Statistics.get_export_path(export_datafile).replace(".txt", "_credit_channels.svg"))
        else:
            plt.show()


    def plot_best_lender(self, export_datafile=None):
        xx = []
        yy = []
        yy2 = []
        max_duration = 0
        final_best_lender = -1
        current_lender = -1
        current_duration = 0
        time_init = 0
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.best_lender[i] / self.model.config.N)
            yy2.append(self.best_lender_clients[i] / self.model.config.N)
            if current_lender!=self.best_lender[i]:
                if max_duration < current_duration:
                    max_duration = current_duration
                    time_init = i - current_duration
                    final_best_lender = current_lender
                current_lender = self.best_lender[i]
                current_duration = 0
            else:
                current_duration += 1

        plt.clf()
        plt.figure(figsize=(14, 6))
        plt.plot(xx, yy, '-', color="blue", label="id")
        plt.plot(xx, yy2, '-', color="red", label="Num clients")
        plt.xlabel(f"Time (best lender={final_best_lender} at t=[{time_init}..{time_init+max_duration}])")

        xx3 = []
        yy3 = []
        for i in range(time_init,time_init+max_duration):
            xx3.append(i)
            yy3.append(self.best_lender[i] / self.model.config.N)
        plt.plot(xx3, yy3, '-', color="orange", linewidth=2.0)
        plt.title("Best Lender (blue) #clients (red)")
        plt.ylabel("Best Lender")
        if export_datafile:
            plt.savefig(Statistics.get_export_path(export_datafile).replace(".txt", "_best_lender.svg"))
        else:
            plt.show()


class Log:
    """
    The class acts as a logger and helpers to represent the data and evol from the Model.
    """
    logger = logging.getLogger("model")
    modules = []
    model = None
    logLevel = "ERROR"

    def __init__(self, model):
        self.model = model

    @staticmethod
    def __format_number__(number):
        result = f"{number:5.2f}"
        while len(result) > 5 and result[-1] == "0":
            result = result[:-1]
        while len(result) > 5 and result.find('.') > 0:
            result = result[:-1]
        return result

    def __get_string_debug_banks__(self, details, bank):
        text = f"{bank.getId():10} C={Log.__format_number__(bank.C)} L={Log.__format_number__(bank.L)}"
        amount_borrowed = 0
        list_borrowers = " borrows=["
        for bank_i in bank.activeBorrowers:
            list_borrowers += self.model.banks[bank_i].getId(short=True) + ","
            amount_borrowed += bank.activeBorrowers[bank_i]
        if amount_borrowed:
            text += f" l={Log.__format_number__(amount_borrowed)}"
            list_borrowers = list_borrowers[:-1] + "]"
        else:
            text += "        "
            list_borrowers = ""
        text += f" | D={Log.__format_number__(bank.D)} E={Log.__format_number__(bank.E)}"
        if details and hasattr(bank, 'd') and bank.d and bank.l:
            text += f" l={Log.__format_number__(bank.d)}"
        else:
            text += "        "
        if details and hasattr(bank, 's') and bank.s:
            text += f" s={Log.__format_number__(bank.s)}"
        else:
            if details and hasattr(bank, 'd') and bank.d:
                text += f" d={Log.__format_number__(bank.d)}"
            else:
                text += "        "
        if bank.failed:
            text += f" FAILED "
        else:
            if details and hasattr(bank, 'd') and bank.d > 0:
                text += f" lender{bank.getLender().getId(short=True)},r={bank.getLoanInterest():.2f}%"
            else:
                text += list_borrowers
        text += f" B={Log.__format_number__(bank.B)}" if bank.B else "        "
        return text

    def debug_banks(self, details: bool = True, info: str = ''):
        for bank in self.model.banks:
            if not info:
                info = "-----"
            self.info(info, self.__get_string_debug_banks__(details, bank))

    @staticmethod
    def get_level(option):
        try:
            return getattr(logging, option.upper())
        except AttributeError:
            logging.error(f" '--log' must contain a valid logging level and {option.upper()} is not.")
            sys.exit(-1)

    def debug(self, module, text):
        if self.modules == [] or module in self.modules:
            if text:
                self.logger.debug(f"t={self.model.t:03}/{module:6} {text}")

    def info(self, module, text):
        if self.modules == [] or module in self.modules:
            if text:
                self.logger.info(f" t={self.model.t:03}/{module:6} {text}")

    def error(self, module, text):
        if text:
            self.logger.error(f"t={self.model.t:03}/{module:6} {text}")

    def define_log(self, log: str, logfile: str = '', modules: str = '', script_name: str = "%(module)s"):
        self.modules = modules.split(",") if modules else []
        formatter = logging.Formatter('%(levelname)s-' + script_name + '- %(message)s')
        self.logLevel = Log.get_level(log.upper())
        self.logger.setLevel(self.logLevel)
        if logfile:
            # if not logfile.startswith(Statistics.OUTPUT_DIRECTORY):
            if not os.path.dirname(logfile):
                logfile = f"{Statistics.OUTPUT_DIRECTORY}/{logfile}"
            fh = logging.FileHandler(logfile, 'a', 'utf-8')
            fh.setLevel(self.logLevel)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setLevel(self.logLevel)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)


class Model:
    """
    It contains the banks and has the logic to execute the simulation
        import interbank
        model = interbank.Model( )
        model.configure( param=x )
        model.forward()
        μ = model.get_current_fitness()
        model.set_policy_recommendation( ŋ=0.5 )
    or
        model.set_policy_recommendation( 1 ) --> equal to n=0.5, because values are int 0,1,2 ---> float 0,0.5,1

    If you want to forward() and backward() step by step, you should use:
        model.configure( backward=True )
        # t=4
        model.set_policy_recommendation( ŋ=0.5 )
        model.forward()
        result = model.get_current_fitness() # t=5
        model.backward() # t=4 again



    """
    banks = []  # An array of Bank with size Model.config.N
    t: int = 0  # current value of time, t = 0..Model.config.T
    ŋ: float = 1  # eta : current value of policy recommendation
    test = False  # it's true when we are inside a test
    default_seed: int = 20579  # seed for this simulation
    backward_enabled = False  # if true, we can execute backward()
    policy_changes = 0

    # if not None, we will debug at this instant i, entering in interactive mode
    debug = None

    # if not None, it should be a list of t in which we generate a graph with lenders, as i.e., [0,800]
    save_graphs = None
    save_graphs_results = []

    log = None
    statistics = None
    config = None
    export_datafile = None
    export_description = None

    policy_actions_translation = [0.0, 0.5, 1.0]
    
    def __init__(self, **configuration):
        self.log = Log(self)
        self.statistics = Statistics(self)
        self.config = Config()
        if configuration:
            self.configure(**configuration)
        self.banks_copy = []

    def configure(self, **configuration):
        for attribute in configuration:
            if hasattr(self.config, attribute):
                current_value = getattr(self.config, attribute)
                if isinstance(current_value, int):
                    setattr(self.config, attribute, int(configuration[attribute]))
                else:
                    if isinstance(current_value, float):
                        setattr(self.config, attribute, float(configuration[attribute]))
                    else:
                        raise Exception(f"type of config {attribute} not allowed: {type(current_value)}")
            else:
                raise LookupError("attribute in config not found")
        self.initialize()

    def initialize(self, seed=None, dont_seed=False, save_graphs_instants=None,
                   export_datafile=None, export_description=None ):
        self.statistics.reset()
        if not dont_seed:
            random.seed(seed if seed else self.default_seed)
        self.save_graphs = save_graphs_instants
        self.banks = []
        self.t = 0
        self.policy_changes = 0
        self.export_datafile = export_datafile
        self.export_description = str(self.config) if export_description is None else export_description
        for i in range(self.config.N):
            self.banks.append(Bank(i, self))
        self.config.lenderchange.initialize_bank_relationships()

    def forward(self):
        self.initialize_step()
        if self.backward_enabled:
            self.banks_copy = copy.deepcopy(self.banks)
        self.do_shock("shock1")
        self.do_loans()
        self.log.debug_banks()
        self.do_shock("shock2")
        self.do_repayments()
        self.log.debug_banks()
        self.statistics.compute_liquidity()
        self.statistics.compute_credit_channels_and_best_lender()
        self.statistics.compute_interest()
        self.statistics.compute_fitness()
        self.statistics.compute_policy()
        self.statistics.compute_bad_debt()
        self.statistics.compute_leverage()
        self.statistics.compute_rationing()
        self.setup_links()
        self.statistics.compute_probability_of_lender_change()
        self.log.debug_banks()
        if self.save_graphs is not None and (self.save_graphs=='*' or self.t in self.save_graphs):
            filename = self.statistics.get_graph(self.t)
            if filename:
                self.save_graphs_results.append(filename)
        if self.debug and self.t == self.debug:
            import code
            code.interact(local=locals())
        self.t += 1

    def backward(self):
        if self.backward_enabled:
            if self.t > 0:
                self.banks = self.banks_copy
                self.t -= 1
            else:
                raise IndexError('t=0 and no backward is possible')
        else:
            raise AttributeError('enable_backward() before')

    def do_debug(self, debug):
        self.debug = debug

    def enable_backward(self):
        self.backward_enabled = True

    def simulate_full(self):
        for t in range(self.config.T):
            self.forward()

    def finish(self):
        if not self.test:
            self.statistics.export_data(self.export_datafile, self.export_description)
        summary = f"Finish: model T={self.config.T}  N={self.config.N}"
        if not self.__policy_recommendation_changed__():
            summary += f" ŋ={self.ŋ}"
        else:
            summary += " ŋ variate during simulation"
        self.log.info("*****", summary)
        if self.save_graphs=='*':
            self.statistics.create_animation_graph(self.save_graphs_results)
        return self.statistics.get_data()

    def set_policy_recommendation(self, n: int = None, ŋ: float = None, ŋ1: float = None):
        if ŋ1 is not None:
            n = round(ŋ1)
        if n is not None and ŋ is None:
            if type(n) is int:
                ŋ = self.policy_actions_translation[n]
            else:
                ŋ = float(n)
        if self.ŋ != ŋ:
            self.log.debug("*****", f"ŋ changed to {ŋ}")
            self.policy_changes += 1
        self.ŋ = ŋ
        
    def limit_to_two_policies(self):
        self.policy_actions_translation = [0.0, 1.0]

    def __policy_recommendation_changed__(self):
        return self.policy_changes > 1

    def get_current_fitness(self):
        """
        Determines the current μ of the model (does the sum of all μ)
        :return:
        float:  Ʃ banks.μ
        """
        return self.statistics.fitness[self.t - 1 if self.t > 0 else 0]

    def get_current_credit_channels(self):
        """
        Determines the number of credits channels USED (each bank has a possible lender, but only if
        it needs it borrows money. This number represents how many banks have set up a credit with a lender
        :return:
        int
        """
        return self.statistics.credit_channels[self.t - 1 if self.t > 0 else 0]

    def get_current_liquidity(self):
        """
        Returns the liquidity (the sum of the liquidity)
        :return:
        float:  Ʃ banks.C
        """
        return self.statistics.liquidity[self.t - 1 if self.t > 0 else 0]

    def get_current_interest_rate(self):
        """
        Returns the interest rate (the average of all banks)
        :return:
        float:  Ʃ banks.ir / config.N
        """
        return self.statistics.interest_rate[self.t - 1 if self.t > 0 else 0]

    def get_current_interest_rate_info(self):
        """
        Returns a tuple with  : max ir, min_ir and avg
        :return:
        (float,float,float)
        """
        max_ir = 0
        min_ir = 1e6
        for bank in self.banks:
            bank_ir = bank.getLoanInterest()
            if max_ir < bank_ir:
                max_ir = bank_ir
            if min_ir > bank_ir:
                min_ir = bank_ir
        return max_ir, min_ir, self.get_current_interest_rate()

    def get_current_liquidity_info(self):
        """
        Returns a tuple with  : max C, min C and avg C
        :return:
        (float,float,float)
        """
        max_c = 0
        min_c = 1e6
        for bank in self.banks:
            bank_c = bank.C
            if max_c < bank_c:
                max_c = bank_c
            if min_c > bank_c:
                min_c = bank_c
        return max_c, min_c, self.get_current_liquidity()

    def get_current_bankruptcies(self):
        """
        Returns the number of bankruptcies in this step
        :return:
        int:  Ʃ failed banks
        """
        return self.statistics.bankruptcy[self.t - 1 if self.t > 0 else 0]

    def do_shock(self, which_shock):
        # (equation 2)
        for bank in self.banks:
            bank.newD = bank.D * (self.config.µ + self.config.ω * random.random())
            bank.ΔD = bank.newD - bank.D
            bank.D = bank.newD
            if bank.ΔD >= 0:
                bank.C += bank.ΔD
                # if "shock1" then we can be a lender:
                if which_shock == "shock1":
                    bank.s = bank.C
                bank.d = 0  # it will not need to borrow
                if bank.ΔD > 0:
                    self.log.debug(which_shock,
                                   f"{bank.getId()} wins ΔD={bank.ΔD:.3f}")
            else:
                # if "shock1" then we cannot be a lender: we have lost deposits
                if which_shock == "shock1":
                    bank.s = 0
                if bank.ΔD + bank.C >= 0:
                    bank.d = 0  # it will not need to borrow
                    bank.C += bank.ΔD
                    self.log.debug(which_shock,
                                   f"{bank.getId()} loses ΔD={bank.ΔD:.3f}, covered by capital")
                else:
                    bank.d = abs(bank.ΔD + bank.C)  # it will need money
                    self.log.debug(which_shock,
                                   f"{bank.getId()} loses ΔD={bank.ΔD:.3f} but has only C={bank.C:.3f}")
                    bank.C = 0  # we run out of capital
            self.statistics.incrementD[self.t] += bank.ΔD

    def do_loans(self):
        for bank in self.banks:
            # decrement in which we should borrow
            if bank.d > 0:
                if bank.getLender().d > 0:
                    # if the lender has no increment then NO LOAN could be obtained: we fire sale L:
                    bank.doFiresalesL(bank.d, f"lender {bank.getLender().getId(short=True)} has no money", "loans")
                    bank.rationing = bank.d
                    bank.l = 0
                else:
                    # if the lender can give us money, but not enough to cover the loan we need also fire sale L:
                    if bank.d > bank.getLender().s:
                        bank.doFiresalesL(bank.d - bank.getLender().s,
                                          f"lender.s={bank.getLender().s:.3f} but need d={bank.d:.3f}", "loans")
                        bank.rationing = bank.d - bank.getLender().s
                        # only if lender has money, because if it .s=0, all is obtained by fire sales:
                        if bank.getLender().s > 0:
                            bank.l = bank.getLender().s  # amount of loan (wrote in the borrower)
                            self.statistics.credit_channels[self.t] += 1
                            bank.getLender().activeBorrowers[
                                bank.id] = bank.getLender().s  # amount of loan (wrote in the lender)
                            bank.getLender().C -= bank.l  # amount of loan that reduces lender capital
                            bank.getLender().s = 0
                    else:
                        bank.l = bank.d  # amount of loan (wrote in the borrower)
                        self.statistics.credit_channels[self.t] += 1
                        bank.getLender().activeBorrowers[bank.id] = bank.d  # amount of loan (wrote in the lender)
                        bank.getLender().s -= bank.d  # the loan reduces our lender's capacity to borrow to others
                        bank.getLender().C -= bank.d  # amount of loan that reduces lender capital
                        self.log.debug("loans",
                                       f"{bank.getId()} new loan l={bank.d:.3f} from {bank.getLender().getId()}")

            # the shock can be covered by own capital
            else:
                bank.l = 0
                if len(bank.activeBorrowers) > 0:
                    list_borrowers = ""
                    amount_borrowed = 0
                    for bank_i in bank.activeBorrowers:
                        list_borrowers += self.banks[bank_i].getId(short=True) + ","
                        amount_borrowed += bank.activeBorrowers[bank_i]
                    self.log.debug("loans", f"{bank.getId()} has a total of {len(bank.activeBorrowers)} loans with " +
                                   f"[{list_borrowers[:-1]}] of l={amount_borrowed}")

    def do_repayments(self):
        # first all borrowers must pay their loans:
        for bank in self.banks:
            if bank.l > 0:
                loan_profits = bank.getLoanInterest() * bank.l
                loan_to_return = bank.l + loan_profits
                # (equation 3)
                if loan_to_return > bank.C:
                    we_need_to_sell = loan_to_return - bank.C
                    bank.C = 0
                    bank.paidloan = bank.doFiresalesL(we_need_to_sell,
                                                      f"to return loan and interest {loan_to_return:.3f} > C={bank.C:.3f}",
                                                      "repay")
                # the firesales of line above could bankrupt the bank, if not, we pay "normally" the loan:
                else:
                    bank.C -= loan_to_return
                    bank.E -= loan_profits
                    bank.paidloan = bank.l
                    bank.l = 0
                    bank.getLender().s -= bank.l  # we reduce the  's' => the lender could have more loans
                    bank.getLender().C += loan_to_return  # we return the loan and it's profits
                    bank.getLender().E += loan_profits  # the profits are paid as E
                    self.log.debug("repay",
                                   f"{bank.getId()} pays loan {loan_to_return:.3f} (E={bank.E:.3f},C={bank.C:.3f}) to lender" +
                                   f" {bank.getLender().getId()} (ΔE={loan_profits:.3f},ΔC={bank.l:.3f})")

        # now  when ΔD<0 it's time to use Capital or sell L again
        # (now we have the loans cancelled, or the bank bankrputed):
        for bank in self.banks:
            if bank.d > 0 and not bank.failed:
                bank.doFiresalesL(bank.d, f"fire sales due to not enough C", "repay")

        for bank in self.banks:
            bank.activeBorrowers = {}
            if bank.failed:
                bank.replaceBank()
        self.log.debug("repay", f"this step ΔD={self.statistics.incrementD[self.t]:.3f} and " +
                       f"failures={self.statistics.bankruptcy[self.t]}")

    def initialize_step(self):
        for bank in self.banks:
            bank.B = 0
            bank.rationing = 0
        if self.t == 0:
            self.log.debug_banks()

    def setup_links(self):
        # (equation 5)
        # p = probability borrower not failing
        # c = lending capacity
        # h = borrower haircut (leverage of bank respect to the maximum)

        maxE = max(self.banks, key=lambda k: k.E).E
        maxC = max(self.banks, key=lambda k: k.C).C
        for bank in self.banks:
            bank.p = bank.E / maxE
            bank.λ = bank.L / bank.E
            bank.ΔD = 0

        maxλ = max(self.banks, key=lambda k: k.λ).λ
        for bank in self.banks:
            bank.h = bank.λ / maxλ
            bank.A = bank.C + bank.L  # bank.L / bank.λ + bank.D

        # determine c (lending capacity) for all other banks (to whom give loans):
        for bank in self.banks:
            bank.c = []
            for i in range(self.config.N):
                c = 0 if i == bank.id else (1 - self.banks[i].h) * self.banks[i].A
                bank.c.append(c)

        # (equation 6)
        minr = sys.maxsize
        lines = []
        for bank_i in self.banks:
            line1 = ""
            line2 = ""
            for j in range(self.config.N):
                try:
                    if j == bank_i.id:
                        bank_i.rij[j] = 0
                    else:
                        if self.banks[j].p == 0 or bank_i.c[j] == 0:
                            bank_i.rij[j] = self.config.r_i0
                        else:
                            bank_i.rij[j] = (self.config.Χ * bank_i.A -
                                             self.config.Φ * self.banks[j].A -
                                             (1 - self.banks[j].p) *
                                             (self.config.ξ * self.banks[j].A - bank_i.c[j])) \
                                            / (self.banks[j].p * bank_i.c[j])
                        if bank_i.rij[j] < 0:
                            bank_i.rij[j] = self.config.r_i0
                # the first t=1, maybe t=2, the shocks have not affected enough to use L (only C), so probably
                # L and E are equal for all banks, and so maxλ=anyλ and h=1 , so cij=(1-1)A=0, and r division
                # by zero -> solution then is to use still r_i0:
                except ZeroDivisionError:
                    bank_i.rij[j] = self.config.r_i0

                line1 += f"{bank_i.rij[j]:.3f},"
                line2 += f"{bank_i.c[j]:.3f},"
            if lines:
                lines.append("  |" + line2[:-1] + "|   |" +
                             line1[:-1] + f"| {bank_i.getId(short=True)} h={bank_i.h:.3f},λ={bank_i.λ:.3f} ")
            else:
                lines.append("c=|" + line2[:-1] + "| r=|" +
                             line1[:-1] + f"| {bank_i.getId(short=True)} h={bank_i.h:.3f},λ={bank_i.λ:.3f} ")
            bank_i.r = np.sum(bank_i.rij) / (self.config.N - 1)
            if bank_i.r < minr:
                minr = bank_i.r

        if self.config.N < 10:
            for line in lines:
                self.log.debug("links", f"{line}")
        self.log.debug("links", f"maxE={maxE:.3f} maxC={maxC:.3f} maxλ={maxλ:.3f} minr={minr:.3f} ŋ={self.ŋ:.3f}")

        # (equation 7)
        loginfo = loginfo1 = ""
        for bank in self.banks:
            bank.μ = self.ŋ * (bank.C / maxC) + (1 - self.ŋ) * (minr / bank.r)
            loginfo += f"{bank.getId(short=True)}:{bank.μ:.3f},"
            loginfo1 += f"{bank.getId(short=True)}:{bank.r:.3f},"
        if self.config.N <= 10:
            self.log.debug("links", f"μ=[{loginfo[:-1]}] r=[{loginfo1[:-1]}]")

        for bank in self.banks:
            self.log.debug("links",self.config.lenderchange.change_lender(model, bank, self.t))

# %%

class Bank:
    """
    It represents an individual bank of the network, with the logic of interaction between it and the interbank system
    """

    def getLender(self):
        return self.model.banks[self.lender]

    def getLoanInterest(self):
        return self.model.banks[self.lender].rij[self.id]

    def getId(self, short: bool = False):
        init = "bank#" if not short else "#"
        if self.failures > 0:
            return f"{init}{self.id}.{self.failures}"
        else:
            return f"{init}{self.id}"

    def __init__(self, new_id, model):
        self.id = new_id
        self.model = model
        self.failures = 0
        self.rationing = 0
        self.__assign_defaults__()

    def new_lender(self):
        # r_i0 is used the first time the bank is created:
        if self.lender is None:
            self.rij = np.full(self.model.config.N, self.model.config.r_i0, dtype=float)
            self.rij[self.id] = 0
            self.r = self.model.config.r_i0
            self.μ = 0
            # if it's just created, only not to be ourselves is enough
            new_value = random.randrange(self.model.config.N - 1)
        else:
            # if we have a previous lender, new should not be the same
            new_value = random.randrange(self.model.config.N - 2 if self.model.config.N > 2 else 1)

        if self.model.config.N == 2:
            new_value = 1 if self.id == 0 else 0
        else:
            if new_value >= self.id:
                new_value += 1
                if self.lender is not None and new_value >= self.lender:
                    new_value += 1
            else:
                if self.lender is not None and new_value >= self.lender:
                    new_value += 1
                    if new_value >= self.id:
                        new_value += 1
        return new_value

    def __assign_defaults__(self):
        self.L = self.model.config.L_i0
        self.C = self.model.config.C_i0
        self.D = self.model.config.D_i0
        self.E = self.model.config.E_i0
        self.μ = 0  # fitness of the bank:  estimated later
        self.l = 0  # amount of loan done:  estimated later
        self.s = 0  # amount of loan received: estimated later
        self.B = 0  # bad debt: estimated later
        self.failed = False

        # identity of the lender
        self.lender = None
        self.lender = self.new_lender()

        self.activeBorrowers = {}

    def replaceBank(self):
        self.failures += 1
        self.__assign_defaults__()

    def __doBankruptcy__(self, phase):
        self.failed = True
        self.model.statistics.bankruptcy[self.model.t] += 1
        recovered_in_fire_sales = self.L * self.model.config.ρ  # we firesale what we have
        recovered = recovered_in_fire_sales - self.D  # we should pay D to clients
        if recovered < 0:
            recovered = 0
        if recovered > self.l:
            recovered = self.l

        badDebt = self.l - recovered  # the fire sale minus paying D: what the lender recovers
        if badDebt > 0:
            self.paidLoan = recovered
            self.getLender().B += badDebt
            self.getLender().E -= badDebt
            self.getLender().C += recovered
            self.model.log.debug(phase, f"{self.getId()} bankrupted (fire sale={recovered_in_fire_sales:.3f}," +
                                 f"recovers={recovered:.3f},paidD={self.D:.3f})(lender{self.getLender().getId(short=True)}" +
                                 f".ΔB={badDebt:.3f},ΔC={recovered:.3f})")
        else:
            # self.l=0 no current loan to return:
            if self.l > 0:
                self.paidLoan = self.l  # the loan was paid, not the interest
                self.getLender().C += self.l  # lender not recovers more than loan if it is
                self.model.log.debug(phase, f"{self.getId()} bankrupted (lender{self.getLender().getId(short=True)}" +
                                     f".ΔB=0,ΔC={recovered:.3f}) (paidD={self.l:.3f})")
        self.D = 0
        # the loan is not paid correctly, but we remove it
        if self.id in self.getLender().activeBorrowers:
            self.getLender().s -= self.l
            del self.getLender().activeBorrowers[self.id]

    def doFiresalesL(self, amountToSell, reason, phase):
        costOfSell = amountToSell / self.model.config.ρ
        recoveredE = costOfSell * (1 - self.model.config.ρ)
        if costOfSell > self.L:
            self.model.log.debug(phase,
                                 f"{self.getId()} impossible fire sale sellL={costOfSell:.3f} > L={self.L:.3f}: {reason}")
            return self.__doBankruptcy__(phase)
        else:
            self.L -= costOfSell
            self.E -= recoveredE


            if self.L <= self.model.config.α:
                self.model.log.debug(phase,
                                     f"{self.getId()} new L={self.L:.3f} makes bankruptcy of bank: {reason}")
                return self.__doBankruptcy__(phase)
            else:
                if self.E <= self.model.config.α:
                    self.model.log.debug(phase,
                                         f"{self.getId()} new E={self.E:.3f} makes bankruptcy of bank: {reason}")
                    return self.__doBankruptcy__(phase)
                else:
                    self.model.log.debug(phase,
                                         f"{self.getId()} fire sale sellL={amountToSell:.3f} at cost {costOfSell:.3f} reducing" +
                                         f"E={recoveredE:.3f}: {reason}")
                    return amountToSell


class Utils:
    """
    Auxiliary class to encapsulate the
    """


    @staticmethod
    def __extract_t_values_from_arg__(param):
        if param is None:
            return None
        else:
            t = []
            if param == '*':
                return '*'
            else:
                for str_t in param.split(","):
                    t.append(int(str_t))
                    if t[-1] > Config.T or t[-1] < 0:
                        raise ValueError(f"{t[-1]} greater than Config.T or below 0")
                return t

    @staticmethod
    def run_interactive():
        """
            Run interactively the model
        """
        global model
        parser = argparse.ArgumentParser()
        parser.add_argument("--log", default='ERROR', help="Log level messages (ERROR,DEBUG,INFO...)")
        parser.add_argument("--modules", default=None, help=f"Log only this modules (separated by ,)")
        parser.add_argument("--logfile", default=None, help="File to send logs to")
        parser.add_argument("--save", default=None, help=f"Saves the output of this execution")
        parser.add_argument("--graph", default=None, help=f"List of t in which save the network config (* for all and an animated gif)")
        parser.add_argument("--n", type=int, default=Config.N, help=f"Number of banks")
        parser.add_argument("--debug", type=int, default=None,
                            help="Stop and enter in debug mode after at this time")
        parser.add_argument("--eta", type=float, default=Model.ŋ, help=f"Policy recommendation")
        parser.add_argument("--t", type=int, default=Config.T, help=f"Time repetitions")

        args = parser.parse_args()

        if args.t != model.config.T:
            model.config.T = args.t
        if args.n != model.config.N:
            model.config.N = args.n
        if args.eta != model.ŋ:
            model.ŋ = args.eta
        if args.debug:
            model.do_debug(args.debug)
        model.log.define_log(args.log, args.logfile, args.modules)
        Utils.run(args.save, Utils.__extract_t_values_from_arg__(args.graph))

    @staticmethod
    def run(save=None, save_graph_instants=None):
        global model
        if not save_graph_instants and Config.GRAPHS_MOMENTS:
            save_graph_instants = Config.GRAPHS_MOMENTS
        model.initialize(export_datafile=save, save_graphs_instants=save_graph_instants)
        model.simulate_full()
        return model.finish()

    @staticmethod
    def is_notebook():
        try:
            __IPYTHON__
            return get_ipython().__class__.__name__!="SpyderShell"
        except NameError:
            return False
    
    @staticmethod
    def is_spyder():
        try:
            return get_ipython().__class__.__name__=="SpyderShell"
        except:
            return False    

# %%


model = Model()
if Utils.is_notebook():
    # if we are running in a Notebook:
    Utils.run()
else:
    # if we are running interactively:
    if __name__ == "__main__":
        Utils.run_interactive()

# in other cases, if you import it, the process will be:
#   model = Model()
#   # step by step:
#   model.enable_backward()
#   model.forward() # t=0 -> t=1
#   model.backward() : reverts the last step (when executed
#   # all in a loop:
#   model.simulate_full()
