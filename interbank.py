# -*- coding: utf-8 -*-
"""
Generates a simulation of an interbank network following the rules described in paper
  Reinforcement Learning Policy Recommendation for Interbank Network Stability
  from Gabrielle and Alessio

  You can use it interactively, but if you import it, the process will be:
    # model = Model()
    #   # step by step:
    #   model.enable_backward()
    #   model.forward() # t=0 -> t=1
    #   model.backward() : reverts the last step (when executed)
    #   # all in a loop:
    #   model.simulate_full()

@author: hector@bith.net
@date:   04/2023
"""
import copy
import random
import logging
import argparse
import numpy as np
import networkx as nx
import sys
import os
import matplotlib.pyplot as plt
import interbank_lenderchange as lc
import pandas as pd
import lxml.etree
import lxml.builder
import gzip


class Config:
    """
    Configuration parameters for the interbank network
    """
    T: int = 1000  # time (1000)
    N: int = 50  # number of banks (50)

    reserves: float  = 0.02

    # seed applied for random values (set during initialize)
    seed = None

    # if False, when a bank fails it's not replaced and N is reduced
    allow_replacement_of_bankrupted = True

    # shocks parameters:
    mi: float = 0.7  # mi µ
    omega: float = 0.6  # omega ω

    # Lender's change mechanism
    lender_change: lc.LenderChange = None

    # screening costs
    phi: float = 0.025  # phi Φ
    ji: float = 0.015  # ji Χ

    # liquidation cost of collateral
    xi: float = 0.3  # xi ξ
    ro: float = 0.3  # ro ρ fire sale cost

    beta: float = 5  # β beta intensity of breaking the connection (5)
    alfa: float = 0.1  # α alfa below this level of E or D, we will bankrupt the bank

    # banks initial parameters
    # L + C + R = D + E
    # but R = 0.02*D and C_i0= 30-2.7=27.3 and R=2.7
    L_i0: float = 120  # long term assets
    C_i0: float = 30  # capital BEFORE RESERVES ESTIMATION, after it will be 27.3
    # R_i0=2.7
    D_i0: float = 135  # deposits
    E_i0: float = 15  # equity
    r_i0: float = 0.02  # initial rate

    # if enabled and != [] the values of t in the array (for instance [150,350]) will generate
    # a graph with the relations of the firms. If * all the instants will generate a graph, and also an animated gif
    # with the results
    GRAPHS_MOMENTS = []

    # what elements are in the results.csv file, and also which are plot.
    # 1 if also plot, 0 not to plot:
    ELEMENTS_STATISTICS = {'B': True, 'liquidity': True, 'interest_rate': True, 'asset_i': True, 'asset_j': True,
                           'equity': True, 'equity_borrowers': True, 'bankruptcy': True, 
                           'potential_credit_channels': True,
                           'P': True, 'best_lender': True,
                           'policy': False, 'fitness': False, 'best_lender_clients': False,
                           'rationing': True, 'leverage': False, 'systemic_leverage':False,
                           'loans': False,
                           'reserves':True, 'deposits':True,
                           'active_lenders': False, 'active_borrowers': False, 'prob_bankruptcy': False,
                           'num_banks': True, 'bankruptcy_rationed': True}

    def __str__(self):
        description = sys.argv[0] if __name__ == '__main__' else ''
        for attr in dir(self):
            value = getattr(self, attr)
            if isinstance(value, int) or isinstance(value, float):
                description += f" {attr}={value}"
        return description + " "


# %%

class Statistics:
    bankruptcy = []
    best_lender = []
    best_lender_clients = []
    potential_credit_channels = []
    liquidity = []
    policy = []
    deposits = []
    reserves = []
    interest_rate = []
    incrementD = []
    fitness = []
    rationing = []
    leverage = []
    systemic_leverage = []
    loans = []
    asset_i = []
    asset_j = []
    equity = []
    P = []
    B = []
    active_borrowers = []
    active_lenders = []
    prob_bankruptcy = []

    # only used if config.allow_replacement_of_bankrupted is false
    num_banks = []
    bankruptcy_rationed = []

    model = None
    graphs = {}
    graphs_pos = None
    plot_format = None
    graph_format = ".svg"
    output_format = ".gdt"
    create_gif = False

    OUTPUT_DIRECTORY = "output"
    NUMBER_OF_ITEMS_IN_ANIMATED_GRAPH = 40

    def __init__(self, in_model):
        self.model = in_model

    def set_gif_graph(self, gif_graph):
        if gif_graph:
            self.create_gif = True

    def reset(self, output_directory=None):
        if output_directory:
            self.OUTPUT_DIRECTORY = output_directory
        if not os.path.isdir(self.OUTPUT_DIRECTORY):
            os.mkdir(self.OUTPUT_DIRECTORY)
        self.best_lender = np.full(self.model.config.T, -1, dtype=int)
        self.best_lender_clients = np.zeros(self.model.config.T, dtype=int)
        self.potential_credit_channels = np.zeros(self.model.config.T, dtype=int)
        self.active_borrowers = np.zeros(self.model.config.T, dtype=int)
        self.prob_bankruptcy = np.zeros(self.model.config.T, dtype=float)
        self.active_lenders = np.zeros(self.model.config.T, dtype=int)
        self.fitness = np.zeros(self.model.config.T, dtype=float)
        self.interest_rate = np.zeros(self.model.config.T, dtype=float)
        self.asset_i = np.zeros(self.model.config.T, dtype=float)
        self.asset_j = np.zeros(self.model.config.T, dtype=float)
        self.equity = np.zeros(self.model.config.T, dtype=float)
        self.equity_borrowers = np.zeros(self.model.config.T, dtype=float)
        self.incrementD = np.zeros(self.model.config.T, dtype=float)
        self.liquidity = np.zeros(self.model.config.T, dtype=float)
        self.rationing = np.zeros(self.model.config.T, dtype=float)
        self.leverage = np.zeros(self.model.config.T, dtype=float)
        self.systemic_leverage = np.zeros(self.model.config.T, dtype=float)
        self.policy = np.zeros(self.model.config.T, dtype=float)
        self.P = np.zeros(self.model.config.T, dtype=float)
        self.P_max = np.zeros(self.model.config.T, dtype=float)
        self.P_min = np.zeros(self.model.config.T, dtype=float)
        self.P_std = np.zeros(self.model.config.T, dtype=float)
        self.B = np.zeros(self.model.config.T, dtype=float)
        self.loans = np.zeros(self.model.config.T, dtype=float)
        self.deposits = np.zeros(self.model.config.T, dtype=float)
        self.reserves = np.zeros(self.model.config.T, dtype=float)
        self.num_banks = np.zeros(self.model.config.T, dtype=int)
        self.bankruptcy = np.zeros(self.model.config.T, dtype=int)
        self.bankruptcy_rationed = np.zeros(self.model.config.T, dtype=int)

    def compute_credit_channels_and_best_lender(self):
        lenders = {}
        for bank in self.model.banks:
            if bank.lender is not None:
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

        # number of possible credit channels:
        credit_channels = self.model.config.lender_change.get_credit_channels()
        if credit_channels is None:
            self.potential_credit_channels[self.model.t] = len(self.model.banks)
        else:
            self.potential_credit_channels[self.model.t] = credit_channels

    def compute_interest_rates_and_loans(self):
        interests_rates_of_borrowers = []
        asset_i = []
        asset_j = []
        sum_of_loans = 0
        maxE = 0
        num_of_banks_that_are_lenders = 0
        num_of_banks_that_are_borrowers = 0
        for bank in self.model.banks:
            if bank.E > maxE:
                maxE = bank.E
            if bank.incrD >= 0:
                asset_i.append(bank.asset_i)
            else:
                asset_j.append(bank.asset_j)
            if bank.active_borrowers:
                num_of_banks_that_are_lenders += 1
                sum_of_loans += bank.l
            elif bank.l > 0:
                num_of_banks_that_are_borrowers += 1
                interests_rates_of_borrowers.append(bank.get_loan_interest())
        avg_prob_bankruptcy = []
        if maxE > 0:
            for bank in self.model.banks:
                if bank.get_loan_interest() is not None and bank.l > 0:
                    avg_prob_bankruptcy.append( (1 - bank.E / maxE) )
        avg_prob_bankruptcy = sum(map(lambda x: x, avg_prob_bankruptcy)) / len(avg_prob_bankruptcy) if avg_prob_bankruptcy else 0
        self.interest_rate[self.model.t] = (sum(map(lambda x: x, interests_rates_of_borrowers)) /
                                            len(interests_rates_of_borrowers)) if interests_rates_of_borrowers else 0
        self.asset_i[self.model.t] = sum(map(lambda x: x, asset_i)) / len(asset_i) if len(asset_i)>0 else np.nan
        self.asset_j[self.model.t] = sum(map(lambda x: x, asset_j)) / len(asset_j) if len(asset_j)>0 else np.nan
        self.loans[self.model.t] = sum_of_loans / num_of_banks_that_are_lenders if num_of_banks_that_are_lenders else np.nan
        self.prob_bankruptcy[self.model.t] = avg_prob_bankruptcy
        self.active_lenders[self.model.t] = num_of_banks_that_are_lenders
        self.active_borrowers[self.model.t] = num_of_banks_that_are_borrowers

    def compute_leverage_and_equity(self):
        sum_of_equity = 0
        sum_of_equity_borrowers = 0
        leverage_of_borrowers = []
        for bank in self.model.banks:
            sum_of_equity += bank.E
            if bank.get_loan_interest() is not None and bank.l > 0:
                leverage_of_borrowers.append(bank.l / bank.E)
            if bank.active_borrowers:
                for borrower in bank.active_borrowers:
                    sum_of_equity_borrowers += self.model.banks[borrower].E
        self.equity[self.model.t] = sum_of_equity
        self.equity_borrowers[self.model.t] = sum_of_equity_borrowers
        self.leverage[self.model.t] = sum(map(lambda x: x, leverage_of_borrowers)) / len(leverage_of_borrowers) if len(leverage_of_borrowers) else 0
        self.systemic_leverage[self.model.t] = sum(map(lambda x: x, leverage_of_borrowers)) / len(self.model.banks)

    def compute_liquidity(self):
        self.liquidity[self.model.t] = sum(map(lambda x: x.C, self.model.banks))

    def compute_fitness(self):
        if self.model.config.N > 0:
            self.fitness[self.model.t] = sum(map(lambda x: x.mu, self.model.banks)) / self.model.config.N

    def compute_policy(self):
        self.policy[self.model.t] = self.model.eta

    def compute_bad_debt(self):
        self.B[self.model.t] = sum(map(lambda x: x.B, self.model.banks))

    def compute_rationing(self):
        self.rationing[self.model.t] = sum(map(lambda x: x.rationing, self.model.banks))

    def compute_deposits_and_reserves(self):
        self.deposits[self.model.t] = sum(map(lambda x: x.D, self.model.banks))
        self.reserves[self.model.t] = sum(map(lambda x: x.R, self.model.banks))

    def compute_probability_of_lender_change_and_num_banks(self):
        probabilities = [bank.P for bank in self.model.banks]
        self.P[self.model.t] = sum(probabilities) / self.model.config.N
        self.P_max[self.model.t] = max(probabilities)
        self.P_min[self.model.t] = min(probabilities)
        self.P_std[self.model.t] = np.std(probabilities)
        # only if we don't replace banks makes sense to report how many banks we have in each step:
        #if not self.config.allow_replacement_of_bankrupted:
        self.num_banks[self.model.t] = len(self.model.banks)

    def export_data(self, export_datafile=None, export_description=None, generate_plots=True):
        if export_datafile:
            self.save_data(export_datafile, export_description)
            if generate_plots:
                self.get_plots(export_datafile)
        if Utils.is_notebook() or Utils.is_spyder():
            self.get_plots(None)

    def get_graph(self, t):
        """
        Extracts from the model the graph that corresponds to the network in this instant
        """
        if 'unittest' in sys.modules.keys():
            return None
        else:
            self.graphs[t] = nx.DiGraph(directed=True)
            for bank in self.model.banks:
                if bank.lender is not None:
                    self.graphs[t].add_edge(bank.lender, bank.id)
            lc.draw(self.graphs[t], new_guru_look_for=True, title=f"t={t}")
            if Utils.is_spyder():
                plt.show()
                filename = None
            else:
                filename = sys.argv[0] if self.model.export_datafile is None else self.model.export_datafile
                filename = self.get_export_path(filename, f"_{t}{self.graph_format}")
                plt.savefig(filename)
            plt.close()
            return filename

    def define_plot_format(self, plot_format):
        match plot_format.lower():
            case 'none':
                self.plot_format = None
            case 'svg':
                self.plot_format = '.svg'
            case 'png':
                self.plot_format = '.png'
            case 'gif':
                self.plot_format = '.gif'
            case 'pdf':
                self.plot_format = '.pdf'
            case 'agr':
                self.plot_format = '.agr'
            case _:
                print(f'Invalid plot file format: {plot_format}')
                sys.exit(-1)

    def define_output_format(self, output_format):
        match output_format.lower():
            case 'both':
                self.output_format = '.both'
            case "gdt":
                self.output_format = '.gdt'
            case 'csv':
                self.output_format = '.csv'
            case 'txt':
                self.output_format = '.txt'
            case _:
                print(f'Invalid output file format: {output_format}')
                sys.exit(-1)

    def create_gif_with_graphs(self, list_of_files):
        if len(list_of_files) == 0 or not self.create_gif:
            return
        else:
            if len(list_of_files) > self.NUMBER_OF_ITEMS_IN_ANIMATED_GRAPH:
                positions_of_images = len(list_of_files) / self.NUMBER_OF_ITEMS_IN_ANIMATED_GRAPH
            else:
                positions_of_images = 1
            filename_output = sys.argv[0] if self.model.export_datafile is None else self.model.export_datafile
            filename_output = self.get_export_path(filename_output, '.gif')
            images = []
            from PIL import Image
            for idx, image_file in enumerate(list_of_files):
                # if more >40 images, only those that are divisible by 40 are incorporated:
                if not (idx % positions_of_images == 0):
                    continue
                images.append(Image.open(image_file))
            images[0].save(fp=filename_output, format='GIF', append_images=images[1:],
                           save_all=True, duration=100, loop=0)

    def get_export_path(self, filename, ending_name=''):
        # we ensure that the output goes to OUTPUT_DIRECTORY:
        if not os.path.dirname(filename):
            filename = f"{self.OUTPUT_DIRECTORY}/{filename}"
        path, extension = os.path.splitext(filename)
        # if there is an ending_name it means that we don't want the output.csv, we are using the
        # function to generate a plot_file, for instance:
        if ending_name:
            return path + ending_name
        else:
            # we ensure that the output goes with the correct extension:
            return path + self.output_format.lower()

    def __generate_csv_or_txt(self, export_datafile, header, delimiter):
        with open(export_datafile, 'w', encoding="utf-8") as savefile:
            for line_header in header:
                savefile.write(f"# {line_header}\n")
            savefile.write(f"# pd.read_csv('file{self.output_format}',header={len(header) + 1}',"
                           f" delimiter='{delimiter}')\nt")
            for element_name, _ in self.enumerate_results():
                savefile.write(f"{delimiter}{element_name}")
            savefile.write("\n")
            for i in range(self.model.config.T):
                savefile.write(f"{i}")
                for _, element in self.enumerate_results():
                    savefile.write(f"{delimiter}{element[i]}")
                savefile.write(f"\n")

    def __generate_gdt(self, export_datafile, header):
        E = lxml.builder.ElementMaker()
        GRETLDATA = E.gretldata
        DESCRIPTION = E.description
        VARIABLES = E.variables
        VARIABLE = E.variable
        OBSERVATIONS = E.observations
        OBS = E.obs
        variables = VARIABLES(count=f"{sum(1 for _ in self.enumerate_results())}")
        for variable_name, _ in self.enumerate_results():
            if variable_name == 'leverage':
                variable_name += "_"
            variables.append(VARIABLE(name=f"{variable_name}"))

        observations = OBSERVATIONS(count=f"{self.model.config.T}", labels="false")
        for i in range(self.model.config.T):
            string_obs = ''
            for _, variable in self.enumerate_results():
                string_obs += f"{variable[i]}  "
            observations.append(OBS(string_obs))
        header_text = ""
        for item in header:
            header_text += item + " "
        gdt_result = GRETLDATA(
            DESCRIPTION(header_text),
            variables,
            observations,
            version="1.4", name='interbank', frequency="special:1", startobs="1",
            endobs=f"{self.model.config.T}", type="time-series"
        )
        with gzip.open(self.get_export_path(export_datafile), 'w') as output_file:
            output_file.write(
                b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
            output_file.write(
                lxml.etree.tostring(gdt_result, pretty_print=True, encoding=str).encode('ascii'))

    @staticmethod
    def __transform_line_from_string(line_with_values):
        items = []
        for i in line_with_values.replace("  ", " ").strip().split(" "):
            try:
                items.append(int(i))
            except ValueError:
                items.append(float(i))
        return items

    @staticmethod
    def read_gdt(filename):
        tree = lxml.etree.parse(filename)
        root = tree.getroot()
        children = root.getchildren()
        values = []
        columns = []
        if len(children) == 3:
            # children[0] = description
            # children[1] = variables
            # children[2] = observations
            for variable in children[1].getchildren():
                column_name = variable.values()[0].strip()
                if column_name == 'leverage_':
                    column_name = 'leverage'
                columns.append(column_name)
            for value in children[2].getchildren():
                values.append(Statistics.__transform_line_from_string(value.text))
        if columns and values:
            return pd.DataFrame(columns=columns, data=values)
        else:
            return pd.DataFrame()

    def save_data(self, export_datafile=None, export_description=None):
        if export_datafile:
            if export_description:
                header = [f"{export_description}"]
            else:
                header = [f"{__name__} T={self.model.config.T} N={self.model.config.N}"]
            if self.output_format.lower() == '.both':
                self.output_format = '.csv'
                self.__generate_csv_or_txt(self.get_export_path(export_datafile), header, ';')
                self.output_format = '.gdt'
                self.__generate_gdt(self.get_export_path(export_datafile), header)
            elif self.output_format.lower() == '.csv':
                self.__generate_csv_or_txt(self.get_export_path(export_datafile), header, ';')
            elif self.output_format.lower() == '.txt':
                self.__generate_csv_or_txt(self.get_export_path(export_datafile), header, '\t')
            else:
                self.__generate_gdt(self.get_export_path(export_datafile), header)

    def enumerate_results(self):
        for element in Config.ELEMENTS_STATISTICS:
            yield self.get_name(element), getattr(self, element)

    def get_name(self, variable):
        match variable:
            case 'bankruptcy':
                return 'bankruptcies'
            case 'P':
                return 'prob_change_lender'
            case 'B':
                return 'bad_debt'
        return variable

    def get_data(self):
        result = pd.DataFrame()
        for variable_name, variable in self.enumerate_results():
            result[variable_name] = np.array(variable)
        return result.iloc[0:self.model.t]

    def plot_pygrace(self, xx, yy_s, variable, title, export_datafile, x_label, y_label):
        from pygrace.project import Project
        plot = Project()
        graph = plot.add_graph()
        graph.title.text = title.capitalize().replace("_", ' ')
        for (yy, color, ticks, title_y) in yy_s:
            data = []
            if isinstance(yy, tuple):
                for i in range(len(yy[0])):
                    data.append((yy[0][i], yy[1][i]))
            else:
                for i in range(len(xx)):
                    data.append((xx[i], yy[i]))
            dataset = graph.add_dataset(data, legend=title_y)
            dataset.symbol.fill_color = color
        graph.xaxis.label.text = x_label
        graph.yaxis.label.text = y_label
        graph.set_world_to_limits()
        graph.autoscale()
        if export_datafile:
            if self.plot_format:
                plot.saveall(self.get_export_path(export_datafile, f"_{variable.lower()}{self.plot_format}"))

    def plot_pyplot(self, xx, yy_s, variable, title, export_datafile, x_label, y_label):
        if self.plot_format == '.agr':
            self.plot_pygrace(xx, yy_s, variable, title, export_datafile, x_label, y_label)
        else:
            plt.clf()
            plt.figure(figsize=(14, 6))
            for (yy, color, ticks, title_y) in yy_s:
                if isinstance(yy, tuple):
                    plt.plot(yy[0], yy[1], ticks, color=color, label=title_y, linewidth=0.2)
                else:
                    plt.plot(xx, yy, ticks, color=color, label=title_y)
            plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)
            plt.title(title.capitalize().replace("_", ' '))
            if len(yy_s) > 1:
                plt.legend()
            if export_datafile:
                if self.plot_format:
                    plt.savefig(self.get_export_path(export_datafile, f"_{variable.lower()}{self.plot_format}"))
            else:
                plt.show()
            plt.close()

    def plot_result(self, variable, title, export_datafile=None):
        xx = []
        yy = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(getattr(self, variable)[i])
        self.plot_pyplot(xx, [(yy, 'blue', '-', '')], variable, title, export_datafile, "Time", '')

    def get_plots(self, export_datafile):
        for variable in Config.ELEMENTS_STATISTICS:
            if Config.ELEMENTS_STATISTICS[variable]:  # True if they are going to be plot
                if f'plot_{variable}' in dir(Statistics):
                    eval(f'self.plot_{variable}(export_datafile)')
                else:
                    self.plot_result(variable, self.get_name(variable), export_datafile)

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
        self.plot_pyplot(xx, [(yy, 'blue', '-', 'Avg prob with $\\gamma$'),
                              (yy_min, "cyan", ':', "Max and min prob"),
                              (yy_max, "cyan", ':', ''),
                              (yy_std, "red", '-', "Std")
                              ],
                         'prob_change_lender', "Prob of change lender " + self.model.config.lender_change.describe(),
                         export_datafile, 'Time', '')

    def plot_num_banks(self, export_datafile=None):
        # we plot only if we have allow_replacement_of_bankrupted, in the csv/gdt is always saved:
        if not self.model.config.allow_replacement_of_bankrupted:
            self.plot_result("num_banks", self.get_name("num_banks"), export_datafile)

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
            if current_lender != self.best_lender[i]:
                if max_duration < current_duration:
                    max_duration = current_duration
                    time_init = i - current_duration
                    final_best_lender = current_lender
                current_lender = self.best_lender[i]
                current_duration = 0
            else:
                current_duration += 1

        xx3 = []
        yy3 = []
        for i in range(time_init, time_init + max_duration):
            xx3.append(i)
            yy3.append(self.best_lender[i] / self.model.config.N)
        self.plot_pyplot(xx, [(yy, 'blue', '-', 'id'),
                              (yy2, "red", '-', "Num clients"),
                              ((xx3, yy3), "orange", '-', "")
                              ],
                         'best_lender', "Best Lender (blue) #clients (red)",
                         export_datafile,
                         f"Time (best lender={final_best_lender} at t=[{time_init}..{time_init + max_duration}])",
                         "Best Lender")


class Log:
    """
    The class acts as a logger and helpers to represent the data and evol from the Model.
    """
    logger = logging.getLogger("model")
    modules = []
    model = None
    logLevel = "ERROR"
    progress_bar = None
    graphical = False

    def __init__(self, its_model):
        self.model = its_model

    def define_gui(self, gui):
        self.graphical = gui.gooey

    def do_progress_bar(self, message, maximum):
        if self.graphical:
            self.progress_bar = Gui()
            self.progress_bar.progress_bar(message, maximum)
        else:
            from progress.bar import Bar
            self.progress_bar = Bar(message, max=maximum)

    @staticmethod
    def __format_number__(number):
        result = f"{number:5.2f}"
        while len(result) > 5 and result[-1] == "0":
            result = result[:-1]
        while len(result) > 5 and result.find('.') > 0:
            result = result[:-1]
        return result

    def __get_string_debug_banks__(self, details, bank):
        text = f"{bank.get_id(short=True):6} C={Log.__format_number__(bank.C)} R={Log.__format_number__(bank.R)} L={Log.__format_number__(bank.L)}"
        amount_borrowed = 0
        list_borrowers = " borrows=["
        for bank_i in bank.active_borrowers:
            list_borrowers += self.model.banks[bank_i].get_id(short=True) + ","
            amount_borrowed += bank.active_borrowers[bank_i]
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
                if bank.get_lender() is None:
                    text += f" no lender"
                else:
                    text += f" lender{bank.get_lender().get_id(short=True)},r={bank.get_loan_interest():.2f}%"
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
        formatter = logging.Formatter('%(levelname)s-'  + '- %(message)s')
        self.logLevel = Log.get_level(log.upper())
        self.logger.setLevel(self.logLevel)
        if logfile:
            if not os.path.dirname(logfile):
                logfile = f"{self.model.statistics.OUTPUT_DIRECTORY}/{logfile}"
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
    eta: float = 1  # ŋ eta : current value of policy recommendation
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

    generate_plots = True

    def __init__(self, **configuration):
        self.log = Log(self)
        self.statistics = Statistics(self)
        self.config = Config()
        if configuration:
            self.configure(**configuration)
        if self.backward_enabled:
            self.banks_backward_copy = []

    def configure(self, **configuration):
        for attribute in configuration:
            if attribute.startswith('lc'):
                attribute = attribute.replace("lc_", "")
                if attribute == 'lc':
                    self.config.lender_change = lc.determine_algorithm(configuration[attribute])
                    print(self.config.lender_change.__name__)
                else:
                    self.config.lender_change.set_parameter(attribute, configuration["lc_" + attribute])
            elif hasattr(self.config, attribute):
                current_value = getattr(self.config, attribute)
                if isinstance(current_value, int):
                    setattr(self.config, attribute, int(configuration[attribute]))
                else:
                    if isinstance(current_value, float):
                        setattr(self.config, attribute, float(configuration[attribute]))
                    else:
                        raise Exception(f"type of config {attribute} not allowed: {type(current_value)}")
            else:
                raise LookupError("attribute in config not found: %s " % attribute)
        self.initialize()

    def initialize(self, seed=None, dont_seed=False, save_graphs_instants=None,
                   export_datafile=None, export_description=None, generate_plots=True, output_directory=None):
        self.statistics.reset(output_directory=output_directory)
        if not dont_seed:
            applied_seed = seed if seed else self.default_seed
            random.seed(applied_seed)
            self.config.seed = applied_seed
        self.save_graphs = save_graphs_instants
        self.banks = []
        self.t = 0
        if not self.config.lender_change:
            self.config.lender_change = lc.determine_algorithm()
        self.policy_changes = 0
        if export_datafile:
            self.export_datafile = export_datafile
        if generate_plots:
            self.generate_plots = generate_plots
        self.export_description = str(self.config) if export_description is None else export_description
        for i in range(self.config.N):
            self.banks.append(Bank(i, self))
        self.config.lender_change.initialize_bank_relationships(self)

    def forward(self):
        self.initialize_step()
        if self.backward_enabled:
            self.banks_backward_copy = copy.deepcopy(self.banks)
        self.do_shock("shock1")
        self.do_loans()
        self.log.debug_banks()
        self.statistics.compute_interest_rates_and_loans()
        self.do_shock("shock2")
        self.do_repayments()
        #TODO equity
        #    after E=E+pi-B+(1-p)A
        #  leverage and equity
        self.log.debug_banks()
        if self.log.progress_bar:
            self.log.progress_bar.next()
        self.statistics.compute_leverage_and_equity()
        self.statistics.compute_liquidity()
        self.statistics.compute_credit_channels_and_best_lender()
        self.statistics.compute_fitness()
        self.statistics.compute_policy()
        self.statistics.compute_bad_debt()
        self.statistics.compute_rationing()
        self.statistics.compute_deposits_and_reserves()
        # only N<=1 could happen if we remove banks:
        if self.config.N > 1:
            self.setup_links()
            self.statistics.compute_probability_of_lender_change_and_num_banks()
        self.log.debug_banks()
        if self.save_graphs is not None and (self.save_graphs == '*' or self.t in self.save_graphs):
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
                self.banks = self.banks_backward_copy
                self.t -= 1
            else:
                raise IndexError('t=0 and no backward is possible')
        else:
            raise AttributeError('enable_backward() before')

    def do_debug(self, debug):
        self.debug = debug

    def enable_backward(self):
        self.backward_enabled = True

    def simulate_full(self, interactive=False):
        if interactive:
            self.log.do_progress_bar(f"Simulating t=0..{self.config.T}", self.config.T)
        for t in range(self.config.T):
            self.forward()
            # if we don't replace the bankrupted banks, and there are no banks  (we need at least two), we finish:
            if not self.config.allow_replacement_of_bankrupted and len(self.banks) <= 2:
                self.config.T = self.t
                self.log.debug("*****", f"Finish because there are only two banks surviving")
                break

    def finish(self):
        if not self.test:
            self.statistics.export_data(export_datafile=self.export_datafile,
                                        export_description=self.export_description,
                                        generate_plots=self.generate_plots)
        summary = f"Finish: model T={self.config.T}  N={self.config.N}"
        if not self.__policy_recommendation_changed__():
            summary += f" ŋ={self.eta}"
        else:
            summary += " ŋ variate during simulation"
        self.log.info("*****", summary)
        self.statistics.create_gif_with_graphs(self.save_graphs_results)
        plt.close()
        return self.statistics.get_data()

    def set_policy_recommendation(self, n: int = None, eta: float = None, eta_1: float = None):
        if eta_1 is not None:
            n = round(eta_1)
        if n is not None and eta is None:
            if type(n) is int:
                eta = self.policy_actions_translation[n]
            else:
                eta = float(n)
        if self.eta != eta:
            self.log.debug("*****", f"eta(ŋ) changed to {eta}")
            self.policy_changes += 1
        self.eta = eta

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
        min_ir = np.inf
        for bank in self.banks:
            bank_ir = bank.get_loan_interest()
            if bank_ir is not None:
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


    def determine_shock_value(self, bank, _shock):
        return bank.D * (self.config.mi + self.config.omega * random.random())

    def do_shock(self, which_shock):
        # (equation 2)
        for bank in self.banks:
            bank.newD = self.determine_shock_value(bank, which_shock)
            bank.incrD = bank.newD - bank.D
            bank.D = bank.newD
            bank.newR = self.config.reserves * bank.D
            bank.incrR = bank.newR - bank.R
            bank.R = bank.newR

            if bank.incrD >= 0:
                bank.C += bank.incrD - bank.incrR
                # if "shock1" then we can be a lender:
                if which_shock == "shock1":
                    bank.s = bank.C
                bank.d = 0  # it will not need to borrow
                if bank.incrD > 0:
                    self.log.debug(which_shock,
                                   f"{bank.get_id()} wins ΔD={bank.incrD:.3f}")
            else:
                # if "shock1" then we cannot be a lender: we have lost deposits
                if which_shock == "shock1":
                    bank.s = 0
                if bank.incrD - bank.incrR + bank.C >= 0:
                    bank.d = 0  # it will not need to borrow
                    bank.C += bank.incrD - bank.incrR
                    self.log.debug(which_shock,
                                   f"{bank.get_id()} loses ΔD={bank.incrD:.3f}, covered by capital")
                else:
                    bank.d = abs(bank.incrD - bank.incrR + bank.C)  # it will need money
                    self.log.debug(which_shock,
                                   f"{bank.get_id()} loses ΔD={bank.incrD:.3f} but has only C={bank.C:.3f}")
                    bank.C = 0  # we run out of capital
            self.statistics.incrementD[self.t] += bank.incrD

    def do_loans(self):
        for bank_index, bank in enumerate(self.banks):
            # decrement in which we should borrow
            if bank.d > 0:

                if bank.get_lender() is None or bank.get_lender().d > 0:
                    bank.l = 0
                    bank.rationing = bank.d
                    bank.do_fire_sales(bank.rationing,
                                       f"no lender for this bank" if bank.get_lender() is None else
                                       f"lender {bank.get_lender().get_id(short=True)} has no money", "loans")
                else:
                    # if the lender can give us money, but not enough to cover the loan we need also fire sale L:
                    if bank.d > bank.get_lender().s:
                        bank.rationing = bank.d - bank.get_lender().s
                        bank.do_fire_sales(bank.rationing,
                                           f"lender.s={bank.get_lender().s:.3f} but need d={bank.d:.3f}", "loans")
                        # only if lender has money, because if it .s=0, all is obtained by fire sales:
                        if bank.get_lender().s > 0:
                            bank.l = bank.get_lender().s  # amount of loan (wrote in the borrower)
                            # amount of loan (wrote in the lender)
                            bank.get_lender().active_borrowers[bank_index] = bank.get_lender().s
                            bank.get_lender().C -= bank.l  # amount of loan that reduces lender capital
                            bank.get_lender().s = 0
                    else:
                        bank.rationing = 0
                        bank.l = bank.d  # amount of loan (wrote in the borrower)
                        bank.get_lender().active_borrowers[bank_index] = bank.d  # amount of loan (wrote in the lender)
                        bank.get_lender().s -= bank.d  # the loan reduces our lender's capacity to borrow to others
                        bank.get_lender().C -= bank.d  # amount of loan that reduces lender capital
                        self.log.debug("loans",
                                       f"{bank.get_id()} new loan l={bank.d:.3f} from {bank.get_lender().get_id()}")

            # the shock can be covered by own capital
            else:
                bank.l = 0
                if bank.active_borrowers:
                    list_borrowers = ""
                    amount_borrowed = 0
                    for bank_i in bank.active_borrowers:
                        list_borrowers += self.banks[bank_i].get_id(short=True) + ","
                        amount_borrowed += bank.active_borrowers[bank_i]
                    self.log.debug("loans", f"{bank.get_id()} has a total of {len(bank.active_borrowers)} loans with " +
                                   f"[{list_borrowers[:-1]}] of l={amount_borrowed}")

    def do_repayments(self):
        # first deposits, which are the preferent payments, but only if we are borrowers:
        for bank in self.banks:
            if bank.l > 0: # if we are borrowers
                if bank.d > 0 and not bank.failed:  # and we have less deposits (incrD=d)
                    bank.do_fire_sales(bank.d, f"fire sales due to not enough C", "repay")

        # second all borrowers must pay their loans:
        for bank in self.banks:
            if bank.l > 0:
                loan_profits = bank.get_loan_interest() * bank.l
                loan_to_return = bank.l + loan_profits
                # (equation 3)
                if loan_to_return > bank.C:
                    # we need to fire sale to cover the debt:
                    lack_of_capital_to_return_loan = loan_to_return - bank.C
                    bank.C = 0
                    returned_by_borrower = bank.do_fire_sales(
                        lack_of_capital_to_return_loan,
                        f"to return loan and interest {loan_to_return:.3f} > C={bank.C:.3f}",
                        "repay")
                    # if returns None if borrower fails, in which case the bad debt and the cancel of loan it is
                    # done inside do_fire_sales() -> __do_bankruptcy__(). But that cancel after bankruptcy does not
                    # consider the profits:
                    if returned_by_borrower is not None:
                        bank.get_lender().s -= bank.l  # we reduce the  's' => the lender could have more loans
                        bank.get_lender().C += loan_profits  # we return the loan and it's profits
                        bank.get_lender().E += loan_profits  # the profits are paid as E
                        bank.paid_loan = returned_by_borrower
                    else:
                        bank.paid_loan = 0
                else:
                    # the can pay the debt normally:
                    bank.C -= loan_to_return
                    bank.paid_loan = loan_to_return
                    bank.get_lender().s -= bank.l  # we reduce the  's' => the lender could have more loans
                    bank.get_lender().C += bank.l  # we return the loan and it's profits
                    bank.get_lender().E += loan_profits  # the profits are paid as E
                bank.E -= loan_profits


                # now we have in bank.paid_loan or l
                #    self.log.debug(
                #        "repay",
                #        f"{bank.get_id()} pays loan {loan_to_return:.3f} (E={bank.E:.3f},C={bank.C:.3f}) to lender" +
                #        f" {bank.get_lender().get_id()} (ΔE={loan_profits:.3f},ΔC={bank.l:.3f})")

        # now we should analyze the banks that were lenders. They can have in .d a value (that
        # should mean that they don't have enough C to cover the negative shock of D) but maybe
        # they have had an income of a paid loan by its borrowers, so let's ignore .d and check
        # again if C > incrD:
        for bank in self.banks:
            if bank.l == 0: # if they are lenders, borrowers have finished at this point
                if bank.C < bank.incrD:
                    bank.d = bank.incrD - bank.C
                else:
                    bank.d = 0
                # and if still they have .d >0 , its means that it should do a fire sale:
                if bank.d > 0:  # and we have less deposits (incrD=d)
                    bank.do_fire_sales(bank.d, f"fire sales due to not enough C", "repay")

        self.statistics.bankruptcy_rationed[self.t] = self.replace_bankrupted_banks()


    def replace_bankrupted_banks(self):
        lists_to_remove_because_replacement_of_bankrupted_is_disabled = []
        num_banks_failed_rationed = 0
        total_removed = 0
        for possible_removed_bank in self.banks:
            if possible_removed_bank.failed:
                total_removed += 1
                if self.config.allow_replacement_of_bankrupted:
                    possible_removed_bank.replace_bank()
                else:
                    if possible_removed_bank.rationing == 0 and possible_removed_bank.lender is not None:
                        num_banks_failed_rationed += 1
                    lists_to_remove_because_replacement_of_bankrupted_is_disabled.append(possible_removed_bank)
        self.log.debug("repay", f"this step ΔD={self.statistics.incrementD[self.t]:.3f} and " +
                       f"failures={total_removed}")
        if not self.config.allow_replacement_of_bankrupted:
            for bank_to_remove in lists_to_remove_because_replacement_of_bankrupted_is_disabled:
                self.__remove_without_replace_failed_bank(bank_to_remove)
            # we update the number of banks we have:
            self.config.N -= len(lists_to_remove_because_replacement_of_bankrupted_is_disabled)
            self.log.debug("repay", f"now we have {self.config.N} banks")
        return num_banks_failed_rationed

    def __remove_without_replace_failed_bank(self, bank_to_remove):
        self.banks.remove(bank_to_remove)
        self.log.debug("repay", f"{bank_to_remove.get_id()} bankrupted and removed")
        for bank_i in self.banks:
            if bank_i.lender is None or bank_i.lender == bank_to_remove.id:
                bank_i.lender = None
            elif bank_i.lender > bank_to_remove.id:
                bank_i.lender -= 1
            if bank_i.id > bank_to_remove.id:
                bank_i.id -= 1
            for borrower in list(bank_i.active_borrowers):
                if borrower > bank_to_remove.id:
                    bank_i.active_borrowers[borrower-1] = bank_i.active_borrowers[borrower]
                    del bank_i.active_borrowers[borrower]
                elif borrower == bank_to_remove.id:
                    del bank_i.active_borrowers[borrower]

    def initialize_step(self):
        for bank in self.banks:
            bank.B = 0
            bank.rationing = 0
            bank.active_borrowers = {}
        # self.config.lender_change.initialize_step(self)
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
            # leverage
            bank.lambda_ = bank.l / bank.E
            bank.incrD = 0

        max_lambda = max(self.banks, key=lambda k: k.lambda_).lambda_
        for bank in self.banks:
            bank.h = bank.lambda_ / max_lambda if max_lambda > 0 else 0
            bank.A = bank.C + bank.L  # bank.L / bank.λ + bank.D

        # determine c (lending capacity) for all other banks (to whom give loans):
        for bank in self.banks:
            bank.c = []
            for i in range(self.config.N):
                c = 0 if i == bank.id else (1 - self.banks[i].h) * self.banks[i].A
                bank.c.append(c)

        # (equation 6)
        min_r = sys.maxsize
        lines = []

        for bank_i in self.banks:
            line1 = ""
            line2 = ""

            bank_i.asset_i = 0
            bank_i.asset_j = 0

            for j in range(self.config.N):
                try:
                    if j == bank_i.id:
                        bank_i.rij[j] = 0
                    else:
                        if self.banks[j].p == 0 or bank_i.c[j] == 0:
                            bank_i.rij[j] = self.config.r_i0
                        else:
                            bank_i.rij[j] = (self.config.ji * bank_i.A -
                                             self.config.phi * self.banks[j].A -
                                             (1 - self.banks[j].p) *
                                             (self.config.xi * self.banks[j].A - bank_i.c[j])) \
                                            / (self.banks[j].p * bank_i.c[j])

                            bank_i.asset_i += self.config.ji * bank_i.A
                            bank_i.asset_j += self.config.phi * self.banks[j].A
                            bank_i.asset_j += (1 - self.banks[j].p)
                        if bank_i.rij[j] < 0:
                            bank_i.rij[j] = self.config.r_i0
                # the first t=1, maybe t=2, the shocks have not affected enough to use L (only C), so probably
                # L and E are equal for all banks, and so max_lambda=any λ and h=1 , so cij=(1-1)A=0, and r division
                # by zero -> solution then is to use still r_i0:
                except ZeroDivisionError:
                    bank_i.rij[j] = self.config.r_i0

                line1 += f"{bank_i.rij[j]:.3f},"
                line2 += f"{bank_i.c[j]:.3f},"
            lines.append('  |' if lines else "c=|" + line2[:-1] + "| r=|" +
                                             line1[
                                             :-1] + f"| {bank_i.get_id(short=True)} h={bank_i.h:.3f},λ={bank_i.lambda_:.3f} ")
            bank_i.r = np.sum(bank_i.rij) / (self.config.N - 1)
            bank_i.asset_i = bank_i.asset_i / (self.config.N - 1)
            bank_i.asset_j = bank_i.asset_j / (self.config.N - 1)
            if bank_i.r < min_r:
                min_r = bank_i.r

        if self.config.N < 10:
            for line in lines:
                self.log.debug("links", f"{line}")
        self.log.debug("links",
                       f"maxE={maxE:.3f} maxC={maxC:.3f} max_lambda={max_lambda:.3f} min_r={min_r:.3f} ŋ={self.eta:.3f}")

        # (equation 7)
        log_info_1 = log_info_2 = ""
        for bank in self.banks:
            # bank.μ mu
            bank.mu = self.eta * (bank.C / maxC) + (1 - self.eta) * (min_r / bank.r)
            log_info_1 += f"{bank.get_id(short=True)}:{bank.mu:.3f},"
            log_info_2 += f"{bank.get_id(short=True)}:{bank.r:.3f},"
        if self.config.N <= 10:
            self.log.debug("links", f"μ=[{log_info_1[:-1]}] r=[{log_info_2[:-1]}]")

        self.config.lender_change.step_setup_links(self)
        for bank in self.banks:
            self.log.debug("links", self.config.lender_change.change_lender(self, bank, self.t))


# %%

class Bank:
    """
    It represents an individual bank of the network, with the logic of interaction between it and the interbank system
    """

    def get_lender(self):
        if self.lender is None or self.lender >= len(self.model.banks):
            return None
        else:
            return self.model.banks[self.lender]

    def get_loan_interest(self):
        if self.lender is None or self.lender >= len(self.model.banks):
            return None
        else:
            # only we take in account if the bank has a lender active, so the others will return always None
            return self.model.banks[self.lender].rij[self.id]

    def get_id(self, short: bool = False):
        init = "bank#" if not short else "#"
        if self.failures > 0:
            return f"{init}{self.id}.{self.failures}"
        else:
            return f"{init}{self.id}"

    def __init__(self, new_id, bank_model):
        self.id = new_id
        self.model = bank_model
        self.failures = 0
        self.rationing = 0
        self.lender = None
        self.__assign_defaults__()

    def __assign_defaults__(self):
        self.L = self.model.config.L_i0
        self.D = self.model.config.D_i0
        self.E = self.model.config.E_i0
        self.R = self.model.config.reserves * self.D
        self.C = self.D + self.E - self.L - self.R
        self.mu = 0  # fitness of the bank:  estimated later
        self.l = 0  # amount of loan done:  estimated later
        self.s = 0  # amount of loan received: estimated later
        self.d = 0  # amount of demand of loan
        self.B = 0  # bad debt: estimated later
        self.incrD = 0
        self.incrR = 0
        self.failed = False
        # identity of the lender
        self.lender = self.model.config.lender_change.new_lender(self.model, self)
        self.active_borrowers = {}
        self.asset_i = 0
        self.asset_j = 0

    def replace_bank(self):
        self.failures += 1
        self.__assign_defaults__()

    def __do_bankruptcy__(self, phase):
        self.failed = True
        self.model.statistics.bankruptcy[self.model.t] += 1
        recovered_in_fire_sales = self.L * self.model.config.ro  # we fire sale what we have
        recovered = recovered_in_fire_sales - self.D  # we should pay D to clients
        if recovered < 0:
            recovered = 0
        if recovered > self.l:
            recovered = self.l

        bad_debt = self.l - recovered  # the fire sale minus paying D: what the lender recovers
        if bad_debt > 0:
            self.paid_loan = recovered
            self.get_lender().B += bad_debt
            self.get_lender().E -= bad_debt
            self.get_lender().C += recovered
            self.model.log.debug(phase, f"{self.get_id()} bankrupted (fire sale={recovered_in_fire_sales:.3f},"
                                        f"recovers={recovered:.3f},paidD={self.D:.3f})"
                                        f"(lender{self.get_lender().get_id(short=True)}"
                                        f".ΔB={bad_debt:.3f},ΔC={recovered:.3f})")
        else:
            # self.l=0 no current loan to return:
            if self.l > 0:
                self.paid_loan = self.l  # the loan was paid, not the interest
                self.get_lender().C += self.l  # lender not recovers more than loan if it is
                self.model.log.debug(phase, f"{self.get_id()} bankrupted "
                                            f"(lender{self.get_lender().get_id(short=True)}"
                                            f".ΔB=0,ΔC={recovered:.3f}) (paidD={self.l:.3f})")
        self.D = 0
        # the loan is not paid correctly, but we remove it
        if self.get_lender() and self.id in self.get_lender().active_borrowers:
            self.get_lender().s -= self.l
            try:
                del self.get_lender().active_borrowers[self.id]
            except IndexError:
                pass

    def do_fire_sales(self, amount_to_sell, reason, phase):
        cost_of_sell = amount_to_sell / self.model.config.ro
        recovered_E = cost_of_sell * (1 - self.model.config.ro)
        if cost_of_sell > self.L:
            self.model.log.debug(phase,
                                 f"{self.get_id()} impossible fire sale "
                                 f"sell_L={cost_of_sell:.3f} > L={self.L:.3f}: {reason}")
            return self.__do_bankruptcy__(phase)
        else:
            self.L -= cost_of_sell
            self.E -= recovered_E

            if self.L <= self.model.config.alfa:
                self.model.log.debug(phase,
                                     f"{self.get_id()} new L={self.L:.3f} makes bankruptcy of bank: {reason}")
                return self.__do_bankruptcy__(phase)
            else:
                if self.E <= self.model.config.alfa:
                    self.model.log.debug(phase,
                                         f"{self.get_id()} new E={self.E:.3f} makes bankruptcy of bank: {reason}")
                    return self.__do_bankruptcy__(phase)
                else:
                    self.model.log.debug(phase,
                                         f"{self.get_id()} fire sale sellL={amount_to_sell:.3f} "
                                         f"at cost {cost_of_sell:.3f} reducing"
                                         f"E={recovered_E:.3f}: {reason}")
                    return amount_to_sell


class Gui:
    gooey = False

    def progress_bar(self, message, maximum):
        print(message)
        self.maximum = maximum
        self.current = 1

    def next(self):
        self.current += 1
        print("progress: {}%".format(self.current / self.maximum * 100))
        sys.stdout.flush()

    def parser(self):
        try:
            import interbank_gui
            parser = interbank_gui.get_interactive_parser()
        except:
            parser = None
        if parser is None:
            parser = argparse.ArgumentParser()
        else:
            self.gooey = True
        return parser


class Utils:
    """
    Auxiliary class to encapsulate the use of the model
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
        gui = Gui()
        parser = gui.parser()
        parser.add_argument("--debug", type=int, default=None,
                            help="Stop and enter in debug mode after at this time")
        parser.add_argument("--log", default='ERROR', help="Log level messages (ERROR,DEBUG,INFO...)")
        parser.add_argument("--modules", default=None, help=f"Log only this modules (separated by ,)")
        parser.add_argument("--logfile", default=None, help="File to send logs to")
        parser.add_argument("--save", default=None, help=f"Saves the output of this execution")
        parser.add_argument("--graph", default=None,
                            help=f"List of t in which save the network config (* for all)")
        parser.add_argument("--gif_graph", default=False,
                            type=bool,
                            help=f"If --graph, then also an animated gif with all graphs ")
        parser.add_argument("--graph_stats", default=False,
                            type=str, help=f"Load a json of a graph and give us statistics of it")
        parser.add_argument("--n", type=int, default=Config.N, help=f"Number of banks")
        parser.add_argument("--eta", type=float, default=Model.eta, help=f"Policy recommendation")
        parser.add_argument("--t", type=int, default=Config.T, help=f"Time repetitions")
        parser.add_argument("--lc_p", type=float, default=None,
                            help=f"For Erdos-Renyi bank lender's change value of p=0.0x")
        parser.add_argument("--lc_m", type=int, default=None,
                            help=f"For Preferential bank lender's change value of graph grade m")
        parser.add_argument("--lc", type=str, default="default",
                            help="Bank lender's change method (?=list)")
        parser.add_argument("--lc_ini_graph_file", type=str, default=None,
                            help="Load a graph in json networkx.node_link_data() format")
        parser.add_argument("--plot_format", type=str, default="none",
                            help="Generate plots with the specified format (svg,png,pdf,gif,agr)")
        parser.add_argument("--output_format", type=str, default="gdt",
                            help="File extension for data (gdt,txt,csv,both)")
        parser.add_argument("--output", type=str, default=None,
                            help="Directory where to store the results")
        parser.add_argument("--no_replace", action='store_true', default=False,
                            help="No replace banks when they go bankrupted")
        parser.add_argument("--seed", type=int, default=None,
                            help="seed used for random generator")
        args = parser.parse_args()
        if args.graph_stats:
            print(lc.GraphStatistics.describe(args.graph_stats))
            sys.exit(0)
        if args.t != model.config.T:
            model.config.T = args.t
        if args.n != model.config.N:
            model.config.N = args.n
        if args.eta != model.eta:
            model.eta = args.eta
        if args.no_replace:
            model.config.allow_replacement_of_bankrupted = False
        if args.debug:
            model.do_debug(args.debug)
        model.config.lender_change = lc.determine_algorithm(args.lc)
        model.config.lender_change.set_parameter("p", args.lc_p)
        model.config.lender_change.set_parameter("m", args.lc_m)
        model.config.lender_change.set_initial_graph_file(args.lc_ini_graph_file)
        model.log.define_log(args.log, args.logfile, args.modules)
        model.log.define_gui(gui)
        model.statistics.define_output_format(args.output_format)
        model.statistics.set_gif_graph(args.gif_graph)
        model.statistics.define_plot_format(args.plot_format)
        Utils.run(args.save, Utils.__extract_t_values_from_arg__(args.graph),
                  output_directory=args.output, seed=args.seed,
                  interactive=(args.log == 'ERROR' or args.logfile is not None))

    @staticmethod
    def run(save=None, save_graph_instants=None, interactive=False, output_directory=None, seed=None):
        global model
        if not save_graph_instants and Config.GRAPHS_MOMENTS:
            save_graph_instants = Config.GRAPHS_MOMENTS
        model.initialize(export_datafile=save, save_graphs_instants=save_graph_instants,
                         output_directory=output_directory, seed=seed)
        model.simulate_full(interactive=interactive)
        return model.finish()

    # noinspection PyStatementEffect
    @staticmethod
    def is_notebook():
        try:
            # noinspection PyBroadException
            __IPYTHON__
            return get_ipython().__class__.__name__ != "SpyderShell"
        except NameError:
            return False

    @staticmethod
    def is_spyder():
        # noinspection PyBroadException
        try:
            return get_ipython().__class__.__name__ == "SpyderShell"
        except NameError:
            return False


# %%


model = Model()
if Utils.is_notebook():
    model.statistics.OUTPUT_DIRECTORY = '/content'
    model.statistics.output_format = 'csv'
    # if we are running in a Notebook:
    Utils.run(save="results")
else:
    # if we are running interactively:
    if __name__ == "__main__":
        Utils.run_interactive()

# %%
