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
import time
from typing import Any
import networkx as nx
import sys
import os
import matplotlib.pyplot as plt
import numpy

import interbank_lenderchange as lc
import pandas as pd
import numpy as np
import lxml.etree
import lxml.builder
import gzip
import scipy.stats

LENDER_CHANGE_DEFAULT = 'ShockedMarket3'
LENDER_CHANGE_DEFAULT_P = 0.2

class Config:
    """
        Configuration parameters for the interbank network
    """
    T: int = 1000  # time (1000)
    N: int = 50    # number of banks (50)

    reserves: float = 0.02

    # seed applied for random values (set during initialize)
    seed : int = None

    # if False, when a bank fails it's not replaced and N is reduced
    allow_replacement_of_bankrupted = True

    # If true allow_replacement, then:
    #    - reintroduce=False we reintroduce bankrupted banks with initial values
    #    - reintroduce=True we reintroduce with median of current values
    reintroduce_with_median = False

    # if the then a gdt with all the data of time evolution of equity of each bank is generated:
    detailed_equity = False

    # shocks parameters: mi=0.7 omega=0.55 for perfect balance
    mu: float = 0.7  # mi µ
    omega: float = 0.55  # omega ω

    # Lender's change mechanism
    lender_change: lc.LenderChange = None

    # screening costs
    phi: float = 0.025  # phi Φ
    chi: float = 0.015  # ji Χ

    xi: float = 0.3  # xi ξ liquidation cost of collateral
    rho: float = 0.3  # ro ρ fire sale cost

    beta: float = 5  # β beta intensity of breaking the connection (5)
    alfa: float = 0.1  # α alfa below this level of E or D, we will bankrupt the bank

    # If true, psi variable will be ignored:
    psi_endogenous = False
    psi: float = 0.3  # market power parameter : 0 perfect competence .. 1 monopoly

    # If it's a value greater than 0, instead of allowing interest rates for borrowers of any value, we normalize
    # them to range [r_i0 .. normalize_interest_rate_max]:
    normalize_interest_rate_max = 1

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
                           'rationing': True, 'leverage': False, 'systemic_leverage': False,
                           'num_of_rationed': True,
                           'loans': False, 'c': True,
                           'reserves': True, 'deposits': True,
                           'psi': True, 'psi_lenders': True,
                           'potential_lenders':False,
                           'active_lenders': False, 'active_borrowers': False, 'prob_bankruptcy': False,
                           'num_banks': True, 'bankruptcy_rationed': True}


    def __str__(self, separator=''):
        description = sys.argv[0] if __name__ == '__main__' else ''
        for attr, value in self:
            description += ' {}={}{}'.format(attr, value, separator)
        return description + ' '

    def __iter__(self):
        for attr in dir(self):
            value = getattr(self, attr)
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
                yield attr, value

    def get_current_value(self, name_config):
        current_value = None
        try:
            current_value = getattr(self, name_config)
        except AttributeError:
            logging.error("Config has no '{}' parameter".format(name_config))
            sys.exit(-1)
        # if current_value is None, then we cannot guess which type is the previous value:
        if current_value is None:
            try:
                current_value = self.__annotations__[name_config]
            except KeyError:
                return False
            if current_value is int:
                return 0
            elif current_value is bool:
                return False
            else:
                return 0.0
        return current_value


    def define_values_from_args(self, config_list):
        if config_list:
            config_list.sort()
            for item in config_list:
                if item == '?':
                    print(self.__str__(separator='\n'))
                    sys.exit(0)
                elif item.startswith('lc='):
                    # to define the lender change algorithm by command line:
                    self.lender_change = lc.determine_algorithm(item.replace("lc=",''))
                else:
                    try:
                        name_config, value_config = item.split('=')
                    except ValueError:
                        name_config, value_config = ('-', '-')
                        logging.error('A Config value should be passed as parameter=value: {}'.format(item))
                        sys.exit(-1)
                    current_value = self.get_current_value(name_config)
                    try:
                        if isinstance(current_value, bool):
                            if value_config.lower() in ('y', 'yes', 't', 'true', 'on', '1'):
                                setattr(self, name_config, True)
                            elif value_config.lower() in ('n', 'no', 'false', 'f', 'off', '0'):
                                setattr(self, name_config, False)
                        elif isinstance(current_value, int):
                            setattr(self, name_config, int(value_config))
                        elif isinstance(current_value, float):
                            setattr(self, name_config, float(value_config))
                        else:
                            setattr(self, name_config, float(value_config))
                    except ValueError:
                        print(current_value, type(current_value), isinstance(current_value, bool),
                              isinstance(current_value, int))
                        logging.error('Value given for {} is not valid: {}'.format(name_config, value_config))
                        sys.exit(-1)

class Statistics:
    lender_no_d = []
    no_lender = []
    enough_money = []
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
    equity_borrowers = []
    P = []
    B = []
    c = []
    P_max = []
    P_min = []
    P_std = []
    active_borrowers = []
    active_lenders = []
    potential_lenders = []
    prob_bankruptcy = []
    num_of_rationed = []
    psi = []
    psi_lenders = []
    num_banks = []
    bankruptcy_rationed = []
    model = None
    graphs = {}
    graphs_pos = None
    plot_format = None
    graph_format = '.svg'
    output_format = '.gdt'
    create_gif = False
    OUTPUT_DIRECTORY = 'output'
    NUMBER_OF_ITEMS_IN_ANIMATED_GRAPH = 40
    # cross correlation of interest rate against bankruptcies
    correlation = []

    def __init__(self, in_model):
        self.detailed_equity = None
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
        self.potential_lenders = np.zeros(self.model.config.T, dtype=int)
        self.fitness = np.zeros(self.model.config.T, dtype=float)
        self.c = np.zeros(self.model.config.T, dtype=float)
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
        self.num_of_rationed = np.zeros(self.model.config.T, dtype=int)
        self.psi = np.zeros(self.model.config.T, dtype=float)
        self.psi_lenders = np.zeros(self.model.config.T, dtype=float)

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
        credit_channels = self.model.config.lender_change.get_credit_channels()
        if credit_channels is None:
            self.potential_credit_channels[self.model.t] = len(self.model.banks)
        else:
            self.potential_credit_channels[self.model.t] = credit_channels

    def compute_potential_lenders(self):
        for bank in self.model.banks:
            if bank.incrD>0:
                self.potential_lenders[self.model.t] += 1


    def compute_interest_rates_and_loans(self):
        interests_rates_of_borrowers = []
        psi_of_lenders = []
        asset_i = []
        asset_j = []
        sum_of_loans = 0
        num_of_banks_that_are_lenders = 0
        num_of_banks_that_are_borrowers = 0
        for bank in self.model.banks:            
            if bank.incrD >= 0:
                if bank.active_borrowers:
                    asset_i.append(bank.asset_i)
            elif bank.d > 0:
                asset_j.append(bank.asset_j)
            if bank.active_borrowers:
                num_of_banks_that_are_lenders += 1
                for bank_that_is_borrower in bank.active_borrowers:
                    sum_of_loans += bank.active_borrowers[bank_that_is_borrower]
                psi_of_lenders.append(bank.psi)
            elif bank.l > 0:
                num_of_banks_that_are_borrowers += 1
                interests_rates_of_borrowers.append(bank.get_loan_interest())
            bank.has_a_loan = bank.get_loan_interest() is not None and bank.l > 0

        self.interest_rate[self.model.t] = np.mean(interests_rates_of_borrowers) \
            if interests_rates_of_borrowers else 0.0
        self.asset_i[self.model.t] = np.mean(asset_i) if asset_i else np.nan
        self.asset_j[self.model.t] = np.mean(asset_j) if asset_j else np.nan
        if self.model.config.psi_endogenous:
            self.psi_lenders[self.model.t] = np.mean(psi_of_lenders) if psi_of_lenders else 0.0
            self.psi[self.model.t] = sum(bank.psi for bank in self.model.banks) / len(self.model.banks)
        else:
            self.psi_lenders[self.model.t] = self.model.config.psi
            self.psi[self.model.t] = self.model.config.psi
        self.loans[self.model.t] = sum_of_loans / num_of_banks_that_are_lenders \
            if num_of_banks_that_are_lenders else np.nan
        self.active_lenders[self.model.t] = num_of_banks_that_are_lenders
        self.active_borrowers[self.model.t] = num_of_banks_that_are_borrowers

    def compute_leverage_and_equity(self):
        sum_of_equity = 0
        sum_of_equity_borrowers = 0
        leverage_of_lenders = []
        self.model.statistics.save_detailed_equity(self.model.t)
        for bank in self.model.banks:
            if not bank.failed:
                sum_of_equity += bank.E
                if bank.l == 0:
                    amount_of_loan = 0
                    if bank.get_lender() is not None and bank.get_lender().l > 0:
                        amount_of_loan = bank.get_lender().l
                    leverage_of_lenders.append(amount_of_loan/ bank.E)
                if bank.active_borrowers:
                    for borrower in bank.active_borrowers:
                        sum_of_equity_borrowers += self.model.banks[borrower].E
                self.model.statistics.save_detailed_equity(bank.E)
            else:
                self.model.statistics.save_detailed_equity('')
        self.model.statistics.save_detailed_equity('\n')
        self.equity[self.model.t] = sum_of_equity
        self.equity_borrowers[self.model.t] = sum_of_equity_borrowers
        self.leverage[self.model.t] = np.mean(leverage_of_lenders) if leverage_of_lenders else 0.0
        # systemic_leverage = how the system is in relation to the total population of banks (big value  of 10 borrowers
        # against a population of 100 banks means that there is a risk
        self.systemic_leverage[self.model.t] = sum(leverage_of_lenders) / len(self.model.banks) \
            if len(self.model.banks) > 0 else 0

    def compute_liquidity(self):
        total_liquidity = 0
        for bank in self.model.banks:
            if not bank.failed:
                total_liquidity += bank.C
        self.liquidity[self.model.t] = total_liquidity

    def compute_fitness(self):
        self.fitness[self.model.t] = np.nan
        if self.model.config.N > 0:
            total_fitness = 0
            num_items = 0
            for bank in self.model.banks:
                if not bank.failed:
                    total_fitness += bank.mu
                    num_items += 1
            self.fitness[self.model.t] = total_fitness / num_items if num_items > 0 else np.nan

    def compute_policy(self):
        self.policy[self.model.t] = self.model.eta

    def compute_bad_debt(self):
        self.B[self.model.t] = sum((bank.B for bank in self.model.banks))

    def compute_rationing(self):
        self.rationing[self.model.t] = sum((bank.rationing for bank in self.model.banks))

    def compute_deposits_and_reserves(self):
        total_deposits = 0
        total_reserves = 0
        for bank in self.model.banks:
            if not bank.failed:
                total_deposits += bank.D
                total_reserves += bank.R
        self.deposits[self.model.t] = total_deposits
        self.reserves[self.model.t] = total_reserves

    def compute_probability_of_lender_change_num_banks_prob_bankruptcy(self):
        self.model.maxE = 0
        for bank in self.model.banks:
            if bank.E > self.model.maxE:
                self.model.maxE = bank.E
        avg_prob_bankruptcy = []
        if self.model.maxE > 0:
            for bank in self.model.banks:
                if bank.has_a_loan:
                    avg_prob_bankruptcy.append(1 - bank.E / self.model.maxE)
        self.prob_bankruptcy[self.model.t] = np.mean(avg_prob_bankruptcy) if avg_prob_bankruptcy else np.nan

        probabilities = []
        lender_capacities = []
        num_banks = 0
        for bank in self.model.banks:
            if not bank.failed:
                probabilities.append(bank.P)
                lender_capacities.append(np.mean(bank.c))
                num_banks += 1
        self.P[self.model.t] = sum(probabilities) / len(probabilities) if len(probabilities) > 0 else np.nan
        self.P_max[self.model.t] = max(probabilities) if probabilities else np.nan
        self.c[self.model.t] = np.mean(lender_capacities) if lender_capacities else np.nan
        self.P_min[self.model.t] = min(probabilities) if probabilities else np.nan
        self.P_std[self.model.t] = np.std(probabilities) if probabilities else np.nan
        self.num_banks[self.model.t] = num_banks

    def get_cross_correlation_result(self, t):
        if t in [0,1] and len(self.correlation)>t:
            status = '  '
            if self.correlation[t][0]>0:
                if self.correlation[t][1]<0.05:
                    status = '**'
                elif self.correlation[t][1]<0.10:
                    status = '* '
            return (f'correl t={t} int_rate/bankrupt {self.correlation[t][0]:4.2} '
                    f'p_value={self.correlation[t][1]:4.2} {status}')
        else:
            return " "

    def determine_cross_correlation(self):
        if np.all(self.bankruptcy == 0) or np.all(self.bankruptcy == self.bankruptcy[0]) or \
            np.all(self.interest_rate == 0) or np.all(self.interest_rate == self.interest_rate[0]):
            self.correlation = []
        else:
            try:
                self.correlation = [
                    # correlation_coefficient = [-1..1] and p_value < 0.10
                    scipy.stats.pearsonr(self.interest_rate,self.bankruptcy),
                    # time delay 1:
                    scipy.stats.pearsonr(self.interest_rate[1:],self.bankruptcy[:-1])
                    ]
            except ValueError:
                self.correlation = []

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
            lc.draw(self.graphs[t], new_guru_look_for=True, title='t={}'.format(t))
            if Utils.is_spyder():
                plt.show()
                filename = None
            else:
                filename = sys.argv[0] if self.model.export_datafile is None else self.model.export_datafile
                filename = self.get_export_path(filename, '_{}{}'.format(t, self.graph_format))
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
                print('Invalid plot file format: {}'.format(plot_format))
                sys.exit(-1)

    def define_output_format(self, output_format):
        match output_format.lower():
            case 'both':
                self.output_format = '.both'
            case 'gdt':
                self.output_format = '.gdt'
            case 'csv':
                self.output_format = '.csv'
            case 'txt':
                self.output_format = '.txt'
            case _:
                print('Invalid output file format: {}'.format(output_format))
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
                if not idx % positions_of_images == 0:
                    continue
                images.append(Image.open(image_file))
            images[0].save(fp=filename_output, format='GIF',
                           append_images=images[1:], save_all=True, duration=100, loop=0)

    def get_export_path(self, filename, ending_name=''):
        if not os.path.dirname(filename):
            filename = '{}/{}'.format(self.OUTPUT_DIRECTORY, filename)
        path, extension = os.path.splitext(filename)
        if ending_name:
            return path + ending_name
        else:
            return path + self.output_format.lower()

    def __generate_csv_or_txt(self, export_datafile, header, delimiter):
        with open(export_datafile, 'w', encoding='utf-8') as save_file:
            for line_header in header:
                save_file.write('# {}\n'.format(line_header))
            save_file.write("# pd.read_csv('file{}',header={}', delimiter='{}')\nt".format(self.output_format,
                                                                                           len(header) + 1, delimiter))
            for element_name, _ in self.enumerate_statistics_results():
                save_file.write('{}{}'.format(delimiter, element_name))
            save_file.write('\n')
            for i in range(self.model.config.T):
                save_file.write('{}'.format(i))
                for _, element in self.enumerate_statistics_results():
                    save_file.write('{}{}'.format(delimiter, element[i]))
                save_file.write('\n'.format())

    def __generate_gdt_file(self, filename, enumerate_results, header):
        element = lxml.builder.ElementMaker()
        gretl_data = element.gretldata
        xml_description = element.description
        xml_variables = element.variables
        variable = element.variable
        xml_observations = element.observations
        observation = element.obs
        variables = xml_variables(count='{}'.format(sum((1 for _ in enumerate_results()))))
        header_text = ''
        for item in header:
            header_text += item + ' '
        # header_text will be present as label in the first variable
        # correlation_result will be present as label in the second variable
        i = 1
        for variable_name, _ in enumerate_results():
            if variable_name == 'leverage':
                variable_name += '_'
            if i==1:
                variables.append(variable(name='{}'.format(variable_name), label='{}'.format(header_text)))
            elif i in [2,3]:
                variables.append(variable(name='{}'.format(variable_name),
                        label=self.get_cross_correlation_result(i-2)))
            else:
                variables.append(variable(name='{}'.format(variable_name)))
            i = i+1
        xml_observations = xml_observations(count='{}'.format(self.model.config.T), labels='false')
        for i in range(self.model.config.T):
            string_obs = ''
            for _, variable in enumerate_results():
                string_obs += '{}  '.format(variable[i])
            xml_observations.append(observation(string_obs))
        gdt_result = gretl_data(xml_description(header_text), variables,
                                xml_observations, version='1.4', name='interbank',
                                frequency='special:1', startobs='1', endobs='{}'.format(self.model.config.T),
                                type='time-series')
        with gzip.open(filename, 'w') as output_file:
            output_file.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
            output_file.write(lxml.etree.tostring(gdt_result, pretty_print=True, encoding=str).encode('ascii'))

    def __generate_gdt(self, export_datafile, header):
        self.__generate_gdt_file(export_datafile, self.enumerate_statistics_results, header)

    @staticmethod
    def __transform_line_from_string(line_with_values):
        items = []
        for i in line_with_values.replace('  ', ' ').strip().split(' '):
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
                header = ['{}'.format(export_description)]
            else:
                header = ['{} T={} N={}'.format(__name__, self.model.config.T, self.model.config.N)]
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

    def enumerate_statistics_results(self):
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
        for variable_name, variable in self.enumerate_statistics_results():
            result[variable_name] = np.array(variable)
        return result.iloc[0:self.model.t]

    def plot_pygrace(self, xx, yy_s, variable, title, export_datafile, x_label, y_label):
        import pygrace.project
        plot = pygrace.project.Project()
        graph = plot.add_graph()
        graph.title.text = title.capitalize().replace('_', ' ')
        for yy, color, ticks, title_y in yy_s:
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
                plot.saveall(self.get_export_path(export_datafile, '_{}{}'.format(variable.lower(), self.plot_format)))

    def plot_pyplot(self, xx, yy_s, variable, title, export_datafile, x_label, y_label):
        if self.plot_format == '.agr':
            self.plot_pygrace(xx, yy_s, variable, title, export_datafile, x_label, y_label)
        else:
            plt.clf()
            plt.figure(figsize=(14, 6))
            for yy, color, ticks, title_y in yy_s:
                if isinstance(yy, tuple):
                    plt.plot(yy[0], yy[1], ticks, color=color, label=title_y, linewidth=0.2)
                else:
                    plt.plot(xx, yy, ticks, color=color, label=title_y)
            plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)
            plt.title(title.capitalize().replace('_', ' '))
            if len(yy_s) > 1:
                plt.legend()
            if export_datafile:
                if self.plot_format:
                    plt.savefig(self.get_export_path(export_datafile,
                                                     '_{}{}'.format(variable.lower(), self.plot_format)))
            else:
                plt.show()
            plt.close()

    def plot_result(self, variable, title, export_datafile=None):
        xx = []
        yy = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(getattr(self, variable)[i])
        self.plot_pyplot(xx, [(yy, 'blue', '-', '')], variable, title, export_datafile, 'Time', '')

    def get_plots(self, export_datafile):
        for variable in Config.ELEMENTS_STATISTICS:
            if Config.ELEMENTS_STATISTICS[variable]:
                if 'plot_{}'.format(variable) in dir(Statistics):
                    eval('self.plot_{}(export_datafile)'.format(variable))
                else:
                    self.plot_result(variable, self.get_name(variable), export_datafile)

    def plot_p(self, export_datafile=None):
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
                              (yy_min, 'cyan', ':', 'Max and min prob'), (yy_max, 'cyan', ':', ''),
                              (yy_std, 'red', '-', 'Std')], 'prob_change_lender',
                         'Prob of change lender ' + self.model.config.lender_change.describe(),
                         export_datafile, 'Time', '')

    def plot_num_banks(self, export_datafile=None):
        if not self.model.config.allow_replacement_of_bankrupted:
            self.plot_result('num_banks', self.get_name('num_banks'), export_datafile)

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
        self.plot_pyplot(xx, [(yy, 'blue', '-', 'id'), (yy2, 'red', '-', 'Num clients'),
                              ((xx3, yy3), 'orange', '-', '')], 'best_lender',
                         'Best Lender (blue) #clients (red)', export_datafile,
                         'Time (best lender={} at t=[{}..{}])'.format(
                             final_best_lender, time_init, time_init + max_duration), 'Best Lender')

    def enable_detailed_equity(self):
        self.define_detailed_equity(self.model.export_datafile)

    def define_detailed_equity(self, detailed_equity, save_file=None):
        if detailed_equity:
            save_file = save_file + '_equity.csv' if save_file else 'equity.csv'
            self.detailed_equity = open(self.get_export_path(save_file).replace('.gdt', '.csv'), 'w')
            self.save_detailed_equity('t')
            for i in range(self.model.config.N):
                self.save_detailed_equity('bank_{}'.format(i))
            self.save_detailed_equity('\n')

    def save_detailed_equity(self, value):
        if self.detailed_equity:
            if isinstance(value, int):
                value = '{};'.format(value)
            elif isinstance(value, float):
                value = '{};'.format(value)
            elif value != '\n':
                value = '{};'.format(value)
            self.detailed_equity.write(value)

class Log:
    """
    The class acts as a logger and helpers to represent the data and evol from the Model.
    """
    logger = logging.getLogger('model')
    modules = []
    model = None
    logLevel = 'ERROR'
    progress_bar = None

    def __init__(self, its_model):
        self.model = its_model

    def do_progress_bar(self, message, maximum):
        from progress.bar import Bar
        self.progress_bar = Bar(message, max=maximum)

    @staticmethod
    def __format_number__(number):
        result = '{}'.format(number)
        while len(result) > 5 and result[-1] == '0':
            result = result[:-1]
        while len(result) > 5 and result.find('.') > 0:
            result = result[:-1]
        return result

    def debug_banks(self, details: bool=True, info: str=''):
        for bank in self.model.banks:
            if not info:
                info = '-----'
            self.info(info, bank.__str__(details=details))

    @staticmethod
    def get_level(option):
        try:
            return getattr(logging, option.upper())
        except AttributeError:
            logging.error(" '--log' must contain a valid logging level and {} is not.".format(option.upper()))
            sys.exit(-1)

    def debug(self, module, text):
        if self.modules == [] or module in self.modules:
            if text:
                self.logger.debug('t={}/{} {}'.format(self.model.t, module, text))

    def info(self, module, text):
        if self.modules == [] or module in self.modules:
            if text:
                self.logger.info(' t={}/{} {}'.format(self.model.t, module, text))

    def error(self, module, text):
        if text:
            self.logger.error('t={}/{} {}'.format(self.model.t, module, text))

    def define_log(self, log: str, logfile: str='', modules: str=''):
        self.modules = modules.split(',') if modules else []
        formatter = logging.Formatter('%(levelname)s-' + '- %(message)s')
        self.logLevel = Log.get_level(log.upper())
        self.logger.setLevel(self.logLevel)
        if logfile:
            if not os.path.dirname(logfile):
                logfile = '{}/{}'.format(self.model.statistics.OUTPUT_DIRECTORY, logfile)
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
    banks = []
    t: int = 0
    eta: float = 1
    test = False
    default_seed: int = 20579
    backward_enabled = False
    policy_changes = 0
    save_graphs = None
    save_graphs_results = []
    log = None
    maxE = Config.E_i0
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
        self.value_for_reintroduced_banks_L = self.config.L_i0
        self.value_for_reintroduced_banks_E = self.config.E_i0
        self.value_for_reintroduced_banks_D = self.config.D_i0
        if configuration:
            self.configure(**configuration)
        if self.backward_enabled:
            self.banks_backward_copy = []

    def configure_json(self, json_string: str):
        import re, json
        json_string = (json_string.strip().
                       replace('=', ':').replace(' ', ', ').
                       replace('True', 'true').replace('False', 'false'))
        if not json_string.startswith('{'):
            json_string = '{' + json_string
        if not json_string.endswith('}'):
            json_string += '}'
        self.configure(**json.loads(re.sub('(?<=\\{|\\s)(\\w+)(?=\\s*:)', '"\\1"', json_string)))

    def configure(self, **configuration):
        for attribute in configuration:
            if attribute.startswith('lc'):
                attribute = attribute.replace('lc_', '')
                if attribute == 'lc':
                    self.config.lender_change = lc.determine_algorithm(configuration[attribute])
                else:
                    self.config.lender_change.set_parameter(attribute, configuration['lc_' + attribute])
            elif hasattr(self.config, attribute):
                setattr(self.config, attribute, configuration[attribute])
            else:
                raise LookupError('attribute in config not found: %s ' % attribute)

    def initialize(self, seed=None, dont_seed=False, save_graphs_instants=None, export_datafile=None,
                   export_description=None, generate_plots=True, output_directory=None):
        self.statistics.reset(output_directory=output_directory)
        if not seed is None and not dont_seed:
            applied_seed = seed if seed else self.default_seed
            random.seed(applied_seed)
            self.config.seed = applied_seed
        self.save_graphs = save_graphs_instants
        self.banks = []
        self.t = 0
        if not self.config.lender_change:
            self.config.lender_change = lc.determine_algorithm()
            self.config.lender_change.set_parameter('p', 0.5)
        self.policy_changes = 0
        if export_datafile:
            self.export_datafile = export_datafile
        if generate_plots:
            self.generate_plots = generate_plots
        if export_description is None:
            self.export_description = str(self.config) + str(self.config.lender_change)
        else:
            self.export_description = export_description
        for i in range(self.config.N):
            self.banks.append(Bank(i, self))
        self.config.lender_change.initialize_bank_relationships(self)

    def forward(self):
        self.initialize_step()
        if self.backward_enabled:
            self.banks_backward_copy = copy.deepcopy(self.banks)
        self.do_shock('shock1')
        self.statistics.compute_potential_lenders()
        self.do_loans()
        self.log.debug_banks()
        self.statistics.compute_interest_rates_and_loans()
        self.statistics.compute_leverage_and_equity()
        self.do_shock('shock2')
        self.do_repayments()
        self.log.debug_banks()
        if self.log.progress_bar:
            self.log.progress_bar.next()
        self.statistics.compute_liquidity()
        self.statistics.compute_credit_channels_and_best_lender()
        self.statistics.compute_fitness()
        self.statistics.compute_policy()
        self.statistics.compute_deposits_and_reserves()
        self.statistics.bankruptcy_rationed[self.t] = self.replace_bankrupted_banks()
        self.setup_links()
        self.statistics.compute_probability_of_lender_change_num_banks_prob_bankruptcy()
        self.log.debug_banks()
        if self.save_graphs is not None and (self.save_graphs == '*' or self.t in self.save_graphs):
            filename = self.statistics.get_graph(self.t)
            if filename:
                self.save_graphs_results.append(filename)
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

    def enable_backward(self):
        self.backward_enabled = True

    def simulate_full(self, interactive=False):
        if interactive:
            self.log.do_progress_bar('Simulating t=0..{}'.format(self.config.T), self.config.T)
        for t in range(self.config.T):
            self.forward()
            if not self.config.allow_replacement_of_bankrupted and len(self.banks) <= 2:
                self.config.T = self.t
                self.log.debug('*****', 'Finish because there are only two banks surviving'.format())
                break


    def finish(self):
        if not self.test:
            self.statistics.determine_cross_correlation()
            self.statistics.export_data(export_datafile=self.export_datafile,
                                        export_description=self.export_description,
                                        generate_plots=self.generate_plots)
        summary = 'Finish: model T={}  N={}'.format(self.config.T, self.config.N)
        if not self.__policy_recommendation_changed__():
            summary += ' ŋ={}'.format(self.eta)
        else:
            summary += ' ŋ variate during simulation'
        self.log.info('*****', summary)
        self.statistics.create_gif_with_graphs(self.save_graphs_results)
        plt.close()
        return self.statistics.get_data()

    def set_policy_recommendation(self, n: int=None, eta: float=None, eta_1: float=None):
        if eta_1 is not None:
            n = round(eta_1)
        if n is not None and eta is None:
            if type(n) is int:
                eta = self.policy_actions_translation[n]
            else:
                eta = float(n)
        if self.eta != eta:
            self.log.debug('*****', 'eta(ŋ) changed to {}'.format(eta))
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
        min_c = 1000000.0
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
        rand_value = random.random()
        return bank.D * (self.config.mu + self.config.omega * rand_value)

    def do_shock(self, which_shock):
        for bank in self.banks:
            bank.newD = self.determine_shock_value(bank, which_shock)
            bank.incrD = bank.newD - bank.D
            bank.D = bank.newD
            bank.newR = self.config.reserves * bank.D
            bank.incrR = bank.newR - bank.R
            bank.R = bank.newR
            if bank.incrD >= 0:
                bank.C += bank.incrD - bank.incrR
                if which_shock == 'shock1':
                    bank.s = bank.C
                bank.d = 0
                if bank.incrD > 0:
                    self.log.debug(which_shock, '{} wins ΔD={}'.format(bank.get_id(), bank.incrD))
                else:
                    self.log.debug(which_shock, '{} has no shock'.format(bank.get_id()))
            else:
                if which_shock == 'shock1':
                    bank.s = 0
                if bank.incrD - bank.incrR + bank.C >= 0:
                    bank.d = 0
                    bank.C += bank.incrD - bank.incrR
                    self.log.debug(which_shock, '{} loses ΔD={}, covered by capital'.format(bank.get_id(), bank.incrD))
                else:
                    bank.d = abs(bank.incrD - bank.incrR + bank.C)
                    self.log.debug(which_shock, '{} loses ΔD={} has C={} and needs {}'.format(
                        bank.get_id(), bank.incrD, bank.C, bank.d))
                    if which_shock == 'shock2':
                        bank.do_fire_sales(bank.d, 'ΔD={},C=0 and we need {}'.format(
                            bank.incrD, bank.d), which_shock)
                    else:
                        bank.C = 0
            self.statistics.incrementD[self.t] += bank.incrD

    def do_loans(self):
        self.config.lender_change.extra_relationships_change(self)

        # first we normalize the banks interest rate if it is necessary:
        if self.config.normalize_interest_rate_max and self.config.normalize_interest_rate_max>0:
            max_r = 0
            min_r = numpy.inf
            for bank in self.banks:
                if not bank.get_loan_interest() is None:
                    if bank.get_loan_interest() > max_r:
                        max_r = bank.get_loan_interest()
                    if min_r > bank.get_loan_interest():
                        min_r = bank.get_loan_interest()
            for bank in self.banks:
                if not bank.lender is None:
                    if max_r > self.config.r_i0 and max_r != min_r:
                        # normalize in range [a,b], where a = self.config.r_i0
                        #                                 b =  self.config.normalize_interest_rate_max
                        # x = a + \frac{(x - x_{\min})}{x_{\max} - x_{\min}} (b - a)
                        self.banks[bank.lender].rij[bank.id] = \
                            self.config.r_i0 +( self.banks[bank.lender].rij[bank.id]  - min_r) / (max_r - min_r) * \
                            ( self.config.normalize_interest_rate_max - self.config.r_i0 )
                    else:
                        # if max_r = min_r or max_r< self.config.r_i0 we have maybe only one bank lending, so
                        # max interest rate:
                        self.banks[bank.lender].rij[bank.id] = self.config.normalize_interest_rate_max


        num_of_rationed = 0
        total_rationed = 0
        total_demanded = 0
        total_loans = 0
        for bank_index, bank in enumerate(self.banks):
            rationing_of_bank = 0
            lender = bank.get_lender()
            demand = bank.d
            if demand > 0:
                total_demanded += demand
                if lender is None or lender.d > 0:
                    bank.l = 0
                    rationing_of_bank = demand
                    total_rationed += rationing_of_bank
                    num_of_rationed += 1
                    bank.do_fire_sales(rationing_of_bank, 
                        f'rationing={{rationing_of_bank}} as no lender for this bank' if lender is None 
                        else f'rationing={{rationing_of_bank}} as lender {{lender.get_id(short=True)}} has no money', 
                        'loans')
                elif demand > lender.s:
                    rationing_of_bank = demand - lender.s
                    total_rationed += rationing_of_bank
                    num_of_rationed += 1
                    bank.do_fire_sales(rationing_of_bank,
                        f'lender.s={{lender.s}} but need d={{demand}}, rationing={{rationing_of_bank}}', 
                        'loans')
                    loan = lender.s if lender.s > 0 else 0
                    bank.l = loan
                    if loan > 0:
                        lender.active_borrowers[bank_index] = loan
                        lender.C -= loan
                        lender.s = 0
                    total_loans += loan
                else:
                    bank.l = demand
                    lender.active_borrowers[bank_index] = demand
                    lender.C -= demand
                    lender.s -= demand
                    total_loans += demand
            else:
                bank.l = 0
                if bank.active_borrowers:
                    pass  # can skip string building/logging
            bank.rationing = rationing_of_bank
        self.statistics.num_of_rationed[self.t] = num_of_rationed
        self.statistics.rationing[self.t] = total_rationed


    def do_repayments(self):
        for bank in self.banks:
            if bank.l > 0 and bank.d > 0 and (not bank.failed):
                amount_we_need = bank.l + bank.d - bank.C
                if amount_we_need > 0:
                    obtained = bank.do_fire_sales(amount_we_need, 'fire sales due to not enough C'.format(), 'repay')
                    bank.d -= obtained
                    if bank.d < 0:
                        bank.l += bank.d
                        bank.d = 0
                        if bank.l < 0 and (not bank.failed):
                            bank.do_bankruptcy('repay')
                bank.reviewed = True
            else:
                bank.reviewed = False
        for bank in self.banks:
            if bank.l > 0 and (not bank.reviewed) and (not bank.failed):
                loan_profits = bank.get_loan_interest() * bank.l
                loan_to_return = bank.l + loan_profits
                bank_lender = bank.get_lender()
                if loan_to_return > bank.C:
                    lack_of_capital_to_return_loan = loan_to_return - bank.C
                    bank.C = 0
                    obtained_in_fire_sales = bank.do_fire_sales(lack_of_capital_to_return_loan,
                            'to return loan and interest {} > C={}'.format(loan_to_return, bank.C), 'repay')
                    gap_of_money_not_covered_of_loan = lack_of_capital_to_return_loan - obtained_in_fire_sales
                    if gap_of_money_not_covered_of_loan > loan_profits:
                        bank.paid_profits = 0
                        bank.paid_loan = bank.l - gap_of_money_not_covered_of_loan + loan_profits
                    else:
                        bank.paid_loan = bank.l
                        bank.paid_profits = loan_profits - gap_of_money_not_covered_of_loan
                    if not bank_lender.failed and (not bank.failed):
                        bank_lender.C += bank.paid_loan
                        bank_lender.E += bank.paid_profits
                        del bank_lender.active_borrowers[bank.id]
                else:
                    bank.C -= loan_to_return
                    bank.paid_loan = bank.l
                    bank.paid_profits = loan_profits
                    bank_lender.C += bank.paid_loan
                    bank_lender.E += bank.paid_profits
                    del bank_lender.active_borrowers[bank.id]
                bank_lender.s += bank.paid_loan
                bank.E -= loan_profits
                if bank.E < 0:
                    bank.failed = True
                    self.log.debug('repay',
                                   '{} fails as paying interest of the loan generates E<0'.format(bank.get_id()))
        for bank in self.banks:
            if bank.l == 0 and (not bank.failed):
                if bank.C < bank.incrD:
                    bank.d = bank.incrD - bank.C
                else:
                    bank.d = 0
                if bank.d > 0:
                    bank.do_fire_sales(bank.d, 'fire sales due to not enough C'.format(), 'repay')

    def replace_bankrupted_banks(self):
        self.estimate_average_values_for_replacement_of_banks()
        self.statistics.compute_bad_debt()
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
        self.log.debug('repay', 'this step ΔD={} and '.format(
            self.statistics.incrementD[self.t]) + 'failures={}'.format(total_removed))
        if not self.config.allow_replacement_of_bankrupted:
            for bank_to_remove in lists_to_remove_because_replacement_of_bankrupted_is_disabled:
                self.__remove_without_replace_failed_bank(bank_to_remove)
            self.config.N -= len(lists_to_remove_because_replacement_of_bankrupted_is_disabled)
            self.log.debug('repay', 'now we have {} banks'.format(self.config.N))
        return num_banks_failed_rationed

    def __remove_without_replace_failed_bank(self, bank_to_remove):
        self.banks.remove(bank_to_remove)
        self.log.debug('repay', '{} bankrupted and removed'.format(bank_to_remove.get_id()))
        for bank_i in self.banks:
            if bank_i.lender is None or bank_i.lender == bank_to_remove.id:
                bank_i.lender = None
            elif bank_i.lender > bank_to_remove.id:
                bank_i.lender -= 1
            if bank_i.id > bank_to_remove.id:
                bank_i.id -= 1
            for borrower in list(bank_i.active_borrowers):
                if borrower > bank_to_remove.id:
                    bank_i.active_borrowers[borrower - 1] = bank_i.active_borrowers[borrower]
                    del bank_i.active_borrowers[borrower]
                elif borrower == bank_to_remove.id:
                    del bank_i.active_borrowers[borrower]

    def initialize_step(self):
        for bank in self.banks:
            bank.B = 0
            bank.rationing = 0
            bank.paid_profits = 0
            bank.paid_loan = 0
            bank.active_borrowers = {}
        if self.t == 0:
            self.log.debug_banks()


    def setup_links(self):
        if len(self.banks) <= 1:
            return
        self.maxE = max(self.banks, key=lambda k: k.E).E
        max_c = max(self.banks, key=lambda k: k.C).C
        for bank in self.banks:
            bank.p = bank.E / self.maxE
            if bank.get_lender() is not None and bank.get_lender().l > 0:
                bank.lambda_ = bank.get_lender().l / bank.E
            else:
                bank.lambda_ = 0
            # bank.lambda_ = bank.l / bank.E
            bank.incrD = 0
        max_lambda = max(self.banks, key=lambda k: k.lambda_).lambda_
        for bank in self.banks:
            bank.h = bank.lambda_ / max_lambda if max_lambda > 0 else 0
            bank.A = bank.C + bank.L + bank.R
        for bank in self.banks:
            bank.c = []
            for i in range(self.config.N):
                c = 0 if i == bank.id else (1 - self.banks[i].h) * self.banks[i].A
                bank.c.append(c)
            if self.config.psi_endogenous:
                bank.psi = bank.E / self.maxE
        # optimized 2:
        N = self.config.N
        r_i0 = self.config.r_i0
        chi = self.config.chi
        phi = self.config.phi
        xi = self.config.xi
        psi_global = self.config.psi
        psi_endogenous = self.config.psi_endogenous

        A = np.array([bank.A for bank in self.banks])
        p = np.array([bank.p for bank in self.banks])
        psi_array = np.array([bank.psi for bank in self.banks]) if psi_endogenous else np.full(N, psi_global)
        c = np.array([bank.c for bank in self.banks])  # Matriz NxN

        # Ajustar psi cercano a 1 para evitar división por cero
        psi_array = np.where(psi_array == 1, 0.99999999999999, psi_array)

        # Máscaras para condiciones
        mask_diag = np.eye(N, dtype=bool)
        mask_invalid = (p == 0)

        # Broadcasting para cálculo matricial
        psi_matrix = np.repeat(psi_array[:, np.newaxis], N, axis=1)
        denom = p * c * (1 - psi_matrix)
        num = (chi * A[:, np.newaxis] - phi * A[np.newaxis, :] - (1 - p[np.newaxis, :]) * (xi * A[np.newaxis, :] - c))

        # Inicializar rij con r_i0 excepto diagonal cero
        rij = np.where(mask_diag, 0, r_i0)
        valid_mask = (~mask_diag) & (~mask_invalid[np.newaxis, :]) & (c != 0) & (denom != 0)
        rij[valid_mask] = num[valid_mask] / denom[valid_mask]
        rij[rij < 0] = r_i0

        # Asignar rij a cada banco
        for i, bank in enumerate(self.banks):
            bank.rij = rij[i]

        # Calcular r, asset_i y asset_j
        asset_i = A.copy()  # asumido igual que en el código original
        asset_j = np.sum(A) - A
        for i, bank in enumerate(self.banks):
            bank.r = np.sum(bank.rij) / (N - 1)
            bank.asset_i = asset_i[i]
            bank.asset_j = asset_j[i] / (N - 1)

        min_r = np.min([bank.r for bank in self.banks])

        # min_r = sys.maxsize
        # for bank_i in self.banks:
        #     bank_i.asset_i = 0
        #     bank_i.asset_j = 0
        #     for j in range(self.config.N):
        #         try:
        #             if j == bank_i.id:
        #                 bank_i.rij[j] = 0
        #             else:
        #                 if self.banks[j].p == 0 or bank_i.c[j] == 0:
        #                     bank_i.rij[j] = self.config.r_i0
        #                 else:
        #                     psi = bank_i.psi if self.config.psi_endogenous else self.config.psi
        #                     if psi==1:
        #                         psi=0.99999999999999
        #                     bank_i.rij[j] = ((self.config.chi * bank_i.A - self.config.phi * self.banks[j].A
        #                                       - (1 - self.banks[j].p) * (self.config.xi * self.banks[j].A - bank_i.c[j]))
        #                                      /
        #                                      (self.banks[j].p * bank_i.c[j] * (1 - psi)))
        #
        #
        #                     bank_i.asset_i += bank_i.A
        #                     bank_i.asset_j += self.banks[j].A
        #                     # bank_i.asset_j += 1 - self.banks[j].p
        #                 if bank_i.rij[j] < 0:
        #                     bank_i.rij[j] = self.config.r_i0
        #         except ZeroDivisionError:
        #             bank_i.rij[j] = self.config.r_i0
        #     bank_i.r = np.sum(bank_i.rij) / (self.config.N - 1)
        #     bank_i.asset_i = bank_i.asset_i / (self.config.N - 1)
        #     bank_i.asset_j = bank_i.asset_j / (self.config.N - 1)
        #     if bank_i.r < min_r:
        #         min_r = bank_i.r
        for bank in self.banks:
            bank.mu = self.eta * (bank.C / max_c) + (1 - self.eta) * (min_r / bank.r)
        self.config.lender_change.step_setup_links(self)
        for bank in self.banks:
            log_change_lender = self.config.lender_change.change_lender(self, bank, self.t)
            self.log.debug('links', log_change_lender)

    def estimate_average_values_for_replacement_of_banks(self):
        self.value_for_reintroduced_banks_L = self.config.L_i0
        self.value_for_reintroduced_banks_E = self.config.E_i0
        self.value_for_reintroduced_banks_D = self.config.D_i0
        if self.config.reintroduce_with_median:
            banks_l = []
            banks_e = []
            banks_d = []
            for bank in self.banks:
                if not bank.failed and bank.E > 0:
                    banks_l.append(bank.L)
                    banks_d.append(bank.D)
                    banks_e.append(bank.E)
            if banks_l:
                self.value_for_reintroduced_banks_L = np.median(banks_l)
                self.value_for_reintroduced_banks_D = np.median(banks_d)
                self.value_for_reintroduced_banks_E = np.median(banks_e)

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
            return self.model.banks[self.lender].rij[self.id]
            #import math
            #return math.sqrt(math.sqrt(self.model.banks[self.lender].rij[self.id]))/20


    def get_id(self, short: bool=False):
        init = 'bank#' if not short else '#'
        if self.failures > 0:
            return '{}{}.{}'.format(init, self.id, self.failures)
        else:
            return '{}{}'.format(init, self.id)

    def __init__(self, new_id=None, bank_model=None):
        if not new_id is None and bank_model:
            self.id = new_id
            self.model = bank_model
            self.lender = None
            self.failures = 0
        self.L = self.model.value_for_reintroduced_banks_L
        self.D = self.model.value_for_reintroduced_banks_D
        self.E = self.model.value_for_reintroduced_banks_E
        self.A = 0
        self.r = 0
        self.rij: list[Any] = [0] * self.model.config.N
        self.c: list[Any] = []
        self.h = 0
        self.p = 0
        self.R = self.model.config.reserves * self.D
        self.C = self.D + self.E - self.L - self.R
        self.mu = 0
        self.l = 0
        self.s = 0
        self.d = 0
        self.B = 0
        self.psi = self.E / self.model.maxE if self.model.config.psi_endogenous else None
        self.incrD = 0
        self.paid_profits = 0
        self.paid_loan = 0
        self.incrR = 0
        self.rationing = 0
        self.lambda_ = 0
        self.failed = False
        self.lender = self.model.config.lender_change.new_lender(self.model, self)
        self.active_borrowers = {}
        self.asset_i = 0
        self.asset_j = 0

    def replace_bank(self):
        self.failures += 1
        self.__init__()

    def do_bankruptcy(self, phase):
        self.failed = True
        self.model.statistics.bankruptcy[self.model.t] += 1
        recovered_in_fire_sales = self.L * self.model.config.rho
        recovered = recovered_in_fire_sales - self.D
        if recovered < 0:
            recovered = 0
        if recovered > self.l:
            recovered = self.l
        bad_debt = self.l - recovered
        self.D = 0
        if not self.get_lender() is None and self.l > 0:
            if bad_debt > 0:
                self.get_lender().B += bad_debt
                self.get_lender().E -= bad_debt
                if self.get_lender().E < 0:
                    self.model.log.debug(phase,
                                         '{} lender is bankrupted  borrower {} does not return loan and lender E<0: {}'.
                                         format(self.get_lender().get_id(), self.get_id(), self.get_lender().E))
                    self.get_lender().failed = True
                self.get_lender().C += recovered
                self.model.log.debug(phase, '{} bankrupted (fire sale={},recovers={},paidD={})(lender{}.ΔB={},ΔC={})'.
                                     format(self.get_id(), recovered_in_fire_sales, recovered, self.D,
                                            self.get_lender().get_id(short=True), bad_debt, recovered))
            elif self.l > 0 and self.get_lender() is not None:
                self.get_lender().C += self.l
                self.model.log.debug(phase, '{} bankrupted (lender{}.ΔB=0,ΔC={}) (paidD={})'.format(
                    self.get_id(), self.get_lender().get_id(short=True), recovered, self.l))
            self.get_lender().s += recovered
            if self.id in self.get_lender().active_borrowers:
                del self.get_lender().active_borrowers[self.id]
        return recovered

    def do_fire_sales(self, amount_to_sell, reason, phase):
        cost_of_sell = (amount_to_sell / self.model.config.rho) if self.model.config.rho else np.inf
        extra_cost_of_selling = cost_of_sell * (1 - self.model.config.rho)
        if cost_of_sell > self.L:
            self.model.log.debug(phase, '{} impossible fire sale to recover {}: cost_sell_L={} > L={}: {}'.format(
                self.get_id(), amount_to_sell, cost_of_sell, self.L, reason))
            return self.do_bankruptcy(phase)
        else:
            self.L -= cost_of_sell
            self.E -= extra_cost_of_selling
            self.model.log.debug(phase, '{} fire sales {} so L-={} and affects to E-={}'.format(
                self.get_id(), amount_to_sell, cost_of_sell, extra_cost_of_selling))
            if self.L <= self.model.config.alfa:
                self.model.log.debug(phase,
                                     '{} new L={} is under threshold {} and makes bankruptcy of bank: {}'.format(
                                         self.get_id(), self.L, self.model.config.alfa, reason))
                self.do_bankruptcy(phase)
                return amount_to_sell
            else:
                if self.E <= self.model.config.alfa:
                    self.model.log.debug(phase,
                                         '{} new E={} is under threshold {} and makes bankruptcy of bank: {}'.format(
                                             self.get_id(), self.E, self.model.config.alfa, reason))
                    self.do_bankruptcy(phase)
                return amount_to_sell

    def __str__(self, details=False):
        text = '{} C={} R={} L={}'.format(self.get_id(short=True), Log.__format_number__(self.C),
                                          Log.__format_number__(self.R), Log.__format_number__(self.L))
        amount_borrowed = 0
        list_borrowers = ' borrows=['
        for bank_i in self.active_borrowers:
            list_borrowers += self.model.banks[bank_i].get_id(short=True) + ','
            amount_borrowed += self.active_borrowers[bank_i]
        if amount_borrowed:
            text += ' l={}'.format(Log.__format_number__(amount_borrowed))
            list_borrowers = list_borrowers[:-1] + ']'
        else:
            text += '        '
            list_borrowers = ''
        text += ' | D={} E={}'.format(Log.__format_number__(self.D), Log.__format_number__(self.E))
        if details and hasattr(self, 'd') and self.d and self.l:
            text += ' l={}'.format(Log.__format_number__(self.d))
        else:
            text += '        '
        if details and hasattr(self, 's') and self.s:
            text += ' s={}'.format(Log.__format_number__(self.s))
        elif details and hasattr(self, 'd') and self.d:
            text += ' d={}'.format(Log.__format_number__(self.d))
        else:
            text += '        '
        if self.failed:
            text += ' FAILED '.format()
        elif details and hasattr(self, 'd') and (self.d > 0):
            if self.get_lender() is None:
                text += ' no lender'.format()
            else:
                text += ' lender{},r={}%'.format(self.get_lender().get_id(short=True), self.get_loan_interest())
        else:
            text += list_borrowers
        text += ' B={}'.format(Log.__format_number__(self.B)) if self.B else '        '
        if self.model.config.psi_endogenous and self.psi:
            text += ' psi={}'.format(Log.__format_number__(self.psi))
        return text

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
                for str_t in param.split(','):
                    t.append(int(str_t))
                    if t[-1] > Config.T or t[-1] < 0:
                        raise ValueError('{} greater than Config.T or below 0'.format(t[-1]))
                return t

    @staticmethod
    def run_interactive():
        """
            Run interactively the model
        """
        global model
        parser = argparse.ArgumentParser()
        parser.description = "<config=value> to set up Config options. use '?' to see values"
        parser.add_argument('--log', default='ERROR', help='Log level messages (ERROR,DEBUG,INFO...)')
        parser.add_argument('--modules', default=None,
                            help='Log only this modules (separated by ,)'.format())
        parser.add_argument('--logfile', default=None, help='File to send logs to')
        parser.add_argument('--save', default=None, help='Saves the output of this execution'.format())
        parser.add_argument('--graph', default=None,
                            help='List of t in which save the network config (* for all)'.format())
        parser.add_argument('--gif_graph', default=False, type=bool,
                            help='If --graph, then also an animated gif with all graphs '.format())
        parser.add_argument('--graph_stats', default=False, type=str,
                            help='Load a json of a graph and give us statistics of it'.format())
        parser.add_argument('--n', type=int, default=Config.N, help='Number of banks'.format())
        parser.add_argument('--eta', type=float, default=Model.eta, help='Policy recommendation'.format())
        parser.add_argument('--t', type=int, default=Config.T, help='Time repetitions'.format())
        parser.add_argument('--lc', type=str, default=LENDER_CHANGE_DEFAULT,
                            help="Bank lender's change method (?=list)")
        parser.add_argument('--lc_p', '--p', type=float, default=LENDER_CHANGE_DEFAULT_P,
                            help="For Erdos-Renyi bank lender's change value of p".format())
        parser.add_argument('--lc_m', '--m', type=int, default=None,
                            help="For Preferential bank lender's change value of graph grade m".format())
        parser.add_argument('--lc_ini_graph_file', type=str, default=None,
                            help='Load a graph in json networkx.node_link_data() format')
        parser.add_argument('--detailed_equity', action='store_true',
                            default=model.config.detailed_equity,
                            help='Store in a gdt the individual E evolution of each individual bank')
        parser.add_argument('--psi_endogenous', action='store_true',
                            default=model.config.psi_endogenous, help='Market power variable psi will be endogenous')
        parser.add_argument('--plot_format', type=str, default='none',
                            help='Generate plots with the specified format (svg,png,pdf,gif,agr)')
        parser.add_argument('--output_format', type=str, default='gdt',
                            help='File extension for data (gdt,txt,csv,both)')
        parser.add_argument('--output', type=str, default=None,

                            help='Directory where to store the results')
        parser.add_argument('--no_replace', action='store_true',
                            default=not model.config.allow_replacement_of_bankrupted,
                            help='No replace banks when they go bankrupted')
        parser.add_argument('--reintr_with_median', action='store_true',
                            default=model.config.reintroduce_with_median,
                            help='Reintroduce banks with the median of current banks')
        parser.add_argument('--seed', type=int, default=None, help='seed used for random generator')
        args, other_possible_config_args = parser.parse_known_args()
        if args.graph_stats:
            lc.GraphStatistics.describe(args.graph_stats, interact=True)
        if args.t != model.config.T:
            model.config.T = args.t
        if args.n != model.config.N:
            model.config.N = args.n
        if args.eta != model.eta:
            model.eta = args.eta
        model.config.allow_replacement_of_bankrupted = not args.no_replace
        model.config.reintroduce_with_median = args.reintr_with_median
        model.config.psi_endogenous = args.psi_endogenous
        model.config.lender_change = lc.determine_algorithm(args.lc, args.lc_p, args.lc_m)
        model.config.define_values_from_args(other_possible_config_args)
        if model.config.seed and model.config.seed != model.default_seed and args.seed is None:
            args.seed = model.config.seed
        model.config.lender_change.set_initial_graph_file(args.lc_ini_graph_file)
        model.log.define_log(args.log, args.logfile, args.modules)
        model.statistics.define_output_format(args.output_format)
        model.statistics.set_gif_graph(args.gif_graph)
        model.statistics.define_plot_format(args.plot_format)
        model.statistics.define_detailed_equity(args.detailed_equity, args.save)
        Utils.run(args.save, Utils.__extract_t_values_from_arg__(args.graph),
                  output_directory=args.output, seed=args.seed,
                  interactive=args.log == 'ERROR' or args.logfile is not None)

    @staticmethod
    def run(save=None, save_graph_instants=None, interactive=False, output_directory=None, seed=None):
        global model
        if not save_graph_instants and Config.GRAPHS_MOMENTS:
            save_graph_instants = Config.GRAPHS_MOMENTS
        initial_time = time.perf_counter()
        model.initialize(export_datafile=save, save_graphs_instants=save_graph_instants,
                         output_directory=output_directory, seed=seed)
        model.simulate_full(interactive=interactive)
        result = model.finish()
        if interactive and model.statistics.get_cross_correlation_result(0):
            print('\n'+model.statistics.get_cross_correlation_result(0))
            print(model.statistics.get_cross_correlation_result(1))
            print('bankruptcy.mean: %s' % model.statistics.bankruptcy.mean())
            print('interest_rate.mean: %s' % model.statistics.interest_rate.mean())
            final_time = time.perf_counter()
            print('execution_time: %2.5f secs' % (final_time - initial_time))
        return result

    @staticmethod
    def is_notebook():
        try:
            __IPYTHON__
            return get_ipython().__class__.__name__ != 'SpyderShell'
        except NameError:
            return False

    @staticmethod
    def is_spyder():
        try:
            return get_ipython().__class__.__name__ == 'SpyderShell'
        except NameError:
            return False
model = Model()
if Utils.is_notebook():
    model.statistics.OUTPUT_DIRECTORY = '/content'
    model.statistics.output_format = 'csv'
    Utils.run(save='results')
elif __name__ == '__main__':
    Utils.run_interactive()