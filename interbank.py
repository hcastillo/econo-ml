# -*- coding: utf-8 -*-
"""
Generates a simulation of an interbank network following the rules described in paper
  Reinforcement Learning Policy Recommendation for Interbank Network Stability
  from Gabrielle and Alessio

@author: hector@bith.net
@date:   04/2023
"""

import random
import logging
import math
import typer
import sys
import bokeh.plotting
import bokeh.io
import numpy as np


class Config:
    """
    Configuration parameters for the interbank network
    """
    T: int    = 1000     # time (1000)
    N: int    = 50       # number of banks (50)

    # not used in this implementation:
    # ȓ: float  = 0.02     # percentage reserves (at the moment, no R is used)
    # đ: int    = 1        # number of outgoing links allowed

    seed: int = 40579    # seed for this simulation

    # shocks parameters:
    µ: float  = 0.7      # mi
    ω: float  = 0.55     # omega

    # screening costs
    Φ: float  = 0.025    # phi
    Χ: float  = 0.015    # ji
    
    # liquidation cost of collateral
    ξ: float  = 0.3      # xi
    ρ: float  = 0.3      # ro fire sale cost

    β: float  = 0        # intensity of breaking the connection
    α: float  = 0.1      # below this level of E or D, we will bankrupt the bank

    # banks initial parameters
    L_i0: float = 120    # long term assets
    C_i0: float = 30     # capital
    D_i0: float = 135    # deposits
    E_i0: float = 15     # equity
    r_i0: float = 0.02   # initial rate


class Statistics:
    bankruptcy = []
    bestLender = []
    bestLenderClients = []
    liquidity = []
    interest = []
    incrementD = []
    B = []
    model = None

    def __init__(self, model):
        self.model = model

    def reset(self):
        self.bankruptcy = np.zeros(self.model.config.T, dtype=int)
        self.bestLender = np.full(self.model.config.T, -1, dtype=int)
        self.bestLenderClients = np.zeros(self.model.config.T, dtype=int)
        self.liquidity = np.zeros(self.model.config.T, dtype=int)
        self.interest = np.zeros(self.model.config.T, dtype=int)
        self.incrementD = np.zeros(self.model.config.T, dtype=int)
        self.B = np.zeros(self.model.config.T, dtype=int)

    def computeBestLender(self):
        lenders = {}
        for bank in self.model.banks:
            if bank.lender in lenders:
                lenders[bank.lender] += 1
            else:
                lenders[bank.lender] = 1
        best = -1
        bestValue = -1
        for lender in lenders.keys():
            if lenders[lender] > bestValue:
                best = lender
                bestValue = lenders[lender]

        self.bestLender[self.model.t] = best
        self.bestLenderClients[self.model.t] = bestValue

    def computeInterest(self):
        interest = 0
        for bank in self.model.banks:
            interest += bank.getLoanInterest()
        interest = interest / self.model.config.N

        self.interest[self.model.t] = interest

    def computeLiquidity(self):
        total = 0
        for bank in self.model.banks:
            total += bank.C
        self.liquidity[self.model.t] = total

    def finish(self):
        totalB = 0
        for bank_i in self.model.banks:
            totalB += bank_i.B
        self.B[self.model.t] = totalB

    def export_data(self):
        if Utils.isNotebook():
            from bokeh.io import output_notebook
            output_notebook()
        self.export_data_bankruptcies()
        self.export_data_liquidity()
        self.export_data_best_lender()
        self.export_data_interest_rate()

    def export_data_bankruptcies(self):
        title = "Bankruptcies"
        xx = []
        yy = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.bankruptcy[i])

        if Utils.isNotebook():
            p = bokeh.plotting.figure(title=title, x_axis_label="Time", y_axis_label="num of bankruptcies",
                                      sizing_mode="stretch_width",
                                      height=550)
            p.line(xx, yy, color="blue", line_width=2)
            bokeh.plotting.show(p)
        else:
            with open(f"output/{title}.txt".replace(" ", "_").lower(), 'w') as f:
                f.write('# t   numBankruptcies\n')
                for i in range(self.model.config.T):
                    f.write(f"{xx[i]} {yy[i]}\n")

            # bokeh.plotting.output_file(filename=f"output/{title}.html".replace(" ", "_").lower(), title=title)
            # bokeh.plotting.save(p)

    def export_data_interest_rate(self):
        title = "Interest"
        xx = []
        yy = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.interest[i])
        if Utils.isNotebook():
            p = bokeh.plotting.figure(title=title, x_axis_label='Time', y_axis_label='interest',
                                      sizing_mode="stretch_width",
                                      height=550)
            p.line(xx, yy, color="blue", line_width=2)
            bokeh.plotting.show(p)
        else:
            with open(f"output/{title}.txt".replace(" ", "_").lower(), 'w') as f:
                f.write('# t   Interest\n')
                for i in range(self.model.config.T):
                    f.write(f"{xx[i]} {yy[i]}\n")
            # bokeh.plotting.output_file(filename=f"output/{title}.html".replace(" ","_").lower(), title=title)
            # bokeh.plotting.save(p)

    def export_data_liquidity(self):
        title = "Liquidity"
        xx = []
        yy = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.liquidity[i])
        if Utils.isNotebook():
            p = bokeh.plotting.figure(title=title, x_axis_label='Time', y_axis_label='Ʃ liquidity',
                                      sizing_mode="stretch_width",
                                      height=550)
            p.line(xx, yy, color="blue", line_width=2)
            bokeh.plotting.show(p)
        else:
            with open(f"output/{title}.txt".replace(" ", "_").lower(), 'w') as f:
                f.write('# t   Liquidity\n')
                for i in range(self.model.config.T):
                    f.write(f"{xx[i]} {yy[i]}\n")
            # bokeh.plotting.output_file(filename=f"output/{title}.html".replace(" ","_").lower(), title=title)
            # bokeh.plotting.save(p)

    def export_data_best_lender(self):
        title = "Best Lender"
        xx = []
        yy = []
        yy2 = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.bestLender[i] / self.model.config.N)
            yy2.append(self.bestLenderClients[i] / self.model.config.N)

        if Utils.isNotebook():
            p = bokeh.plotting.figure(title=title, x_axis_label='Time', y_axis_label='Best lenders',
                                      sizing_mode="stretch_width",
                                      height=550)
            p.line(xx, yy, legend_label=f"{title} id", color="black", line_width=2)
            p.line(xx, yy2, legend_label=f"{title} num clients", color="red", line_width=2, line_dash='dashed')
            bokeh.plotting.show(p)
        else:
            with open(f"output/{title}.txt".replace(" ", "_").lower(), 'w') as f:
                f.write('# t   bestLenders numClients\n')
                for i in range(self.model.config.T):
                    f.write(f"{xx[i]} {yy[i]} {yy2[i]}\n")
            # bokeh.plotting.output_file(filename=f"output/{title}.html".replace(" ","_").lower(), title=title)
            # bokeh.plotting.save(p)


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

    def __format_number__(self, number):
        result = f"{number:5.2f}"
        while len(result) > 5 and result[-1] == "0":
            result = result[:-1]
        while len(result) > 5 and result.find('.') > 0:
            result = result[:-1]
        return result

    def __get_string_debug_banks__(self, details, bank):
        text = f"{bank.getId():10} C={self.__format_number__(bank.C)} L={self.__format_number__(bank.L)}"
        amount_borrowed = 0
        list_borrowers = " borrows=["
        for bank_i in bank.activeBorrowers:
            list_borrowers += self.model.banks[bank_i].getId(short=True) + ","
            amount_borrowed += bank.activeBorrowers[bank_i]
        if amount_borrowed:
            text += f" l={self.__format_number__(amount_borrowed)}"
            list_borrowers = list_borrowers[:-1] + "]"
        else:
            text += "        "
            list_borrowers = ""
        text += f" | D={self.__format_number__(bank.D)} E={self.__format_number__(bank.E)}"
        if details and hasattr(bank, 'd') and bank.d and bank.l:
            text += f" l={self.__format_number__(bank.d)}"
        else:
            text += "        "
        if details and hasattr(bank, 's') and bank.s:
            text += f" s={self.__format_number__(bank.s)}"
        else:
            if details and hasattr(bank, 'd') and bank.d:
                text += f" d={self.__format_number__(bank.d)}"
            else:
                text += "        "
        if bank.failed:
            text += f" FAILED "
        else:
            if details and hasattr(bank, 'd') and bank.d > 0:
                text += f" lender{bank.getLender().getId(short=True)},r={bank.getLoanInterest():.2f}%"
            else:
                text += list_borrowers
        text += f" B={self.__format_number__(bank.B)}" if bank.B else "        "
        return text

    def debugBanks(self, details: bool = True, info: str = ''):
        for bank in self.model.banks:
            if not info:
                info = "-----"
            self.debug(info, self.__get_string_debug_banks__(details, bank))

    def getLevel(self, option):
        try:
            return getattr(logging, option.upper())
        except AttributeError:
            logging.error(f" '--log' must contain a valid logging level and {option.upper()} is not.")
            sys.exit(-1)

    def debug(self, module, text):
        if self.modules == [] or module in self.modules:
            self.logger.debug(f"t={self.model.t:03}/{module:6} {text}")

    def info(self, module, text):
        if self.modules == [] or module in self.modules:
            self.logger.info(f" t={self.model.t:03}/{module:6} {text}")

    def error(self, module, text):
        self.logger.error(f"t={self.model.t:03}/{module:6} {text}")

    def defineLog(self, log: str, logfile: str = '', modules: str = '', script: str = ''):
        self.modules = modules.split(",") if modules else []
        # https://typer.tiangolo.com/
        scriptName = script if script else "%(module)s"
        formatter = logging.Formatter('%(levelname)s-' + scriptName + '- %(message)s')
        self.logLevel = self.getLevel(log.upper())
        self.logger.setLevel(self.logLevel)
        if logfile:
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
        model.simulate_step()
        μ = model.get_current_fitness()
        model.set_policy_recommendation( ŋ=0.5 )
    """
    banks = []    # An array of Bank with size Model.config.N
    t: int = 0    # current value of time, t = 0..Model.config.T
    ŋ: float = 1  # eta : current policy recommendation currently

    log = None
    statistics = None
    config = None

    def __init__(self):
        self.log = Log(self)
        self.statistics = Statistics(self)
        self.config = Config()

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

    def initialize(self):
        self.statistics.reset()
        random.seed(self.config.seed)
        self.banks = []
        self.t = 0
        for i in range(self.config.N):
            self.banks.append(Bank(i, self))

    def simulate_step(self):
        self.initStep()
        self.doShock("shock1")
        self.doLoans()
        self.log.debugBanks()
        self.doShock("shock2")
        self.doRepayments()
        self.log.debugBanks()
        self.statistics.computeLiquidity()
        self.statistics.computeBestLender()
        self.statistics.computeInterest()
        self.setupLinks()
        self.log.debugBanks()

    def simulate_full(self):
        for self.t in range(self.config.T):
            self.simulate_step()

    def finish(self):
        self.statistics.finish()
        if 'unittest' not in sys.modules:
            self.statistics.export_data()

    def get_fitness(self):
        sum_μ = 0
        for bank in self.banks:
            sum_μ += bank.μ
        return sum_μ

    def set_policy_recommendation(self, ŋ):
        self.ŋ = ŋ

    def doShock(self, whichShock):
        # (equation 2)
        for bank in self.banks:
            bank.newD = bank.D * (self.config.µ + self.config.ω * random.random())
            bank.ΔD = bank.newD - bank.D
            bank.D = bank.newD
            if bank.ΔD >= 0:
                bank.C += bank.ΔD
                # if "shock1" then we can be a lender:
                if whichShock == "shock1":
                    bank.s = bank.C
                bank.d = 0  # it will not need to borrow
                if bank.ΔD > 0:
                    self.log.debug(whichShock,
                                   f"{bank.getId()} wins ΔD={bank.ΔD:.3f}")
            else:
                # if "shock1" then we cannot be a lender: we have lost deposits
                if whichShock == "shock1":
                    bank.s = 0
                if bank.ΔD + bank.C >= 0:
                    bank.d = 0  # it will not need to borrow
                    bank.C += bank.ΔD
                    self.log.debug(whichShock,
                                   f"{bank.getId()} loses ΔD={bank.ΔD:.3f}, covered by capital")
                else:
                    bank.d = abs(bank.ΔD + bank.C)  # it will need money
                    self.log.debug(whichShock,
                                   f"{bank.getId()} loses ΔD={bank.ΔD:.3f} but has only C={bank.C:.3f}")
                    bank.C = 0  # we run out of capital
            self.statistics.incrementD[self.t] += bank.ΔD

    def doLoans(self):
        for bank in self.banks:
            # decrement in which we should borrow
            if bank.d > 0:
                if bank.getLender().d > 0:
                    # if the lender has no increment then NO LOAN could be obtained: we fire sale L:
                    bank.doFiresalesL(bank.d, f"lender {bank.getLender().getId(short=True)} has no money", "loans")
                    bank.l = 0
                else:
                    # if the lender can give us money, but not enough to cover the loan we need also fire sale L:
                    if bank.d > bank.getLender().s:
                        bank.doFiresalesL(bank.d - bank.getLender().s,
                                          f"lender.s={bank.getLender().s:.3f} but need d={bank.d:.3f}", "loans")
                        # only if lender has money, because if it .s=0, all is obtained by fire sales:
                        if bank.getLender().s > 0:
                            bank.l = bank.getLender().s  # amount of loan (wrote in the borrower)
                            bank.getLender().activeBorrowers[
                                bank.id] = bank.getLender().s  # amount of loan (wrote in the lender)
                            bank.getLender().C -= bank.l  # amount of loan that reduces lender capital
                            bank.getLender().s = 0
                    else:
                        bank.l = bank.d  # amount of loan (wrote in the borrower)
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

    def doRepayments(self):
        # first all borrowers must pay their loans:
        for bank in self.banks:
            if bank.l > 0:
                loanProfits = bank.getLoanInterest() * bank.l
                loanToReturn = bank.l + loanProfits
                # (equation 3)
                if loanToReturn > bank.C:
                    weNeedToSell = loanToReturn - bank.C
                    bank.C = 0
                    bank.paidloan = bank.doFiresalesL(weNeedToSell,
                                          f"to return loan and interest {loanToReturn:.3f} > C={bank.C:.3f}", "repay")
                # the firesales of line above could bankrupt the bank, if not, we pay "normally" the loan:
                else:
                    bank.C -= loanToReturn
                    bank.E -= loanProfits
                    bank.paidloan = bank.l
                    bank.l = 0
                    bank.getLender().s -= bank.l  # we reduce the  's' => the lender could have more loans
                    bank.getLender().C += loanToReturn  # we return the loan and it's profits
                    bank.getLender().E += loanProfits  # the profits are paid as E
                    self.log.debug("repay",
                            f"{bank.getId()} pays loan {loanToReturn:.3f} (E={bank.E:.3f},C={bank.C:.3f}) to lender" +
                            f" {bank.getLender().getId()} (ΔE={loanProfits:.3f},ΔC={bank.l:.3f})")

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

    def initStep(self):
        for bank in self.banks:
            bank.B = 0
        if self.t == 0:
            self.log.debugBanks()

    def setupLinks(self):
        # (equation 5)
        # p = probability borrower not failing
        # c = lending capacity
        # λ = leverage
        # h = borrower haircut

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
        if self.config.N < 10:
            self.log.debug("links", f"μ=[{loginfo[:-1]}] r=[{loginfo1[:-1]}]")

        # we can now break old links and set up new lenders, using probability P
        # (equation 8)
        for bank in self.banks:
            possible_lender = bank.newLender()
            possible_lender_μ = self.banks[possible_lender].μ
            current_lender_μ = bank.getLender().μ
            bank.P = 1 / (1 + math.exp(-self.config.β * (possible_lender_μ - current_lender_μ)))

            if bank.P >= 0.5:
                self.log.debug("links",
                             f"{bank.getId()} new lender is #{possible_lender} from #{bank.lender} with %{bank.P:.3f}")
                bank.lender = possible_lender
            else:
                self.log.debug("links", f"{bank.getId()} maintains lender #{bank.lender} with %{1 - bank.P:.3f}")


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
        self.__assign_defaults__()

    def newLender(self):
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
        self.μ = 0    # fitness of the bank:  estimated later
        self.l = 0    # amount of loan done:  estimated later
        self.s = 0    # amount of loan received: estimated later
        self.B = 0    # bad debt: estimated later
        self.failed = False

        # identity of the lender
        self.lender = None
        self.lender = self.newLender()

        self.activeBorrowers = {}

    def replaceBank(self):
        self.failures += 1
        self.__assign_defaults__()

    def __doBankruptcy__(self, phase):
        self.failed = True
        self.model.statistics.bankruptcy[self.model.t] += 1
        recoveredFiresale = self.L * self.model.config.ρ  # we firesale what we have
        recovered = recoveredFiresale - self.D  # we should pay D to clients
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
            self.model.log.debug(phase, f"{self.getId()} bankrupted (fire sale={recoveredFiresale:.3f}," +
                   f"recovers={recovered:.3f},paidD={self.D:.3f})(lender{self.getLender().getId(short=True)}" +
                   f".ΔB={badDebt:.3f},ΔC={recovered:.3f})")
        else:
            # self.l=0 no current loan to return:
            if self.l > 0:
                self.paidLoan = self.l  # the loan was paid, not the interest
                self.getLender().C += self.l  # lender not recovers more than loan if it is
                self.model.log.debug(phase,f"{self.getId()} bankrupted (lender{self.getLender().getId(short=True)}" +
                             f".ΔB=0,ΔC={recovered:.3f}) (paidD={self.l:.3f})")
        self.D = 0
        # the loan is not paid correctly, but we remove it
        if self.id in self.getLender().activeBorrowers:
            self.getLender().s -= self.l
            del self.getLender().activeBorrowers[self.id]
        else:
            # TODO . que pasa si quiebra el banco que TIENE prestamos, y no el prestado
            pass

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
                                 f"{self.getId()} fire sale sellL={amountToSell:.3f} at cost {costOfSell:.3f} reducing"+
                                 f"E={recoveredE:.3f}: {reason}")
                    return amountToSell


class Utils:
    """
    Auxiliary class to encapsulate the
    """
    @staticmethod
    def runInteractive(log: str = typer.Option('ERROR', help="Log level messages (ERROR,DEBUG,INFO...)"),
                       modules: str = typer.Option(None, help=f"Log only this modules (separated by ,)"),
                       logfile: str = typer.Option(None, help="File to send logs to"),
                       n: int = typer.Option(Config.N, help=f"Number of banks"),
                       t: int = typer.Option(Config.T, help=f"Time repetitions")):
        """
            Run interactively the model
        """

        model = Model()
        if t != model.config.T:
            model.config.T = t
        if n != model.config.N:
            model.config.N = n
        model.log.defineLog(log, logfile, modules)
        Utils.run(model)

    @staticmethod
    def run(model:Model):
        model.initialize()
        model.simulate_full()
        model.finish()

    @staticmethod
    def isNotebook():
        try:
            from IPython import get_ipython
            if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
                return False
        except ImportError:
            return False
        except AttributeError:
            return False
        return True

# %%


if Utils.isNotebook():
    # if we are running in a Notebook:
    Utils.run(Model())
else:
    # if we are running interactively:
    if __name__ == "__main__":
        typer.run(Utils.runInteractive)

# in other cases, if you import it, the process will be:
#   model = Model()
#   model.simulate_step() # t=0 -> t=1
