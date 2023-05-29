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
from bokeh.io import export_svg
import numpy as np
import matplotlib.pyplot as plt


class Config:
    """
    Configuration parameters for the interbank network
    You can change whatever directly in the file, or:
    - Interactive using --config N=5
    - Creating an instance of model and calling model.configure( N=5 )
    """
    T: int    = 1000     # time (1000)
    N: int    = 50       # number of banks (50)

    ## not used in this implementation:
    ##ȓ: float  = 0.02     # percentage reserves (at the moment, no R is used)
    ##đ: int    = 1        # number of outgoing links allowed

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

    def reset(self, model):
        self.model = model
        self.bankruptcy = [0 for i in range(model.config.T)]
        self.bestLender = [-1 for i in range(model.config.T)]
        self.bestLenderClients = [0 for i in range(model.config.T)]
        self.liquidity = [0 for i in range(model.config.T)]
        self.interest = [0 for i in range(model.config.T)]
        self.incrementD = [0 for i in range(model.config.T)]
        self.B = [0 for i in range(model.config.T)]

    def computeBestLender(self):

        lenders = {}
        for bank in Model.banks:
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

        self.bestLender[Model.t] = best
        self.bestLenderClients[Model.t] = bestValue

    def computeInterest(self):
        interest = 0
        for bank in Model.banks:
            interest += bank.getLoanInterest()
        interest = interest / Model.config.N

        self.interest[Model.t] = interest


    def computeLiquidity(self):
        total = 0
        for bank in Model.banks:
            total += bank.C
        self.liquidity[Model.t] = total


    def finish(self):
        totalB = 0
        for bank_i in Model.banks:
            totalB += bank_i.B
        self.B[Model.t] = totalB


    def export_data(self):
        if Model.log.isNotebook():
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
        for i in range(Model.config.T):
            xx.append(i)
            yy.append(self.bankruptcy[i])

        if Model.log.isNotebook():
            p = bokeh.plotting.figure(title=title, x_axis_label="Time", y_axis_label="num of bankruptcies",
                                      sizing_mode="stretch_width",
                                      height=550)
            p.line(xx, yy, color="blue", line_width=2)
            bokeh.plotting.show(p)
        else:
            with open(f"output/{title}.txt".replace(" ", "_").lower(), 'w') as f:
                f.write('# t   numBankruptcies\n')
                for i in range(Model.config.T):
                    f.write(f"{xx[i]} {yy[i]}\n")

            # bokeh.plotting.output_file(filename=f"output/{title}.html".replace(" ", "_").lower(), title=title)
            # bokeh.plotting.save(p)

    def export_data_interest_rate(self):
        title = "Interest"
        xx = []
        yy = []
        for i in range(Model.config.T):
            xx.append(i)
            yy.append(self.interest[i])
        if Model.log.isNotebook():
            p = bokeh.plotting.figure(title=title, x_axis_label='Time', y_axis_label='interest',
                                      sizing_mode="stretch_width",
                                      height=550)
            p.line(xx, yy, color="blue", line_width=2)
            bokeh.plotting.show(p)
        else:
            with open(f"output/{title}.txt".replace(" ", "_").lower(), 'w') as f:
                f.write('# t   Interest\n')
                for i in range(Model.config.T):
                    f.write(f"{xx[i]} {yy[i]}\n")
            # bokeh.plotting.output_file(filename=f"output/{title}.html".replace(" ","_").lower(), title=title)
            # bokeh.plotting.save(p)

    def export_data_liquidity(self):
        title = "Liquidity"
        xx = []
        yy = []
        for i in range(Model.config.T):
            xx.append(i)
            yy.append(self.liquidity[i])
        if Model.log.isNotebook():
            p = bokeh.plotting.figure(title=title, x_axis_label='Time', y_axis_label='Ʃ liquidity',
                                      sizing_mode="stretch_width",
                                      height=550)
            p.line(xx, yy, color="blue", line_width=2)
            bokeh.plotting.show(p)
        else:
            with open(f"output/{title}.txt".replace(" ", "_").lower(), 'w') as f:
                f.write('# t   Liquidity\n')
                for i in range(Model.config.T):
                    f.write(f"{xx[i]} {yy[i]}\n")
            # bokeh.plotting.output_file(filename=f"output/{title}.html".replace(" ","_").lower(), title=title)
            # bokeh.plotting.save(p)

    def export_data_best_lender(self):
        title = "Best Lender"
        xx = []
        yy = []
        yy2 = []
        for i in range(Model.config.T):
            xx.append(i)
            yy.append(self.bestLender[i] / Model.config.N)
            yy2.append(self.bestLenderClients[i] / Model.config.N)

        if Model.log.isNotebook():
            p = bokeh.plotting.figure(title=title, x_axis_label='Time', y_axis_label='Best lenders',
                                      sizing_mode="stretch_width",
                                      height=550)
            p.line(xx, yy, legend_label=f"{title} id", color="black", line_width=2)
            p.line(xx, yy2, legend_label=f"{title} num clients", color="red", line_width=2, line_dash='dashed')
            bokeh.plotting.show(p)
        else:
            with open(f"output/{title}.txt".replace(" ", "_").lower(), 'w') as f:
                f.write('# t   bestLenders numClients\n')
                for i in range(Model.config.T):
                    f.write(f"{xx[i]} {yy[i]} {yy2[i]}\n")
            # bokeh.plotting.output_file(filename=f"output/{title}.html".replace(" ","_").lower(), title=title)
            # bokeh.plotting.save(p)


class Log:
    logger = logging.getLogger("model")
    modules = []

    def __format_number__(self,number):
        result = f"{number:5.2f}"
        while len(result) > 5 and result[-1] == "0":
            result = result[:-1]
        while len(result) > 5 and result.find('.') > 0:
            result = result[:-1]
        return result

    def __get_string_debug_banks__(self,details, bank):
        text = f"{bank.getId():10} C={self.__format_number__(bank.C)} L={self.__format_number__(bank.L)}"
        amount_borrowed = 0
        list_borrowers = " borrows=["
        for bank_i in bank.activeBorrowers:
            list_borrowers += Model.banks[bank_i].getId(short=True) + ","
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

    def debugBanks(self,details: bool = True, info: str = ''):
        for bank in Model.banks:
            if not info:
                info = "-----"
            self.debug(info, self.__get_string_debug_banks__(details, bank))

    def getLevel(self,option):
        try:
            return getattr(logging, option.upper())
        except:
            logging.error(f" '--log' must contain a valid logging level and {option.upper()} is not.")
            sys.exit(-1)
            return None

    def debug(self,module, text):
        if self.modules == [] or module in self.modules:
            self.logger.debug(f"t={Model.t:03}/{module:6} {text}")

    def info(self,module, text):
        if self.modules == [] or module in self.modules:
            self.logger.info(f" t={Model.t:03}/{module:6} {text}")

    def error(self,module, text):
        self.logger.error(f"t={Model.t:03}/{module:6} {text}")

    def defineLog(self,log: str, logfile: str = '', modules: str = '', script: str = ''):
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
    To call one of step of T:
        Model.initialize()
        ...
        Model.doAStepOfSimulation()
    To execute the full simulation:
        Model.initialize()
        Model.doFullSimulation()

    """
    banks = []    # An array of Bank with size Model.config.N
    t: int = 0    # current value of time, t = 0..Model.config.T
    ŋ: float = 1  # eta : current policy recommendation currently

    log = Log()
    statistics = Statistics()
    config = Config()

    @staticmethod
    def initialize():
        Model.statistics.reset(Model)
        random.seed(Model.config.seed)
        Model.banks = []
        t = 0
        for i in range(Model.config.N):
            Model.banks.append(Bank(i))

    @staticmethod
    def doAStepOfSimulation():
        Model.initStep()
        Model.doShock("shock1")
        Model.doLoans()
        Model.log.debugBanks()
        Model.doShock("shock2")
        Model.doRepayments()
        Model.log.debugBanks()
        Model.statistics.computeLiquidity()
        Model.statistics.computeBestLender()
        Model.statistics.computeInterest()
        Model.setupLinks()
        Model.log.debugBanks()


    @staticmethod
    def doFullSimulation():
        for Model.t in range(Model.config.T):
            Model.doAStepOfSimulation()

        
    @staticmethod
    def finish():
        Model.statistics.finish()
        if not 'unittest' in sys.modules:
            Model.statistics.export_data()

    @staticmethod
    def getFitness():
        globalμ = 0
        for bank in Model.banks:
            globalμ += bank.μ
        return globalμ

    @staticmethod
    def doShock(whichShock):
        # (equation 2)
        for bank in Model.banks:
            bank.newD = bank.D * (Model.config.µ + Model.config.ω * random.random())
            bank.ΔD = bank.newD - bank.D
            bank.D = bank.newD
            if bank.ΔD >= 0:
                bank.C += bank.ΔD
                # if "shock1" then we can be a lender:
                if whichShock == "shock1":
                    bank.s = bank.C
                bank.d = 0  # it will not need to borrow
                if bank.ΔD > 0:
                    Model.log.debug(whichShock,
                                 f"{bank.getId()} wins ΔD={bank.ΔD:.3f}")
            else:
                # if "shock1" then we cannot be a lender: we have lost deposits
                if whichShock == "shock1":
                    bank.s = 0
                if bank.ΔD + bank.C >= 0:
                    bank.d = 0  # it will not need to borrow
                    bank.C += bank.ΔD
                    Model.log.debug(whichShock,
                                 f"{bank.getId()} loses ΔD={bank.ΔD:.3f}, covered by capital")
                else:
                    bank.d = abs(bank.ΔD + bank.C)  # it will need money
                    Model.log.debug(whichShock,
                                 f"{bank.getId()} loses ΔD={bank.ΔD:.3f} but has only C={bank.C:.3f}")
                    bank.C = 0  # we run out of capital
            Model.statistics.incrementD[Model.t] += bank.ΔD

    @staticmethod
    def doLoans():
        for bank in Model.banks:
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
                            bank.l = bank.getLender().s  # amount of loan (writed in the borrower)
                            bank.getLender().activeBorrowers[
                                bank.id] = bank.getLender().s  # amount of loan (writed in the lender)
                            bank.getLender().C -= bank.l  # amount of loan that reduces lender capital
                            bank.getLender().s = 0
                    else:
                        bank.l = bank.d  # amount of loan (writed in the borrower)
                        bank.getLender().activeBorrowers[bank.id] = bank.d  # amount of loan (writed in the lender)
                        bank.getLender().s -= bank.d  # the loan reduces our lender's capacity to borrow to others
                        bank.getLender().C -= bank.d  # amount of loan that reduces lender capital
                        Model.log.debug("loans",
                                     f"{bank.getId()} new loan l={bank.d:.3f} from {bank.getLender().getId()}")

            # the shock can be covered by own capital
            else:
                bank.l = 0
                if len(bank.activeBorrowers) > 0:
                    list_borrowers = ""
                    amount_borrowed = 0
                    for bank_i in bank.activeBorrowers:
                        list_borrowers += Model.banks[bank_i].getId(short=True) + ","
                        amount_borrowed += bank.activeBorrowers[bank_i]
                    Model.log.debug("loans", f"{bank.getId()} has a total of {len(bank.activeBorrowers)} loans with " +
                                 f"[{list_borrowers[:-1]}] of l={amount_borrowed}")

    @staticmethod
    def doRepayments():
        # first all borrowers must pay their loans:
        for bank in Model.banks:
            if bank.l > 0:
                loanProfits = bank.getLoanInterest() * bank.l
                loanToReturn = bank.l + loanProfits
                # (equation 3)
                if loanToReturn > bank.C:
                    weNeedToSell = loanToReturn - bank.C
                    bank.C = 0
                    bank.paidloan = bank.doFiresalesL(weNeedToSell,
                                                      f"to return loan and interest {loanToReturn:.3f} > C={bank.C:.3f}",
                                                      "repay")
                # the firesales of line above could bankrupt the bank, if not, we pay "normally" the loan:
                else:
                    bank.C -= loanToReturn
                    bank.E -= loanProfits
                    bank.paidloan = bank.l
                    bank.l = 0
                    bank.getLender().s -= bank.l  # we reduce the  's' => the lender could have more loans
                    bank.getLender().C += loanToReturn  # we return the loan and it's profits
                    bank.getLender().E += loanProfits  # the profits are paid as E
                    Model.log.debug("repay",
                                 f"{bank.getId()} pays loan {loanToReturn:.3f} (E={bank.E:.3f},C={bank.C:.3f}) to lender" +
                                 f" {bank.getLender().getId()} (ΔE={loanProfits:.3f},ΔC={bank.l:.3f})")

        # now  when ΔD<0 it's time to use Capital or sell L again (now we have the loans cancelled, or the bank bankrputed):
        for bank in Model.banks:
            if bank.d > 0 and not bank.failed:
                bank.doFiresalesL(bank.d, f"fire sales due to not enough C", "repay")

        for bank in Model.banks:
            bank.activeBorrowers = {}
            if bank.failed:
                bank.replaceBank()
        Model.log.debug("repay",
                     f"this step ΔD={Model.statistics.incrementD[Model.t]:.3f} and failures={Model.statistics.bankruptcy[Model.t]}")

    @staticmethod
    def initStep():
        for bank in Model.banks:
            bank.B = 0
        if Model.t == 0:
            Model.log.debugBanks()

    @staticmethod
    def setupLinks():
        # (equation 5)
        # p = probability borrower not failing
        # c = lending capacity
        # λ = leverage
        # h = borrower haircut

        maxE = max(Model.banks, key=lambda i: i.E).E
        maxC = max(Model.banks, key=lambda i: i.C).C
        for bank in Model.banks:
            bank.p = bank.E / maxE
            bank.λ = bank.L / bank.E
            bank.ΔD = 0

        maxλ = max(Model.banks, key=lambda i: i.λ).λ
        for bank in Model.banks:
            bank.h = bank.λ / maxλ
            bank.A = bank.C + bank.L  # bank.L / bank.λ + bank.D

        # determine c (lending capacity) for all other banks (to whom give loans):
        for bank in Model.banks:
            bank.c = []
            for i in range(Model.config.N):
                ##print((1 - Model.banks[i].h) * Model.banks[i].A, Model.banks[i].h,Model.banks[i].A)
                c = 0 if i == bank.id else (1 - Model.banks[i].h) * Model.banks[i].A
                bank.c.append(c)

        # (equation 6)
        minr = sys.maxsize
        lines = []
        for bank_i in Model.banks:
            line1 = ""
            line2 = ""
            for j in range(Model.config.N):
                try:
                    if j == bank_i.id:
                        bank_i.rij[j] = 0
                    else:
                        bank_i.rij[j] = (Model.config.Χ * bank_i.A -
                                         Model.config.Φ * Model.banks[j].A -
                                         (1 - Model.banks[j].p) *
                                         (Model.config.ξ * Model.banks[j].A - bank_i.c[j])) \
                                        / (Model.banks[j].p * bank_i.c[j])
                        if bank_i.rij[j] < 0:
                            bank_i.rij[j] = Model.config.r_i0
                # the first t=1, maybe t=2, the shocks have not affected enough to use L (only C), so probably
                # L and E are equal for all banks, and so maxλ=anyλ and h=1 , so cij=(1-1)A=0, and r division
                # by zero -> solution then is to use still r_i0:
                except ZeroDivisionError:
                    bank_i.rij[j] = Model.config.r_i0

                line1 += f"{bank_i.rij[j]:.3f},"
                line2 += f"{bank_i.c[j]:.3f},"
            if lines != []:
                lines.append("  |" + line2[:-1] + "|   |" +
                             line1[:-1] + f"| {bank_i.getId(short=True)} h={bank_i.h:.3f},λ={bank_i.λ:.3f} ")
            else:
                lines.append("c=|" + line2[:-1] + "| r=|" +
                             line1[:-1] + f"| {bank_i.getId(short=True)} h={bank_i.h:.3f},λ={bank_i.λ:.3f} ")
            bank_i.r = np.sum(bank_i.rij) / (Model.config.N - 1)
            if bank_i.r < minr:
                minr = bank_i.r

        if Model.config.N < 10:
            for line in lines:
                Model.log.debug("links", f"{line}")
        Model.log.debug("links", f"maxE={maxE:.3f} maxC={maxC:.3f} maxλ={maxλ:.3f} minr={minr:.3f} ŋ={Model.ŋ:.3f}")

        # (equation 7)
        loginfo = loginfo1 = ""
        for bank in Model.banks:
            bank.μ = Model.ŋ * (bank.C / maxC) + (1 - Model.ŋ) * (minr / bank.r)
            loginfo += f"{bank.getId(short=True)}:{bank.μ:.3f},"
            loginfo1 += f"{bank.getId(short=True)}:{bank.r:.3f},"
        if Model.config.N < 10:
            Model.log.debug("links", f"μ=[{loginfo[:-1]}] r=[{loginfo1[:-1]}]")

        # we can now break old links and set up new lenders, using probability P
        # (equation 8)
        for bank in Model.banks:
            possible_lender = bank.newLender()
            possible_lender_μ = Model.banks[possible_lender].μ
            current_lender_μ = bank.getLender().μ
            bank.P = 1 / (1 + math.exp(-Model.config.β * (possible_lender_μ - current_lender_μ)))

            if bank.P >= 0.5:
                Model.log.debug("links",
                             f"{bank.getId()} new lender is #{possible_lender} from #{bank.lender} with %{bank.P:.3f}")
                bank.lender = possible_lender
            else:
                Model.log.debug("links", f"{bank.getId()} maintains lender #{bank.lender} with %{1 - bank.P:.3f}")


# %%

class Bank:
    """
    It represents an individual bank of the network, with the logic of interaction between it and the interbank system

    """
    def getLender(self):
        return Model.banks[self.lender]

    def getLoanInterest(self):
        return Model.banks[self.lender].rij[self.id]

    def getId(self, short: bool = False):
        init = "bank#" if not short else "#"
        if self.failures > 0:
            return f"{init}{self.id}.{self.failures}"
        else:
            return f"{init}{self.id}"

    def __init__(self, id):
        self.id = id
        self.failures = 0
        self.__assign_defaults__()

    def newLender(self):
        newvalue = None
        # r_i0 is used the first time the bank is created:
        if self.lender == None:
            self.rij = [Model.config.r_i0 for i in range(Model.config.N)]
            self.rij[self.id] = 0
            self.r = Model.config.r_i0
            self.μ = 0
            # if it's just created, only not to be ourselves is enough
            newvalue = random.randrange(Model.config.N - 1)
        else:
            # if we have a previous lender, new should not be the same
            newvalue = random.randrange(Model.config.N - 2 if Model.config.N > 2 else 1)

        if Model.config.N == 2:
            newvalue = 1 if self.id == 0 else 0
        else:
            if newvalue >= self.id:
                newvalue += 1
                if self.lender != None and newvalue >= self.lender:
                    newvalue += 1
            else:
                if self.lender != None and newvalue >= self.lender:
                    newvalue += 1
                    if newvalue >= self.id:
                        newvalue += 1
        return newvalue

    def __assign_defaults__(self):
        self.L = Model.config.L_i0
        self.C = Model.config.C_i0
        self.D = Model.config.D_i0
        self.E = Model.config.E_i0
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
        Model.statistics.bankruptcy[Model.t] += 1
        recoveredFiresale = self.L * Model.config.ρ  # we firesale what we have
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
            Model.log.debug(phase,f"{self.getId()} bankrupted (fire sale={recoveredFiresale:.3f},recovers={recovered:.3f}"+
                f",paidD={self.D:.3f})(lender{self.getLender().getId(short=True)}.ΔB={badDebt:.3f},ΔC={recovered:.3f})")
        else:
            # self.l=0 no current loan to return:
            if self.l > 0:
                self.paidLoan = self.l  # the loan was paid, not the interest
                self.getLender().C += self.l  # lender not recovers more than loan if it is
                Model.log.debug(phase,f"{self.getId()} bankrupted (lender{self.getLender().getId(short=True)}.ΔB=0,"+
                             f"ΔC={recovered:.3f}) (paidD={self.l:.3f})")
        self.D = 0
        # the loan is not paid correctly, but we remove it
        if self.id in self.getLender().activeBorrowers:
            self.getLender().s -= self.l
            del self.getLender().activeBorrowers[self.id]
        else:
            # TODO . que pasa si quiebra el banco que TIENE prestamos, y no el prestado
            pass

    def doFiresalesL(self, amountToSell, reason, phase):
        costOfSell = amountToSell / Model.config.ρ
        recoveredE = costOfSell * (1 - Model.config.ρ)
        ##self.C = 0
        if costOfSell > self.L:
            Model.log.debug(phase,
                         f"{self.getId()} impossible fire sale sellL={costOfSell:.3f} > L={self.L:.3f}: {reason}")
            return self.__doBankruptcy__(phase)
        else:
            self.L -= costOfSell
            self.E -= recoveredE

            if self.L <= Model.config.α:
                Model.log.debug(phase,
                             f"{self.getId()} new L={self.L:.3f} makes bankruptcy of bank: {reason}")
                return self.__doBankruptcy__(phase)
            else:
                if self.E <= Model.config.α:
                    Model.log.debug(phase,
                                 f"{self.getId()} new E={self.E:.3f} makes bankruptcy of bank: {reason}")
                    return self.__doBankruptcy__(phase)
                else:
                    Model.log.debug(phase,
                                 f"{self.getId()} fire sale sellL={amountToSell:.3f} at cost {costOfSell:.3f} reducing"+
                                 f"E={recoveredE:.3f}: {reason}")
                    return amountToSell



# %%

def runInteractive(log: str = typer.Option('ERROR', help="Log level messages (ERROR,DEBUG,INFO...)"),
                       modules: str = typer.Option(None, help=f"Log only this modules (separated by ,)"),
                       logfile: str = typer.Option(None, help="File to send logs to"),
                       n: int = typer.Option(Model.config.N, help=f"Number of banks"),
                       t: int = typer.Option(Model.config.T, help=f"Time repetitions")):
    """
    Run interactively the model
    """

    if t != Model.config.T:
        Model.config.T = t
    if n != Model.config.N:
        Model.config.N = n
    Model.log.defineLog(log, logfile, modules)
    Model.initialize()
    Model.doFullSimulation()
    Model.finish()


def run():

    Model.initialize()
    Model.doFullSimulation()
    Model.finish()

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

if isNotebook():
    run()
else:
    if __name__ == "__main__":
        typer.run(runInteractive)


# functions to use when it is called as a package:
# ----
# import bank_net
# bank_net.config( param=x )
# bank_net.do_step()
# bank_net.get_current_fitness()
# bank_net.set_policy_recommendation()

def configure(**kwargs):
    for attribute in kwargs:
        if hasattr(Config, attribute):
            currentValue = getattr(Config, attribute)
            if isinstance(currentValue, int):
                setattr(Config, attribute, int(kwargs[attribute]))
            else:
                if isinstance(currentValue, float):
                    setattr(Config, attribute, float(kwargs[attribute]))
                else:
                    raise Exception(f"type of Config.{attribute} not configured: {type(currentValue)}")
        else:
            raise LookupError("attribute not found in Config")
    Model.initialize()



def do_step():
    Model.doAStepOfSimulation()

def get_current_fitness():
    return Model.getFitness()

def set_policy_recommendation( ŋ ):
    Model.ŋ = ŋ
