# -*- coding: utf-8 -*-
"""
Generates in MEMORY the simulation of Config.T iterations over Config.N banks.
  Model is Reinforcement Learning Policy Recommendation for Interbank Network Stability
  from Gabrielle and Alessio

@author: hector@bith.net
@date:   04/2023
"""

import random
import logging
import math
import typer
import sys
import matplotlib.pyplot as plt

class Config:
    T: int = 10    # time (1000)
    N: int = 3     # number of banks (50)

    #  ȓ = 0.02  # percentage reserves (at the moment, no R is used)

    # shocks parameters:
    µ: float = 0.7    # pi
    ω: float = 0.55   # omega

    # screening costs
    Φ: float = 0.025  # phi
    Χ: float = 0.015  # ji
    # liquidation cost of collateral
    ξ: float = 0.3    # xi
    ρ: float = 0.3    # fire sale cost

    β: float = 5      # intensity of breaking the connection

    # banks initial parameters
    L_i0: float = 120   # long term assets
    C_i0: float = 30    # capital
    D_i0: float = 135   # deposits
    E_i0: float = 15    # equity
    r_i0: float = 0.02  # initial rate


class Model:
    banks = []
    t: int = 0
    ŋ: float = 0.5  # eta : policy

    @staticmethod
    def initilize():
        random.seed(40579)
        for i in range(Config.N):
            Model.banks.append(Bank(i))

    @staticmethod
    def doSimulation():
        # initially all banks have a random lender
        for Model.t in range(Config.T):
            doShock("First shock")
            doLoans()
            doShock("Second shock")
            doRepayments()
            Statistics.computeLiquidity()
            Statistics.computeBestLender()
            determineMu()
            setupLinks()


# %%

class Bank:
    def getLender(self):
        return Model.banks[self.lender]

    def getLoanInterest(self):
        return Model.banks[self.lender].r[self.id]

    def getId(self):
        if self.failures > 0:
            return f"bank#{self.id}.{self.failures}"
        else:
            return f"bank#{self.id}"

    def __init__(self, id):
        self.id = id
        self.failures = 0
        self.__assign_defaults__()

    def newLender(self):
        newvalue = None
        # r_i0 is used the first time the bank is created:
        if self.lender == None:
            self.r = [Config.r_i0 for i in range(Config.N)]
            self.π = [ 0 for i in range(Config.N) ]
            self.r[self.id] = None  # to ourselves, we don't want to lend anything
            # if it's just created, only not to be ourselves is enough
            newvalue = random.randrange(Config.N - 1 )
        else:
            # if we have a previous lender, new should not be the same
            newvalue = random.randrange(Config.N - 2 )

        if newvalue >= self.id:
            newvalue += 1
            if self.lender and newvalue >= self.lender:
                newvalue += 1
        else:
          if self.lender and newvalue >= self.lender:
            newvalue += 1
            if newvalue >= self.id:
                newvalue += 1

        return newvalue

    def __assign_defaults__(self):
        self.L = Config.L_i0
        self.C = Config.C_i0
        self.D = Config.D_i0
        self.E = Config.E_i0
        # interbank rates
        self.lender = None
        self.lender = self.newLender()
        # TODO self.R = Config.ȓ * Config.D_i0

    def replaceBank(self):
        self.failures += 1
        self.__assign_defaults__()
        Statistics.bankruptcy[Model.t] += 1



def doShock(whichShock):
    # (equation 2)
    for bank in Model.banks:
        newD = bank.D * (Config.µ + Config.ω * random.random())
        bank.ΔD = newD - bank.D
        bank.D  = newD



    Status.debugBanks(details=False,info=whichShock)


def doLoans():
    for bank in Model.banks:
        bank.B = 0
        if bank.ΔD + bank.C > 0:
            bank.s = bank.ΔD + bank.C  # lender
            bank.d = 0
        else:
            bank.d = abs(bank.ΔD + bank.C)  # borrower
            bank.s = 0

    for bank in Model.banks:
        # decrement in which we should borrow
        if bank.ΔD + bank.C < 0:
            if bank.d > bank.getLender().s:
                bank.l = bank.getLender().s
                bank.getLender().s = 0
                bank.sellL = (bank.d - bank.l) / Config.ρ
            else:
                bank.l = bank.d
                bank.getLender().s -= bank.d  # bank.d
                bank.sellL = 0

            if bank.sellL > 0:
                bank.L -= bank.sellL
                bank.C = 0
                Status.debug("loans",
                    f"{bank.getId()} firesales L={bank.sellL:.3f} to cover ΔD={bank.ΔD:.3f} as {bank.getLender().getId()} gives {bank.l:.3f} and C={bank.C:.3f}")
            else:
                bank.C -= bank.d
                Status.debug("loans",
                    f"{bank.getId()} borrows {bank.l:.3f} from {bank.getLender().getId()} (who still has {bank.getLender().s:.3f}) to cover ΔD={bank.ΔD:.3f} and C={bank.C:.3f}")

        # increment of deposits, or decrement covered by own capital
        else:
            bank.l = 0
            bank.sellL = 0
            if bank.ΔD < 0:
                bank.C += bank.ΔD
                Status.debug("loans",
                    f"{bank.getId()} loses ΔD={bank.ΔD:.3f}, covered by capital, now C={bank.C:.3f}")
    # Status.debugBanks(info='Loans after')


def doRepayments():
    for bank in Model.banks:
        # let's return previous loan bank.l (if it exists):
        # if bank.l>0 , sure we have bank.C = ΔD
        if bank.l > 0:
            loanProfits  = bank.getLoanInterest() * bank.l
            loanToReturn = bank.l + loanProfits
            # (equation 3)
            # i) the new shock gives ΔD positive and with it, we can pay the loan:
            if bank.ΔD - loanToReturn > 0:
                bank.C -= loanToReturn
                bank.E -= loanProfits
                bank.getLender().s -= bank.l       # we reduce the current 's' => the bank could have more loans
                bank.getLender().C += bank.l       # we return the loan
                bank.getLender().E += loanProfits  # the profits are paid as E
                Status.debug("repay",
                    f"{bank.getId()} pays loan {loanToReturn:.3f} (E={bank.E:.3f},C={bank.C:.3f}) to lender {bank.getLender().getId()} (ΔE={loanProfits:.3f},ΔC={bank.l:.3f})")

            # ii) not enough increment to pay the loan: then firesales of L to cover the loan to return and interests
            else:
                weNeedToSell = loanToReturn - bank.ΔD
                firesaleCost = weNeedToSell / Config.ρ
                # the firesale covers the loan
                if firesaleCost <= bank.L:
                    bank.C = 0
                    bank.L -= firesaleCost
                    bank.E = bank.L - bank.D
                    bank.getLender().s -= bank.l  # we reduce the current 's' => the bank could have more loans
                    bank.getLender().C += bank.l  # we return the loan
                    bank.getLender().E += loanProfits  # the profits are paid as E
                    Status.debug("repay",
                        f"{bank.getId()} fire sales={firesaleCost:.3f} to pay loan {loanToReturn:.3f} (C=0,L={bank.L:.3f},E={bank.E:.3f}) to lender {bank.getLender().getId()} (ΔE={loanProfits:.3f},ΔC={bank.l:.3f})")

                # iii) the worst: bankrupcy, not enough L to sell and cover bank obligations:
                else:
                    bank.getLender().s -= bank.l  # the loan is not paid correctly, but we remove it
                    # bank.getLender().C is not increased -> the loan is failed. We move the fire sales to lender E
                    fireSaleBankrupt = bank.L * (1 - Config.ρ)
                    # l = 5
                    # fireSaleBankrupt = 3 -> B=2 (not recovered) E+=3,
                    if bank.l>fireSaleBankrupt:
                        bank.getLender().E += fireSaleBankrupt
                        bank.getLender().B += (bank.l-fireSaleBankrupt)
                    # l= 5
                    # fireSaleBankrupt = 7 -> B=0, E+=5, all recovered
                    else:
                        bank.getLender().E += bank.l
                    Status.debug("repay",
                          f"t={bank.getId()} bankrupted (should return {loanToReturn:.3f}, ΔD={bank.ΔD:.3f} and L={bank.L:.3f})")
                    bank.replaceBank()
        else:
            # if no previous loan (bank.l=0), but also in this new shock bank.d>0, we have not enough money and we should fire sales!
            # (no loan at this moment)
            if bank.d<0:
                weNeedToSell = bank.d
                firesaleCost = weNeedToSell / Config.ρ
                # the fire sale is enough:
                if firesaleCost <= bank.L:
                    bank.C = 0
                    bank.L -= firesaleCost
                    bank.E = bank.L - bank.D
                    Status.debug("repay",
                        f"{bank.getId()} fire sales={firesaleCost:.3f} to pay {bank.d:.3f} (C=0,L={bank.L:.3f},E={bank.E:.3f}) in second shock)")
                else:
                    # the fire sale is not enough: could this really happen? TODO
                    Status.debug("repay",
                        f"t={bank.getId()} bankrupted (should fire sale {weNeedToSell:.3f} but L={bank.L})")
                    bank.replaceBank()


        # let's balance results: the solution is to decrease or increase E
        if bank.C + bank.L != bank.D + bank.E:
            bank.E = bank.C+ bank.L - bank.E
            Status.debug("repay",f"{bank.getId()} modifies E={bank.C:.3f}")

    Status.debugBanks(info="After payments")




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

    maxλ = max(Model.banks, key=lambda i: i.λ).λ
    for bank in Model.banks:
        bank.h = bank.λ / maxλ
        bank.A = bank.L / bank.λ + bank.D

    # determine c (lending capacity) for all other banks (to whom give loans):
    for bank in Model.banks:
        bank.c = []
        for i in range(Config.N):
            c = 0 if i == bank.id else (1 - Model.banks[i].h) * Model.banks[i].A
            bank.c.append(c)

    # (equation 6)
    minr = Config.r_i0 * 1000
    for bank_i in Model.banks:
        for j in range(Config.N):
            try:
                if j == bank_i.id:
                    bank_i.r[j] = None
                else:
                    bank_i.r[j] = (Config.Χ * Model.banks[j].A -
                                   Config.Φ * Model.banks[j].A -
                                   (1 - Model.banks[j].p) *
                                   (Config.ξ * Model.banks[j].A - bank_i.c[j])) \
                                  / (Model.banks[j].p * bank_i.c[j])
            # the first t=1, maybe t=2, the shocks have not affected enough to use L (only C), so probably
            # L and E are equal for all banks, and so maxλ=anyλ and h=1 , so cij=(1-1)A=0, and r division
            # by zero -> solution then is to use still r_i0:
            except ZeroDivisionError:
                bank_i.r[j] = Config.r_i0
            if bank_i.r[j] and bank_i.r[j] < minr:
                minr = bank_i.r[j]

    Status.debug("links",f"maxE={maxE:.3f} maxC={maxC:.3f} maxλ={maxλ:.3f} minr={minr:.3f} ŋ={Model.ŋ:.3f}")

    # (equation 7)
    for bank_i in Model.banks:
        loginfo = ""
        for j in range(Config.N):
            if j != bank_i.id:
                ##TODO print(maxC,bank_i.r[j]) --> maxC es cero si Config.N = 3
                bank_i.π[j] = Model.ŋ * (bank.C / maxC) + (1 - Model.ŋ) * (minr / bank_i.r[j])
                loginfo += f"{j}:{bank_i.π[j]:.3f},"
            else:
                bank_i.π[j] = None
        Status.debug("links", f"{bank_i.getId()} π=[{loginfo}]")

    # we can now break old links and set up new lenders, using probability P
    # (equation 8)
    for bank in Model.banks:
        possible_lender  = bank.newLender()
        possible_lender_π= Model.banks[possible_lender].π[bank.id]
        current_lender_π = bank.getLender().π[bank.id]
        bank.P = 1 / (1 + math.exp(-Config.β * ( possible_lender_π - current_lender_π ) ))

        if bank.P >= 0.5:
            bank.lender = possible_lender
            Status.debug("links", f"{bank.getId()} new lender is {possible_lender} with %{bank.P:.3f} ( {possible_lender_π:.3f} - {current_lender_π:.3f} )")
        else:
            Status.debug("links", f"{bank.getId()} maintains lender {bank.getLender().getId()} with %{1-bank.P:.3f}")

def determineMu():
    pass


# %%


class Statistics:
    bankruptcy = []
    bestLender = []
    bestLenderClients = []
    liquidity = []

    @staticmethod
    def reset():
        Statistics.bankruptcy = [0 for i in range(Config.T)]
        Statistics.bestLender = [-1 for i in range(Config.T)]
        Statistics.bestLenderClients = [0 for i in range(Config.T)]
        Statistics.liquidity = [0 for i in range(Config.T)]

    @staticmethod
    def computeBestLender():

        lenders = {}
        for bank in Model.banks:
            if bank.lender in lenders:
                lenders[bank.lender] += 1
            else:
                lenders[bank.lender] = 1
        best = -1
        bestValue = -1
        for lender in lenders.keys():
            if lenders[lender]>bestValue:
                best = lender
                bestValue = lenders[lender]

        Statistics.bestLender[ Model.t ] = best
        Statistics.bestLenderClients[ Model.t ] = bestValue

    @staticmethod
    def computeLiquidity():
        total = 0
        for bank in Model.banks:
            total += bank.C
        Statistics.liquidity[ Model.t ] = total


class Status:
    logger = logging.getLogger("model")
    modules= []

    [Config.r_i0 for i in range(Config.N)]

    @staticmethod
    def debugBanks(details: bool = True, info: str = ''):
        if info:
            info += ': '
        for bank in Model.banks:
            text = f"{info}{bank.getId()} C={bank.C:.3f} L={bank.L:.3f} | D={bank.D:.3f} E={bank.E:.3f}"
            if not details and hasattr(bank, 'ΔD'):
                text += f" ΔD={bank.ΔD:.3f}"
            if details and hasattr(bank, 'l'):
                text += f" s={bank.s:.3f} d={bank.d:.3f} l={bank.l:.3f}"
            Status.debug("-----",text)

    @staticmethod
    def getLevel(option):
        try:
            return getattr(logging, option.upper())
        except:
            logging.error(f" '--log' must contain a valid logging level and {option.upper()} is not.")
            sys.exit(-1)
            return None

    @staticmethod
    def debug(module,text):
        if Status.modules == [] or module in Status.modules:
            Status.logger.debug(f"t={Model.t:03}/{module} {text}")

    @staticmethod
    def info(module,text):
        if Status.modules == [] or module in Status.modules:
            Status.logger.info(f" t={Model.t:03}/{module} {text}")

    @staticmethod
    def error(module,text):
        Status.logger.error(f"t={Model.t:03}/{module} {text}")

    @staticmethod
    def runInteractive(log: str = typer.Option('ERROR', help="Log level messages (ERROR,DEBUG,INFO...)"),
                       modules: str = typer.Option(None, help=f"Log only this modules (separated by ,)"),
                       logfile: str = typer.Option(None, help="File to send logs to"),
                       n: int = typer.Option(Config.N, help=f"Number of banks"),
                       t: int = typer.Option(Config.T, help=f"Time repetitions")):
        """
        Run interactively the model
        """
        # https://typer.tiangolo.com/
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        Status.logLevel = Status.getLevel(log.upper())
        Status.logger.setLevel(Status.logLevel)
        if t != Config.T:
            Config.T = t
        if n != Config.N:
            Config.N = n
        Status.modules = modules.split(",") if modules else []
        if logfile:
            fh = logging.FileHandler(logfile, 'a', 'utf-8')
            fh.setLevel(Status.logLevel)
            fh.setFormatter(formatter)
            Status.logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setLevel(Status.logLevel)
            ch.setFormatter(formatter)
            Status.logger.addHandler(ch)
        Status.run()

    @staticmethod
    def run():
        Statistics.reset()
        Model.initilize()
        Model.doSimulation()
        Graph.generate()

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

class Graph:
    @staticmethod
    def bankruptcies():
        plt.clf()
        xx = []
        yy = []
        for i in range(Config.T):
            xx.append(i)
            yy.append(Statistics.bankruptcy[i])
        plt.plot(xx, yy, 'b-')
        plt.ylabel("num of bankruptcies")
        plt.xlabel("t")
        plt.title("Bankrupted firms")
        plt.show() if Status.isNotebook() else plt.savefig("output/bankrupted.svg")

    @staticmethod
    def liquidity():
        plt.clf()
        xx = []
        yy = []
        for i in range(Config.T):
            xx.append(i)
            yy.append(Statistics.liquidity[i])
        plt.plot(xx, yy, 'b-')
        plt.ylabel("liquidity")
        plt.xlabel("t")
        plt.title("C of all banks")
        plt.show() if Status.isNotebook() else plt.savefig("output/liquidity.svg")

    @staticmethod
    def bestLender():
        plt.clf()
        fig,ax = plt.subplots()
        xx = []
        yy = []
        yy2= []
        for i in range(Config.T):
            xx.append(i)
            yy.append(Statistics.bestLender[i])
            yy2.append(Statistics.bestLenderClients[i])
        ax.plot(xx, yy, 'b-')
        ax.set_ylabel("Best Lender",color='blue')
        ax.set_xlabel("t")
        ax2 = ax.twinx()
        ax2.plot(xx,yy2,'r-')
        ax2.set_ylabel("Best Lender # clients",color='red')
        plt.title("Best lender and # clients")
        plt.show() if Status.isNotebook() else plt.savefig("output/best_lender.svg")

    @staticmethod
    def generate():
        Graph.bankruptcies()
        Graph.liquidity()
        Graph.bestLender()

# %%


if Status.isNotebook():
    Status.run()
else:
    if __name__ == "__main__":
        typer.run(Status.runInteractive)
