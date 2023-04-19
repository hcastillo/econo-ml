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
    T: int = 1000    # time (1000)
    N: int = 50      # number of banks (50)

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
        # -> la matriz de creditos hasta t=10, para generar heterogeneidad
        for Model.t in range(Config.T):
            # -----> matrix de creditos deberia de ir aqui
            doShock("First shock")
            doLoans()
            doShock("Second shock")
            doRepayments()
            Statistics.computeLiquidity()
            Statistics.computeBestLender()
            determineMu()
            setupLinks()
        Statistics.finalInfo()

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
        self.failed = False
        # interbank rates
        self.lender = None
        self.lender = self.newLender()
        # TODO self.R = Config.ȓ * Config.D_i0

    def replaceBank(self):
        self.failures += 1
        self.__assign_defaults__()
        Statistics.bankruptcy[Model.t] += 1

    def __doBankrupcy__(self,phase):
        self.failed = True
        whatWeObtainAtFiresaleAll = self.L * Config.ρ
        badDebt = self.l - whatWeObtainAtFiresaleAll
        if badDebt > 0:
            self.getLender().B += badDebt
            self.getLender().E += whatWeObtainAtFiresaleAll
            Status.debug(phase,
                         f"t={self.getId()} bankrupted (B={badDebt:.3f}, recovered E={whatWeObtainAtFiresaleAll:.3f})")
        else:
            self.getLender().E = self.l  # we don't recover more than the loan if it is
            Status.debug(phase,
                         f"t={self.getId()} bankrupted (recovered all as E={whatWeObtainAtFiresaleAll:.3f})")
        self.getLender().s -= self.l  # the loan is not paid correctly, but we remove it

    def doFiresalesL( self,amountToSell, reason, phase):
        costOfSell = amountToSell / Config.ρ
        self.C = 0
        if costOfSell > self.L:
            self.__doBankrupcy__(phase)
            Status.debug("loans",
                         f"{self.getId()} cost fire sale {costOfSell:.3f} > L{self.L:.3f}: {reason}")
        else:
            self.L -= costOfSell
            Status.debug("loans",
                         f"{self.getId()} fire sale {amountToSell:.3f} , cost {costOfSell:.3f}: {reason}")


def doShock(whichShock):
    # (equation 2)
    for bank in Model.banks:
        newD = bank.D * (Config.µ + Config.ω * random.random())
        bank.ΔD = newD - bank.D
        bank.D  = newD
        if bank.ΔD > 0:
            bank.C += bank.ΔD
        Statistics.incrementD[Model.t] += bank.ΔD
    Status.debugBanks(details=False,info=whichShock)


def doLoans():
    for bank in Model.banks:
        bank.B = 0
        if bank.ΔD + bank.C > 0:
            if bank.ΔD>=0:
                bank.s = bank.ΔD + bank.C  # lender
            else:
                # it will not lend money due to not increase
                bank.s = 0
            bank.d = 0 # it will not need to borrow
        else:
            bank.d = abs(bank.ΔD + bank.C)  # it will need money
            bank.s = 0 # no lender this time

    for bank in Model.banks:
        # decrement in which we should borrow
        if bank.d > 0:
            if bank.getLender().d > 0:
                # if the lender has no increment then NO LOAN could be obtained: we fire sale L:
                bank.doFiresalesL( bank.d,"lender gives no money","loans" )
                bank.l = 0
            else:
                # if the lender can give us money, but not enough to cover the loan we need also fire sale L:
                if bank.d > bank.getLender().s:
                    bank.doFiresalesL( bank.d - bank.getLender().s,
                                       f"lender gives {bank.getLender().s:.3f} and we need {bank.d:.3f}","loans")
                    bank.l = bank.getLender().s
                    bank.getLender().s = 0
                else:
                    bank.l = bank.d
                    bank.getLender().s -= bank.d  # bank.d

        # the shock can be covered by own capital
        else:
            bank.l = 0
            if bank.ΔD < 0:  # increment D negative -> decreases C also
                bank.C += bank.ΔD
                Status.debug("loans",
                    f"{bank.getId()} loses ΔD={bank.ΔD:.3f}, covered by capital, now C={bank.C:.3f}")
    # Status.debugBanks(info='Loans after')

def doRepayments():
    for bank in Model.banks:
        # first we should pay D (if ΔD < 0): in this moment, only with own C
        # or doing firesales:
        if bank.ΔD < 0:
            if bank.C + bank.ΔD >= 0:
                bank.C -= bank.ΔD
                Status.debug("repay",
                         f"{bank.getId()} second shock ΔD={bank.ΔD:.3f} paid with cash, now C={bank.C:.3f}")
            else:
                bank.doFiresalesL(abs(bank.ΔD + bank.C),f"second shock ΔD={bank.ΔD:.3f} but C={bank.C:.3f}","repay")

        # now, even ΔD<0 nor ΔD>0, let's return previous loan bank.l (if it exists), but only
        # if the bank is not bankrupted!
        if bank.l > 0 and not bank.failed:
            loanProfits  = bank.getLoanInterest() * bank.l
            loanToReturn = bank.l + loanProfits
            # (equation 3)
            if loanToReturn > bank.C:
                weNeedToSell = loanToReturn - bank.C
                bank.doFiresalesL( weNeedToSell,f"return loan and interest {loanToReturn:.3f} is higher than C={bank.C:.3f}","repay")
            # the firesales of line above could bankrupt the bank, if not, we pay "normally" the loan:
            if not bank.failed:
                bank.C -= loanToReturn
                bank.E -= loanProfits
                bank.getLender().s -= bank.l       # we reduce the  's' => the lender could have more loans
                bank.getLender().C += bank.l       # we return the loan
                bank.getLender().E += loanProfits  # the profits are paid as E
                Status.debug("repay",
                    f"{bank.getId()} pays loan {loanToReturn:.3f} (E={bank.E:.3f},C={bank.C:.3f}) to lender {bank.getLender().getId()} (ΔE={loanProfits:.3f},ΔC={bank.l:.3f})")

        if bank.failed:
            bank.replaceBank()

        # let's balance results: if we have fire sale L, surely it's time to decrease E
        if bank.C + bank.L != bank.D + bank.E:
            bank.E = bank.C + bank.L - bank.D
            if bank.E <= 0:
                bank.C = -bank.E-2
                bank.E = 2 #TODO
                Status.debug("repay", f"{bank.getId()} modifies C={bank.C:.3f}")
            else:
                Status.debug("repay", f"{bank.getId()} modifies E={bank.E:.3f}")

    Status.debug("repay",f"this step ΔD={Statistics.incrementD[Model.t]:.3f} and failures={Statistics.bankruptcy[Model.t]}")
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
        print(bank.E)
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
    incrementD = []

    @staticmethod
    def reset():
        Statistics.bankruptcy = [0 for i in range(Config.T)]
        Statistics.bestLender = [-1 for i in range(Config.T)]
        Statistics.bestLenderClients = [0 for i in range(Config.T)]
        Statistics.liquidity = [0 for i in range(Config.T)]
        Statistics.incrementD = [0 for i in range(Config.T)]

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

    @staticmethod
    def finalInfo():
        total = 0
        for i in Statistics.incrementD:
            total += i
        Status.debug("final",f"after {Config .T} we have Σ ΔD={total}")

class Status:
    logger = logging.getLogger("model")
    modules= []

    ## [Config.r_i0 for i in range(Config.N)]

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
    def bestLenderBokeh():
        from bokeh.plotting import figure, show

        xx = []
        yy = []
        yy2 = []
        for i in range(Config.T):
            xx.append(i)
            yy.append(Statistics.bestLender[i])
            yy2.append(Statistics.bestLenderClients[i])
        p = figure(title="Best Lender", x_axis_label='x', y_axis_label='y',
                   sizing_mode="stretch_width",
                   height=350)
        p.line(xx, yy, legend_label="Best lender id", color="blue", line_width=2)
        p.line(xx, yy2, legend_label="Best lender num clients", color="red", line_width=2)
        show(p)

    @staticmethod
    def generate():
        Graph.bankruptcies()
        Graph.liquidity()
        Graph.bestLender()
        Graph.bestLenderBokeh()


# %%


if Status.isNotebook():
    Status.run()
else:
    if __name__ == "__main__":
        typer.run(Status.runInteractive)
