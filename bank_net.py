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
    α: float = 0.1    # below this level of E or D, we will bankrupt the bank

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
        Model.banks = []
        t = 0
        for i in range(Config.N):
            Model.banks.append(Bank(i))

    @staticmethod
    def doSimulation():
        # initially all banks have a random lender
        # -> la matriz de creditos hasta t=10, para generar heterogeneidad
        for Model.t in range(Config.T):
            doShock("shock1")
            doLoans()
            doShock("shock2")
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

    def getId(self,short:bool=False):
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
            self.r = [Config.r_i0 for i in range(Config.N)]
            self.π = [ 0 for i in range(Config.N) ]
            self.r[self.id] = None  # to ourselves, we don't want to lend anything
            # if it's just created, only not to be ourselves is enough
            newvalue = random.randrange(Config.N - 1 )
        else:
            # if we have a previous lender, new should not be the same
            newvalue = random.randrange(Config.N - 2 if Config.N>2 else 1 )

        if Config.N == 2:
            newvalue = 1 if self.id == 0 else 0
        else:
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
        self.B = 0
        self.failed = False
        # interbank rates
        self.lender = None
        self.lender = self.newLender()

        self.activeBorrowers = {}
        # TODO self.R = Config.ȓ * Config.D_i0

    def replaceBank(self):
        self.failures += 1
        self.__assign_defaults__()

    def __doBankruptcy__(self,phase):
        self.failed = True
        Statistics.bankruptcy[Model.t] += 1
        whatWeObtainAtFiresaleAll = self.L * Config.ρ            # we firesale what we have
        recovered = whatWeObtainAtFiresaleAll - self.D           # we should pay D to clients
        if recovered < 0:
            recovered = 0
        if recovered > self.l:
            recovered = self.l

        badDebt = self.l - recovered                             # the fire sale minus paying D: what the lender recovers
        if badDebt > 0:
            self.paidLoan = recovered
            self.getLender().B += badDebt
            self.getLender().E -= badDebt
            self.getLender().C += recovered
            Status.debug(phase,
                         f"{self.getId()} bankrupted (fire sale={whatWeObtainAtFiresaleAll:.3f},recovers={recovered},paidD={self.D})(lender{self.getLender().getId(short=True)}.ΔB={badDebt:.3f},ΔC={recovered:.3f})")
        else:
            # self.l=0 no current loan to return:
            if self.l>0:
                self.paidLoan = self.l       # the loan was paid, not the interest
                self.getLender().C += self.l  # lender not recovers more than loan if it is
                Status.debug(phase,
                         f"{self.getId()} bankrupted (lender{self.getLender().getId(short=True)}.ΔB=0,ΔC={recovered:.3f}) (paidD={self.l:.3f})")
        self.D = 0
        # the loan is not paid correctly, but we remove it
        if self.id in self.getLender().activeBorrowers:
            self.getLender().s -= self.l
            del self.getLender().activeBorrowers[ self.id ]
        else:
            # TODO . que pasa si quiebra el banco que TIENE prestamos, y no el prestado
            pass

    def doFiresalesL( self,amountToSell, reason, phase):
        costOfSell = amountToSell / Config.ρ
        recoveredE = costOfSell * ( 1-Config.ρ )
        ##self.C = 0
        if costOfSell > self.L:
            Status.debug(phase,
                         f"{self.getId()} impossible fire sale sellL={costOfSell:.3f} > L={self.L:.3f}: {reason}")
            return self.__doBankruptcy__(phase)
        else:
            self.L -= costOfSell
            self.E -= recoveredE

            if self.L <= Config.α:
                Status.debug(phase,
                             f"{self.getId()} new L={self.L:.3f} makes bankruptcy of bank: {reason}")
                return self.__doBankruptcy__(phase)
            else:
                if self.E <= Config.α:
                    Status.debug(phase,
                                 f"{self.getId()} new E={self.E:.3f} makes bankruptcy of bank: {reason}")
                    return self.__doBankruptcy__(phase)
                else:
                    Status.debug(phase,
                         f"{self.getId()} fire sale sellL={amountToSell:.3f} at cost {costOfSell:.3f} reducing E={recoveredE:.3f}: {reason}")
                    return amountToSell

def doShock(whichShock):
    # (equation 2)
    for bank in Model.banks:
        bank.newD = bank.D * (Config.µ + Config.ω * random.random())
        bank.ΔD = bank.newD - bank.D
        bank.D  = bank.newD
        if bank.ΔD >= 0:
            bank.C += bank.ΔD
            # if "shock1" then we can be a lender:
            if whichShock=="shock1":
                bank.s = bank.C
            bank.d = 0       # it will not need to borrow
            if bank.ΔD>0:
                Status.debug(whichShock,
                         f"{bank.getId()} wins ΔD={bank.ΔD:.3f}")
        else:
            # if "shock1" then we cannot be a lender: we have lost deposits
            if whichShock=="shock1":
                bank.s = 0
            if bank.ΔD + bank.C >= 0:
                bank.d = 0  # it will not need to borrow
                bank.C += bank.ΔD
                Status.debug(whichShock,
                    f"{bank.getId()} loses ΔD={bank.ΔD:.3f}, covered by capital")
            else:
                bank.d = abs(bank.ΔD + bank.C)  # it will need money
                Status.debug(whichShock,
                    f"{bank.getId()} loses ΔD={bank.ΔD:.3f} but has only C={bank.C:.3f}")
                bank.C = 0  # we run out of capital

        bank.B = 0  # no bad debt we have from previous step
        Statistics.incrementD[Model.t] += bank.ΔD
    Status.debugBanks(details=False,info=whichShock)


def doLoans():
    for bank in Model.banks:
        # decrement in which we should borrow
        if bank.d > 0:
            if bank.getLender().d > 0:
                # if the lender has no increment then NO LOAN could be obtained: we fire sale L:
                bank.doFiresalesL( bank.d,"lender has no money to borrow us","loans" )
                bank.l = 0
            else:
                # if the lender can give us money, but not enough to cover the loan we need also fire sale L:
                if bank.d > bank.getLender().s:
                    bank.doFiresalesL( bank.d - bank.getLender().s,
                                       f"lender.s={bank.getLender().s:.3f} but need d={bank.d:.3f}","loans")
                    # only if lender has money, because if it .s=0, all is obtained by fire sales:
                    if bank.getLender().s > 0:
                        bank.l = bank.getLender().s   # amount of loan (writed in the borrower)
                        bank.getLender().activeBorrowers[ bank.id ] = bank.getLender().s # amount of loan (writed in the lender)
                        bank.getLender().C -= bank.l  # amount of loan that reduces lender capital
                        bank.getLender().s = 0
                else:
                    bank.l = bank.d # amount of loan (writed in the borrower)
                    bank.getLender().activeBorrowers[ bank.id ] = bank.d # amount of loan (writed in the lender)
                    bank.getLender().s -= bank.d # the loan reduces our lender's capacity to borrow to others
                    bank.getLender().C -= bank.d # amount of loan that reduces lender capital
                    Status.debug("loans",
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
                Status.debug("loans",
                             f"{bank.getId()} has a total of {len(bank.activeBorrowers)} loans with [{list_borrowers[:-1]}] of l={amount_borrowed}")


def doRepayments():
    # first all borrowers must pay their loans:
    for bank in Model.banks:
        if bank.l > 0:
            loanProfits = bank.getLoanInterest() * bank.l
            loanToReturn = bank.l + loanProfits
            # (equation 3)
            if loanToReturn > bank.C:
                weNeedToSell  = loanToReturn - bank.C
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
                Status.debug("repay",
                             f"{bank.getId()} pays loan {loanToReturn:.3f} (E={bank.E:.3f},C={bank.C:.3f}) to lender {bank.getLender().getId()} (ΔE={loanProfits:.3f},ΔC={bank.l:.3f})")

    # now  when ΔD<0 it's time to use Capital or sell L again (now we have the loans cancelled, or the bank bankrputed):
    for bank in Model.banks:
        if bank.d>0 and not bank.failed:
            bank.doFiresalesL(bank.d,f"fire sales due to not enough C","repay")




        # let's balance results: if we have fire sale L, surely it's time to decrease E
        #if bank.C + bank.L != bank.D + bank.E:
        #    diff = bank.C+bank.L-bank.D-bank.E
        #    bank.E = bank.C + bank.L - bank.D
        #    if bank.E <= 0:
        #        bank.C = -bank.E-2
        #        bank.E = 2 #TODO
        #        Status.debug("repay", f"{bank.getId()} modifies C={bank.C:.3f} {diff}")
        #    else:
        #        Status.debug("repay", f"{bank.getId()} modifies E={bank.E:.3f} {diff}")
    for bank in Model.banks:
        bank.activeBorrowers = {}
        if bank.failed:
            bank.replaceBank()
    Status.debugBanks()
    Status.debug("repay",f"this step ΔD={Statistics.incrementD[Model.t]:.3f} and failures={Statistics.bankruptcy[Model.t]}")




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

class Status:
    logger = logging.getLogger("model")
    modules= []

    ## [Config.r_i0 for i in range(Config.N)]

    @staticmethod
    def __get_string_debug_banks__(details,bank):
        text = f"{bank.getId():8} C={bank.C:5.2f} L={bank.L:5.2f}"
        text += f" B={bank.B:5.2f}" if bank.B else "        "
        if details and hasattr(bank, 'd') and bank.d:
            text += f" d={bank.d:5.2f}"
        else:
            text += "        "
        text += f" | D={bank.D:5.2f} E={bank.E:5.2f}"
        #if details and hasattr(bank, 'l') and bank.l:
        #    text += f" l={bank.l:5.2f}"
        #else:
        #    text += "        "
        if details and hasattr(bank, 's') and bank.s:
            text += f" s={bank.s:5.2f}"
        else:
            text += "        "
        #if details and hasattr(bank, 'ΔD') and bank.ΔD:
        #    text += f" ΔD={bank.ΔD:6.2f}"
        #else:
        #    text += "        "
        text += f" lender={bank.getLender().getId(short=True)}"
        return text

    @staticmethod
    def debugBanks(details: bool = True, info: str = ''):
        for bank in Model.banks:
            if not info:
                info="-----"
            Status.debug(info,Status.__get_string_debug_banks__(details,bank))

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
            Status.logger.debug(f"t={Model.t:03}/{module:6} {text}")

    @staticmethod
    def info(module,text):
        if Status.modules == [] or module in Status.modules:
            Status.logger.info(f" t={Model.t:03}/{module:6} {text}")

    @staticmethod
    def error(module,text):
        Status.logger.error(f"t={Model.t:03}/{module:6} {text}")

    @staticmethod
    def defineLog( log:str,logfile:str='',modules:str=''):
        Status.modules = modules.split(",") if modules else []
        # https://typer.tiangolo.com/
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        Status.logLevel = Status.getLevel(log.upper())
        Status.logger.setLevel(Status.logLevel)
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

    @staticmethod
    def runInteractive(log: str = typer.Option('ERROR', help="Log level messages (ERROR,DEBUG,INFO...)"),
                       modules: str = typer.Option(None, help=f"Log only this modules (separated by ,)"),
                       logfile: str = typer.Option(None, help="File to send logs to"),
                       n: int = typer.Option(Config.N, help=f"Number of banks"),
                       t: int = typer.Option(Config.T, help=f"Time repetitions")):
        """
        Run interactively the model
        """
        if t != Config.T:
            Config.T = t
        if n != Config.N:
            Config.N = n
        Status.defineLog(log,logfile,modules)
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
