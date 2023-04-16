# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:08:29 2023
@author: hector@bith.net
"""

import random
import logging
import math
import typer
import sys
#TODO from utils.utilities import readConfigYaml, saveConfigYaml, format_tousands, generate_logger, GeneratePathFolder

class Config:
    T = 10  # 1000  # time (1000)
    N = 5  # 50    # number of banks

    d = 1  # out-degree

    # TODO ȓ = 0.02  # reserves percentage

    c = 1  # parameter bankruptcy cost equation
    α = 0.08  # alpha, ratio equity-loan
    g = 1.1  # variable cost
    # markdown interest rate (the higher it is, the monopolistic power of banks)

    d = 100  # location cost
    e = 0.1  # sensivity

    µ = 0.7  # pi
    ω = 0.55  # omega

    # screening costs
    Φ = 0.025  # phi
    Χ = 0.015  # ji
    # liquidation cost of collateral
    ξ = 0.3  # xi
    ρ = 0.3  # fire-sale price

    β = 5  # intensity of breaking the connection
    prob_initially_isolated = 0.25

    # banks initial parameters
    L_i0 = 120  # long term assets
    C_i0 = 30  # capital
    D_i0 = 135  # deposits
    E_i0 = 15  # equity
    r_i0 = 0.02  # initial rate


class Model:
    banks = []
    t = 0

    ŋ = 0.5  # eta : policy

    @staticmethod
    def initilize():
        random.seed(40579)
        for i in range(Config.N):
            Model.banks.append(Bank(i))

    @staticmethod
    def doSimulation():
        doShock()
        for Model.t in range(Config.T):
            doLoans()
            doShock()
            doRepayments()
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



def doShock():
    # (equation 2)
    for bank in Model.banks:
        bank.B = 0  # bad debt
        bank.newD = bank.D * (Config.µ + Config.ω * random.random())
        bank.ΔD = bank.newD - bank.D
    Status.debugBanks(details=False)


def doLoans():
    for bank in Model.banks:
        if bank.ΔD + bank.C > 0:
            bank.s = bank.ΔD + bank.C  # lender
            bank.d = 0
        else:
            bank.d = abs(bank.ΔD + bank.C)  # borrower
            bank.s = 0
        bank.D = bank.newD

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

        # increment of deposits or decrement covered by own capital 
        else:
            bank.l = 0
            bank.sellL = 0
            if bank.ΔD < 0:
                bank.C += bank.ΔD
                Status.debug("loans",
                    f"{bank.getId()} loses ΔD={bank.ΔD:.3f}, covered by capital, now C={bank.C:.3f}")
    Status.debugBanks(info='Loans after')


def doRepayments():
    for bank in Model.banks:
        # let's return previous loan bank.l (if it exists):  
        if bank.l > 0:
            loanProfits = bank.getLoanInterest() * bank.l
            loanToReturn = bank.l + loanProfits
            # (equation 3)
            # i) the change is sufficent to repay the principal and the interest
            if bank.ΔD - loanToReturn > 0:
                bank.E -= loanToReturn
                bank.getLender().s += bank.l
                bank.getLender().E += loanProfits
                Status.debug("repay",
                    f"{bank.getId()} pays loan {loanToReturn:.3f} to {bank.getLender().getId()} (lender also ΔE={loanProfits:.3f}), now E={bank.C:.3f}")

            # ii) not enough increment: then firesale of L to cover the loan to return and interests
            else:
                bank.sellL = -(bank.ΔD - loanToReturn) / Config.ρ
                # bankrupcy: not enough L to sell and cover the obligations:
                if bank.sellL > bank.L:
                    Status.logger.debug("repay",
                        f"t={bank.getId()} bankrupted (should return {loanToReturn:.3f}, ΔD={bank.ΔD:.3f} and L={bank.L})")
                    bank.getLender().B += bank.l - bank.L * (1 - Config.ρ)
                    bank.replaceBank()
                # the firesale covers the loan 
                else:
                    bank.L -= bank.sellL
                    bank.getLender().s += bank.l
                    bank.getLender().E += loanProfits
                    Status.debug("repay",
                        f"{bank.getId()} loses ΔL={bank.sellL:.3f} to return loan {loanToReturn:.3f} {bank.getLender().getId()} (lender also ΔE={loanProfits:.3f})")
        # let's balance results:
        if bank.C + bank.L != bank.D + bank.E:
            bank.C = bank.D + bank.E - bank.L
            Status.debug("repay",f"{bank.getId()} modifies capital and C={bank.C:.3f}")

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
            except:  # TODO: problema, como no toco E,C, esta división es cero la primera vez
                bank_i.r[j] = Config.r_i0
            if bank_i.r[j] and bank_i.r[j] < minr:
                minr = bank_i.r[j]

    Status.debug("links",f"maxE={maxE} maxC={maxC} maxλ={maxλ} minr={minr} ŋ={Model.ŋ}")

    # (equation 7)
    for bank_i in Model.banks:
        loginfo = ""
        for j in range(Config.N):
            if j != bank_i.id:
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
            Status.debug("links", f"{bank.getId()} new lender is {possible_lender} with %{bank.P:.3f} ( {possible_lender_π} - {current_lender_π} )")
        else:
            Status.debug("links", f"{bank.getId()} maintains lender {bank.getLender().getId()} with %{1-bank.P:.3f}")

def determineMu():
    pass


# %%


class Status:
    logger = logging.getLogger("model")
    modules= []
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
            Status.info("-----",text)

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
            fh = logging.FileHandler(logfile, 'w', 'utf-8')
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
        Model.initilize()
        Model.doSimulation()

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


if Status.isNotebook():
    Status.run()
else:
    if __name__ == "__main__":
        typer.run(Status.runInteractive)
