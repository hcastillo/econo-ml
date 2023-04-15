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

# from utils.utilities import readConfigYaml, saveConfigYaml, format_tousands, generate_logger, GeneratePathFolder


class Config:
    T = 10# 1000  # time (1000)
    N = 5 # 50    # number of banks
    
    d=  1     # out-degree
    
    prob_initially_isolated = 0.25
    
    ȓ = 0.02  # reserves (Basel)
    
    c = 1     # parameter bankruptcy cost equation
    α = 0.08  # alpha, ratio equity-loan
    g = 1.1   # variable cost
     # markdown interest rate (the higher it is, the monopolistic power of banks)
    λ = 0.3   # credit assets rate
    d = 100   # location cost
    e = 0.1   # sensivity

    # ŋ
    # 
    µ=0.7 # pi
    ω = 0.55 
    
    
    ρ = 0.3   # fire-sale price
    
    # banks initial parameters
    L_i0 = 120   # liability
    C_i0 = 30
    D_i0 = 135
    E_i0 = 15


    
class Model:    
    banks = []
    t = 0
    
    @staticmethod
    def initilize():
        random.seed(40579)
        for i in range(Config.N):
            Model.banks.append( Bank(i) )
        
    
    @staticmethod
    def doSimulation():
        for Model.t in range(Config.T):
            setupLinks()
            
            doShocks()
            doRepayments()
            
            removeAddBanks()
            
            determineMu()
             
        
       
    

#%%

class Bank:
    def getLender(self): #TODO 1
        idLender=int(self.id.split(".")[0])-1
        return Model.banks[idLender]
    
    def getLoanInterest(self): #TODO 2
        return 0.03
    
    def getId(self):
        return f"bank#{self.id}"
    
    def __init__(self,id):
        self.id =str(id)
        self.__assign_defaults__()
        
    def __assign_defaults__(self):
        self.L = Config.L_i0
        self.C = Config.C_i0
        self.D = Config.D_i0
        self.E = Config.E_i0
        self.R = Config.ȓ * Config.D_i0

    def replaceBank(self):
        parts = self.id.split(".")
        if len(parts)==2:
            self.id= parts[0]+"."+str(int(parts[1])+1)
        else:    
            self.id = self.id+".1"
        self.__assign_defaults__()
    

            
def setupLinks():
    pass

def doShocks():
    # first shock: (equation 2)            
    Status.logBanks(details=False)
    for bank in Model.banks:
        bank.B = 0  # bad debt
        bank.newD = bank.D * ( Config.µ + Config.ω * random.random())        
        bank.ΔD= bank.newD - bank.D
        if bank.ΔD + bank.C > 0: 
            bank.s= bank.ΔD + bank.C      # lender
            bank.d= 0
        else:
            bank.d= abs(bank.ΔD + bank.C) # borrower
            bank.s= 0
        bank.D = bank.newD

    for bank in Model.banks:
        # decrement in which we should borrow
        if bank.ΔD + bank.C < 0: 
            #TODO getLender
            if bank.d>bank.getLender().s:                
                bank.l = bank.getLender().s
                bank.getLender().s = 0
                bank.sellL = (bank.d-bank.l)/Config.ρ
            else:
                bank.l = bank.d
                bank.getLender().s -= bank.d # bank.d
                bank.sellL = 0
            
            if bank.sellL > 0:
                bank.L -= bank.sellL
                bank.C = 0
                Status.logger.debug(f"t={Model.t:03},SHOCK1: {bank.getId()} firesales L={bank.sellL:.3f} to cover ΔD={bank.ΔD:.3f} as {bank.getLender().getId()} gives {bank.l:.3f} and C={bank.C:.3f}")
            else:
                bank.C -= bank.d
                Status.logger.debug(f"t={Model.t:03},SHOCK1: {bank.getId()} borrows {bank.l:.3f} from {bank.getLender().getId()} (who still has {bank.getLender().s:.3f}) to cover ΔD={bank.ΔD:.3f} and C={bank.C:.3f}")
                
        # increment of deposits or decrement covered by own capital 
        else:
            bank.l = 0
            bank.sellL = 0
            if bank.ΔD<0:
                bank.C += bank.ΔD
                Status.logger.debug(f"t={Model.t:03},SHOCK1: {bank.getId()} loses ΔD={bank.ΔD:.3f}, covered by capital, now C={bank.C:.3f}")
        
    Status.logBanks(info='After SHOCK1')
    # second shock: (equation 2)
    # bank should return bank.s + costs of bank.sellL + possible shock 2 effects
    for bank in Model.banks:
        bank.newD = bank.D * ( Config.µ + Config.ω * random.random())        
        bank.ΔD= bank.newD - bank.D        
        bank.D = bank.newD
           
    # first return previous loan bank.l (if it exists):    
    for bank in Model.banks:
        if bank.l>0:            
            loanProfits  = bank.getLoanInterest()*bank.l
            loanToReturn = bank.l + loanProfits
            # (equation 3)
            # i) the change is sufficent to repay the principal and the interest
            if bank.ΔD - loanToReturn>0:
                bank.E -= loanToReturn
                bank.getLender().s += bank.l
                bank.getLender().E += loanProfits
                Status.logger.debug(f"t={Model.t:03},SHOCK2: {bank.getId()} pays loan {loanToReturn:.3f} to {bank.getLender().getId()} (lender also ΔE={loanProfits:.3f}), now E={bank.C:.3f}")
            
            # ii) not enough increment: then firesale of L to cover the loan to return and interests
            else:
                bank.sellL = -(bank.ΔD - loanToReturn)/Config.ρ                    
                # bankrupcy: not enough L to sell and cover the obligations:
                if bank.sellL > bank.L:
                    Status.logger.debug(f"t={Model.t:03},SHOCK2: {bank.getId()} bankrupted (should return {loanToReturn:.3f} and L={bank.L}")
                    bank.getLender().B += bank.l - bank.L*(1-Config.ρ)
                    bank.replaceBank()
                # the firesale covers the loan 
                else:                
                    bank.L -= bank.sellL
                    bank.getLender().s += bank.l 
                    bank.getLender().E += loanProfits
                    Status.logger.debug(f"t={Model.t:03},SHOCK2: {bank.getId()} loses ΔL={bank.sellL:.3f} to return loan {loanToReturn:.3f} {bank.getLender().getId()} (lender also ΔE={loanProfits:.3f})")
                    
    # and now let's see if with new shock ΔD we should reduce C or fail also (no loans now):
    for bank in Model.banks:
        if bank.ΔD + bank.C > 0:
            bank.s= +bank.ΔD + bank.C  # lender
            bank.d= 0
        else:
            bank.d= -bank.ΔD + bank.C  # borrower
            bank.s= 0
    Status.logBanks( info="After SHOCK2" )
    
def doRepayments():
    pass
def removeAddBanks():
    pass


def determineMu():
    pass


#%%


class Status:    
    logger = logging.getLogger("model")

    @staticmethod
    def logBanks(details:bool=True,info:str=''):
        for bank in Model.banks:
            text = f"t={Model.t:03}"
            if info:
                text += f",{info}:"
            text += f" {bank.getId()} C={bank.C:.3f} L={bank.L:.3f} | D={bank.D:.3f} E={bank.E:.3f}"
            if details and hasattr(bank,'l'):
                text += f" s={bank.s:.3f} d={bank.d:.3f} l={bank.l:.3f} ΔD={bank.ΔD:.3f}"
            Status.logger.debug(text)

    @staticmethod
    def getLevel( option ):
        try:
            return getattr(logging,option.upper())
        except:
            logging.error( f" '--log' must contain a valid logging level and {option.upper()} is not.")
            sys.exit(-1)
            return None
            
    @staticmethod
    def runInteractive(log:str= typer.Option( 'ERROR', help="Log level messages (ERROR,DEBUG,INFO...)" ), \
                       logfile:str=typer.Option(None,help="File to send logs to")):
        """
        Run interactively the model
        """        
        # https://typer.tiangolo.com/
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        Status.logger.setLevel( Status.getLevel(log.upper()))        
        if logfile:
            fh = logging.FileHandler( logfile, 'w', 'utf-8' )
            fh.setLevel( Status.getLevel(log.upper()))
            fh.setFormatter(formatter)
            Status.logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setLevel( Status.getLevel(log.upper()))
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


#%%


if Status.isNotebook():
    Status.run()
else:
    if __name__=="__main__":
        typer.run(Status.runInteractive)

    
    
    
    

