# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:08:29 2023

@author: hector@bith.net
"""

import random
import logging
import math
import typer

# from utils.utilities import readConfigYaml, saveConfigYaml, format_tousands, generate_logger, GeneratePathFolder


class Config:
    T = 1000  # time (1000)
    N = 50    # number of banks
    
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
            Status.logger.debug(f"Iteration {Model.t}")

        
       
    

#%%

class Bank:
    L = Config.L_i0
    C = Config.C_i0
    D = Config.D_i0
    E = Config.E_i0
    R = Config.ȓ * Config.D_i0
    
    
    def getLender(self):
        return Model.banks[self.id-1]
    
    def getId(self):
        return f"bank#{self.id}"
    
    def __init__(self,id):
        self.id = id
    

def setupLinks():
    pass

def doShocks():
    # first shock: (equation 2)    
    for bank in Model.banks:
        bank.newD = bank.D * ( Config.µ + Config.ω * random.random())
    
    for bank in Model.banks:
        bank.incrD = bank.newD - bank.D
        # decrement -> should reduce C
        if bank.incrD + bank.C < 0: 
            
            bank.s = bank.getLender().incrD if bank.getLender().incrD>0 else 0 
            bank.d = -bank.incrD
            bank.l = bank.d if bank.d<bank.s else bank.s
            bank.incrL = (bank.d-bank.s)/Config.ρ
            #Status.logger.debug(f"{bank.getId()} s={bank.s} d={bank.d} l={bank.l} incrD={bank.incrD}")
            
            if bank.incrL > 0:
                pass # Status.logger.debug(f"{bank.getId()} should sell L={bank.incrL} to cover {bank.incrD} as lender only gives {bank.s}")
            else:
                Status.logger.debug(f"{bank.getId()} obtains loan from {bank.getLender().getId()} of {bank.s}")
                
            
    # second shock: (equation 2)
    for bank in Model.banks:
        bank.newD = bank.D * ( Config.µ + Config.ω * random.random())
    
    # 
    
    
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
    def runInteractive(debug:bool= typer.Option( False, help="Log DEBUG messages" )):
        """
        Run interactively the model
        """        
        # https://typer.tiangolo.com/
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)    
        if debug:
            Status.logger.setLevel(logging.DEBUG)
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

    
    
    
    

