# -*- coding: utf-8 -*-
"""
LenderChange is a class used from interbank.py to control the change of lender in a bank.
   It contains different

@author: hector@bith.net
@date:   04/2023
"""
import random
import math
import numpy as np


class LenderChange:
    # parameter to control the change of guru: 0 then Boltzmann only, 1 everyone will move randomly
    γ: float = 0.5  # [0..1] gamma
    CHANGE_LENDER_IF_HIGHER = 0.5

    """ Call from Model.initialization()
    """

    def initialize_bank_relationships(self):
        pass

    """ Call at the end of each step before going to the next
    """

    def change_lender(self, model, bank, t):
        pass

    """ Call at the end of each step before going to the next
    """

    def new_lender(self, model, bank):
        pass


class Boltzman(LenderChange):
    """ changes lender for the bank 'bank' in instant 't'
    """

    def change_lender(self, model, bank, t):
        # sale de .new_lender()
        possible_lender = self.new_lender(model, bank)
        possible_lender_μ = model.banks[possible_lender].μ
        current_lender_μ = bank.getLender().μ

        # we can now break old links and set up new lenders, using probability P
        # (equation 8)
        boltzmann = 1 / (1 + math.exp(-model.config.β * (possible_lender_μ - current_lender_μ)))

        if t < 20:
            # bank.P = 0.35
            # bank.P = random.random()
            bank.P = boltzmann
            # option a) bank.P initially 0.35
            # option b) bank.P randomly
            # option c) with t<=20 boltzmann, and later, stabilize it
        else:
            bank.P_yesterday = bank.P
            # gamma is not sticky/loyalty, persistence of the previous attitude
            bank.P = self.γ * bank.P_yesterday + (1 - self.γ) * boltzmann

        if bank.P >= self.CHANGE_LENDER_IF_HIGHER:
            text_to_return = f"{bank.getId()} new lender is #{possible_lender} from #{bank.lender} with %{bank.P:.3f}"
            bank.lender = possible_lender
        else:
            text_to_return = f"{bank.getId()} maintains lender #{bank.lender} with %{1 - bank.P:.3f}"
        return text_to_return

    """ gives to the bank a random new lender. It's used initially and from change_lender() 
    """

    def new_lender(self, model, bank):
        # r_i0 is used the first time the bank is created:
        if bank.lender is None:
            bank.rij = np.full(model.config.N, model.config.r_i0, dtype=float)
            bank.rij[bank.id] = 0
            bank.r = model.config.r_i0
            bank.μ = 0
            # if it's just created, only not to be ourselves is enough
            new_value = random.randrange(model.config.N - 1)
        else:
            # if we have a previous lender, new should not be the same
            new_value = random.randrange(model.config.N - 2 if model.config.N > 2 else 1)

        if model.config.N == 2:
            new_value = 1 if bank.id == 0 else 0
        else:
            if new_value >= bank.id:
                new_value += 1
                if bank.lender is not None and new_value >= bank.lender:
                    new_value += 1
            else:
                if bank.lender is not None and new_value >= bank.lender:
                    new_value += 1
                    if new_value >= bank.id:
                        new_value += 1
        return new_value


class InitialStability(Boltzman):
    """ We define a Barabási–Albert graph with 1 edges for each node and we convert to a directed graph
    """

    def initialize_bank_relationships(self):
        g = nx.barabasi_albert_graph(nodos, 3)
        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=True)
        plt.show()


class ShockedMarkt(LenderChange):
    pass


if __name__ == "__main__":
    intento = InitialStability()
    intento.initialize_bank_relationships()
