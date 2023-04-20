import unittest,bank_net


# 1. borrower obtains a loan,
# -------------------

class BankTest(unittest.TestCase):

    shocks = []

    def configureTest(self, shocks:list, N:int=None,T:int=None ):
        BankTest.shocks = shocks
        if N:
            bank_net.Config.N = N
        if T:
            bank_net.Config.T = T
        bank_net.Status.defineLog('DEBUG')
        bank_net.Model.initilize()

    def doTest(self):
        bank_net.Statistics.reset()
        bank_net.Status.debugBanks()
        bank_net.Model.doSimulation()
        bank_net.Status.debugBanks()

    def setBank(self, bank: bank_net.Bank, C: float, L: float, D: float, E: float):
        bank.L = L
        bank.E = E
        bank.C = C
        bank.D = D

    def mockedShock(whichShock):
        for bank in bank_net.Model.banks:
            bank.newD = bank.D + BankTest.shocks[ bank_net.Model.t ][whichShock][bank.id]
            bank.ΔD = bank.newD - bank.D
            bank.D = bank.newD
            if bank.ΔD > 0:
                bank.C += bank.ΔD
            bank_net.Statistics.incrementD[bank_net.Model.t] += bank.ΔD
        bank_net.Status.debugBanks(details=False, info=whichShock)

    def assertBank(self, bank: bank_net.Bank, C: float, L: float, D: float, E: float, l:float =None, s:float=None, d:float=None ):
        self.assertEqual(bank.L, L)
        self.assertEqual(bank.E, E)
        self.assertEqual(bank.C, C)
        self.assertEqual(bank.D, D)
        if l:
            self.assertEqual(bank.l,l)
        if d:
            self.assertEqual(bank.d,d)
        if s:
            self.assertEqual(bank.s,s)
