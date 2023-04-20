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
            bank.ΔD = BankTest.shocks[ bank_net.Model.t ][whichShock][bank.id]
            bank.D += bank.ΔD
            if bank.ΔD > 0:
                bank.C += bank.ΔD
                bank.s = bank.C  # lender capital to borrow
                bank.d = 0  # it will not need to borrow
                bank_net.Status.debug(whichShock,
                             f"{bank.getId()} wins ΔD={bank.ΔD:.3f}")

            else:
                bank.s = 0  # we will not be a lender this time
                if bank.ΔD + bank.C >= 0:
                    bank.d = 0  # it will not need to borrow
                    bank.C += bank.ΔD
                    bank_net.Status.debug(whichShock,
                                 f"{bank.getId()} loses ΔD={bank.ΔD:.3f}, covered by capital, now C={bank.C:.3f}")
                else:
                    bank.d = abs(bank.ΔD + bank.C)  # it will need money
                    bank_net.Status.debug(whichShock,
                                 f"{bank.getId()} loses ΔD={bank.ΔD:.3f} but has only C={bank.C:.3f}, now C=0")
                    bank.C = 0  # we run out of capital
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
