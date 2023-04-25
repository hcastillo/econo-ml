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

    def __check_values__(self,bank,name,value):
        if value<0:
            bank_net.Status.debug("******",
                                  f"{bank.getId()} value {name}={value} <0 is not valid: I changed to 0")
            return 0
        else:
            return value

    def setBank(self, bank: bank_net.Bank, C: float, L: float, D: float, E: float):
        C = self.__check_values__(bank,'C',C)
        D = self.__check_values__(bank,'D',D)
        L = self.__check_values__(bank,'L',L)
        E = self.__check_values__(bank,'E',E)

        if L + C != D + E:
            E = L+C-D
            if E<0:
                E = 0
            bank_net.Status.debug("******",
                                  f"{bank.getId()}  L+C must be equal to D+E => E modified to {E:.3f}")
        bank.L = L
        bank.E = E
        bank.C = C
        bank.D = D

    def mockedShock(whichShock):
        for bank in bank_net.Model.banks:
            bank.ΔD = BankTest.shocks[ bank_net.Model.t ][whichShock][bank.id]
            if bank.D + bank.ΔD < 0:
                bank.ΔD = bank.D + bank.ΔD if bank.D>0 else 0
                bank_net.Status.debug("******",
                                f"{bank.getId()} modified simulated ΔD={bank.ΔD:.3f} because we had only D={bank.D:.3f}")
            bank.D += bank.ΔD
            if bank.ΔD >= 0:
                bank.C += bank.ΔD
                if whichShock == "shock1":
                    bank.s = bank.C  # lender capital to borrow
                bank.d = 0  # it will not need to borrow
                if bank.ΔD>0:
                    bank_net.Status.debug(whichShock,
                             f"{bank.getId()} wins ΔD={bank.ΔD:.3f}")

            else:
                if whichShock == "shock1":
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

    def assertBank(self, bank: bank_net.Bank, C: float=None, L: float=None, D: float=None, E: float=None,
                                              paidloan:float=None, s:float=None, d:float=None,
                                              B:float=None, bankrupted:bool=False ):
        if L:
            self.assertEqual(bank.L, L)
        if E:
            self.assertEqual(bank.E, E)
        if C:
            self.assertEqual(bank.C, C)
        if D:
            self.assertEqual(bank.D, D)
        if paidloan:
            self.assertEqual(bank.paidloan,paidloan)
        if d:
            self.assertEqual(bank.d,d)
        if s:
            self.assertEqual(bank.s,s)
        if B:
            self.assertEqual(bank.B,B)
        if bankrupted:
            self.assertGreater(bank.failures,0)
            self.assertEqual(bank.C, bank_net.Config.C_i0)
            self.assertEqual(bank.E, bank_net.Config.E_i0)
            self.assertEqual(bank.D, bank_net.Config.D_i0)
            self.assertEqual(bank.L, bank_net.Config.L_i0)
        else:
            self.assertEqual(bank.failures,0)
