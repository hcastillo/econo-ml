import unittest
import interbank


# 1. borrower obtains a loan,
# -------------------

class InterbankTest(unittest.TestCase):

    shocks = []
    model = None

    def configureTest(self, shocks:list, N:int=None, T:int=None ):
        self.model = interbank.Model()
        self.model.test = True
        InterbankTest.shocks = shocks
        if N:
            self.model.configure(N=N)
        if T:
            self.model.configure(T=T)
        self.model.log.define_log(log='DEBUG', script_name=self.id().split('.')[0])
        self.model.initialize()

    def doTest(self):
        self.model.simulate_full()

    def __check_values__(self,bank,name,value):
        if value<0:
            self.model.log.debug("******",
                                  f"{bank.get_id()} value {name}={value} <0 is not valid: I changed it to 0")
            return 0
        else:
            return value

    def setBank(self, bank: interbank.Bank, C: float, L: float, D: float, E: float):
        C = self.__check_values__(bank,'C',C)
        D = self.__check_values__(bank,'D',D)
        L = self.__check_values__(bank,'L',L)
        E = self.__check_values__(bank,'E',E)

        if L + C != D + E:
            E = L+C-D
            if E<0:
                E = 0
            self.model.log.debug("******",
                                  f"{bank.get_id()}  L+C must be equal to D+E => E modified to {E:.3f}")
        bank.L = L
        bank.E = E
        bank.C = C
        bank.D = D

    def assertBank(self, bank: interbank.Bank, C: float=None, L: float=None, D: float=None, E: float=None,
                                              paid_loan:float=None, s:float=None, d:float=None,
                                              B:float=None, bankrupted:bool=False ):
        if L:
            self.assertEqual(bank.L, L)
        if E:
            self.assertEqual(bank.E, E)
        if C:
            self.assertEqual(bank.C, C)
        if D:
            self.assertEqual(bank.D, D)
        if paid_loan:
            self.assertEqual(bank.paid_loan,paid_loan)
        if d:
            self.assertEqual(bank.d,d)
        if s:
            self.assertEqual(bank.s,s)
        if B:
            self.assertEqual(bank.B,B)
        if bankrupted:
            self.assertGreater(bank.failures,0)
        else:
            self.assertEqual(bank.failures,0)



def mockedShock(model,whichShock):
    for bank in model.banks:
        bank.incrD = InterbankTest.shocks[model.t][whichShock][bank.id]
        if bank.D + bank.incrD < 0:
            bank.incrD = bank.D + bank.incrD if bank.D > 0 else 0
            model.log.debug("******",
                        f"{bank.get_id()} modified simulated ΔD={bank.incrD:.3f} because we had only D={bank.D:.3f}")
        bank.D += bank.incrD
        if bank.incrD >= 0:
            bank.C += bank.incrD
            if whichShock == "shock1":
                bank.s = bank.C  # lender capital to borrow
            bank.d = 0  # it will not need to borrow
            if bank.incrD > 0:
                model.log.debug(whichShock, f"{bank.get_id()} wins ΔD={bank.incrD:.3f}")

        else:
            if whichShock == "shock1":
                bank.s = 0  # we will not be a lender this time
            if bank.incrD + bank.C >= 0:
                bank.d = 0  # it will not need to borrow
                bank.C += bank.incrD
                model.log.debug(whichShock,
                                     f"{bank.get_id()} loses ΔD={bank.incrD:.3f}, covered by capital, now C={bank.C:.3f}")
            else:
                bank.d = abs(bank.incrD + bank.C)  # it will need money
                model.log.debug(whichShock,
                                     f"{bank.get_id()} loses ΔD={bank.incrD:.3f} but has only C={bank.C:.3f}, now C=0")
                bank.C = 0  # we run out of capital
        model.statistics.incrementD[model.t] += bank.incrD