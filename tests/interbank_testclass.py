import unittest
import interbank
import interbank_lenderchange

# 1. borrower obtains a loan,
# -------------------

class InterbankTest(unittest.TestCase):
    shocks = []
    model = None

    def configureTest(self, shocks: list, N: int = None, T: int = None, lc : str = None):
        self.model = interbank.Model()
        if lc:
            self.model.config.lender_change = interbank_lenderchange.determine_algorithm(lc)
        self.model.test = True
        InterbankTest.shocks = shocks
        if N:
            self.model.configure(N=N)
        if T:
            self.model.configure(T=T)
        self.model.log.define_log(log='DEBUG')
        self.model.initialize()

    def doTest(self):
        self.model.simulate_full()

    def __check_values__(self, bank, name, value):
        if value < 0:
            self.model.log.debug("******",
                                 f"{bank.get_id()} value {name}={value} <0 is not valid: I changed it to 0")
            return 0
        else:
            return value

    def setBank(self, bank: interbank.Bank, C: float, L: float, D: float, E: float, lender: int = None):
        D = self.__check_values__(bank, 'D', D)
        L = self.__check_values__(bank, 'L', L)
        E = self.__check_values__(bank, 'E', E)
        R = D*self.model.config.reserves
        C = self.__check_values__(bank, 'C', C) - R
        if C < 0:
            C = 0
            L -= R
        if L + C + R != D + E:
            E = L + C + R - D
            if E < 0:
                E = 0
            self.model.log.debug("******",
                                 f"{bank.get_id()}  L+C must be equal to D+E => E modified to {E:.3f}")
        bank.L = L
        bank.E = E
        bank.C = C
        bank.D = D
        bank.R = R
        if not lender is None:
            bank.lender = lender

    def assertBank(self, bank: interbank.Bank, C: float = None, L: float = None, R: float = None, D: float = None,
                   E: float = None,
                   paid_loan: float = None, paid_profits: float = None, s: float = None, d: float = None,
                   B: float = None, bankrupted: bool = False, active_borrowers: dict = None):
        if L:
            self.assertEqual(bank.L, L)
        if E:
            self.assertEqual(bank.E, E)
        if C:
            self.assertEqual(bank.C, C)
        if R:
            self.assertEqual(bank.R, R)
        if D:
            self.assertEqual(bank.D, D)
        if paid_loan:
            self.assertEqual(bank.paid_loan, paid_loan)
        if paid_profits:
            self.assertEqual(bank.paid_profits, paid_profits)
        if d:
            self.assertEqual(bank.d, d)
        if s:
            self.assertEqual(bank.s, s)
        if B:
            self.assertEqual(bank.B, B)
        if not active_borrowers is None:
            self.assertEqual(bank.active_borrowers, active_borrowers)
        if bankrupted:
            self.assertGreater(bank.failures, 0)
        else:
            self.assertEqual(bank.failures, 0)


def determine_shock_value_mocked(model, bank, whichShock):
    return bank.D + InterbankTest.shocks[model.t][whichShock][bank.id]


def mockedShock(model, whichShock):
    for bank in model.banks:
        bank.incrD = InterbankTest.shocks[model.t][whichShock][bank.id]
        if bank.D + bank.incrD < 0:
            model.log.debug(f"mocked{whichShock}",
                            f"{bank.get_id()} modified simulated ΔD={bank.incrD:.3f} because we had only D={bank.D:.3f}")
            bank.incrD = bank.D + bank.incrD if bank.D > 0 else 0
        bank.D += bank.incrD
        bank.newR = model.config.reserves * bank.D
        bank.incrR = bank.newR - bank.R
        bank.R = bank.newR
        if bank.incrD >= 0:
            bank.C += bank.incrD - bank.incrR
            if whichShock == "shock1":
                bank.s = bank.C  # lender capital to borrow
            bank.d = 0  # it will not need to borrow
            if bank.incrD > 0:
                model.log.debug(f"mocked{whichShock}", f"{bank.get_id()} wins ΔD={bank.incrD:.3f}")
            else:
                model.log.debug(f"mocked{whichShock}", f"{bank.get_id()} has no shock")
        else:
            if whichShock == "shock1":
                bank.s = 0  # we will not be a lender this time
            if bank.incrD - bank.incrR + bank.C >= 0:
                bank.d = 0  # it will not need to borrow
                bank.C += bank.incrD - bank.incrR
                model.log.debug(f"mocked{whichShock}",
                                f"{bank.get_id()} loses ΔD={bank.incrD:.3f}, covered by capital")
            else:
                bank.d = abs(bank.incrD - bank.incrR + bank.C)  # it will need money
                model.log.debug(f"mocked{whichShock}",
                                f"{bank.get_id()} loses ΔD={bank.incrD:.3f}, has C={bank.C:.3f} and needs {bank.d:.3f}")
                if whichShock == "shock2":
                    # in case shock2, we need to fire sale to cover that bank.d:
                    bank.do_fire_sales(bank.d, f"fire sale to cover shock", whichShock)
                else:
                    bank.C = 0
        model.statistics.incrementD[model.t] += bank.incrD
