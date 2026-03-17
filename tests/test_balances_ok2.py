# -*- coding: utf-8 -*-

import unittest
from cmath import phase

from mock import patch
import interbank
import tests.interbank_testclass


class BalanceTestCase(tests.interbank_testclass.InterbankTest):
    """
    Balance is ok after do_loans and do_repayments
     self.statistics.compute_potential_lenders() is the mocked function
     self.do_repayments() is also the other mocked function
     where we checked that balances are ok
    """

    errors_in_balance = []

    #     if self.backward_enabled:
    #         self.banks_backward_copy = copy.deepcopy(self.banks)
    #     self.do_shock('shock1')
    #     self.statistics.compute_potential_lenders()
    #     if not isinstance(self.config.lender_change, lc.Boltzmann):
    #         self.do_interest_rate()
    #     self.do_loans()
    #     self.statistics.compute_interest_rates_and_loans_equity_lenders()
    #     self.log.debug_banks()
    #     self.statistics.compute_leverage()  <--------------------------------------------------
    #     self.do_shock('shock2')
    #     self.do_repayments()
    #     self.log.debug_banks()
    #     if self.log.progress_bar:
    #        self.log.progress_bar.next()
    #     self.statistics.compute_equity() <-----------------------------------------------------
    #     self.statistics.compute_liquidity()


    def initialValues(self):
        self.setBank(bank=self.model.banks[0], C=10.0, L=20.0, D=20.0, E=10.0, lender=1)
        self.setBank(bank=self.model.banks[1], C=20.0, L=14.0, D=30.0, E=4.0)
        self.setBank(bank=self.model.banks[2], C=9.0, L=20.0, D=20.0, E=9.0, lender=1)
        BalanceTestCase.errors_in_balance = []

    @staticmethod
    def check_balance(statistics, phase):
        for bank in statistics.model.banks:
            if bank.not_balanced():
                BalanceTestCase.errors_in_balance.append(f"t={statistics.model.t} {bank.not_balanced()} {phase}")

    def check_balance1(self):
        BalanceTestCase.check_balance(self, 'loans')

    def check_balance2(self):
        BalanceTestCase.check_balance(self, 'repayments')

    @patch.object(interbank.Statistics, "compute_leverage", check_balance1)
    @patch.object(interbank.Statistics, "compute_equity", check_balance2)
    def setUp(self):
        self.configureTest(N=3, T=5,
                           shocks=[
                               {"shock1": [-5, 5, 5], "shock2": [-3, 3, -3], },
                               {"shock1": [2, 5, 2], "shock2": [-5, 3, -5], },
                               {"shock1": [-5, 0, 0], "shock2": [-1, 0, 0], },
                               {"shock1": [-5, 5, 5], "shock2": [-3, 3, -3], },
                               {"shock1": [-3, 2, 2], "shock2": [5, 0, 5], },
                           ], lc="Boltzmann", seed=1234)
        self.initialValues()
        self.doTest()

    def test_values_after_execution(self):
        for error in BalanceTestCase.errors_in_balance:
            print(error)
        self.assertFalse(len(BalanceTestCase.errors_in_balance))


if __name__ == '__main__':
    unittest.main()
