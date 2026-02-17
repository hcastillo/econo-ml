# -*- coding: utf-8 -*-

import unittest
import interbank
import tests.interbank_testclass
from mock import patch


class BalanceTestCase(tests.interbank_testclass.InterbankTest):
    """
       test borrower bank does fire sales, but it is covered ok
       """

    #       #0             #1
    #   -----------    -----------
    #   C=0   | D=3    C=9.5| D=25
    #   L=7.94| E=5    L=20 | E=10
    #   R=0.06|        R=0.5
    #
    #   shock1=-5      shock1=+5   --> #0 obtains from #1 a loan of 3.40
    #
    #   shock2=0       shock2=0    --> #0 has to return 3.468
    #                                  no enough C -> fire sales 3.468
    #                                  that costs 11.56 in L -> new E = -6.092 fails
    #                              --> #1 new
    def initialValues(self):
        interbank.Config.rho = 0.4 # fire sale cost
        self.setBank(bank=self.model.banks[0], C=0.0, L=8, D=3.0, E=5.0)
        self.setBank(bank=self.model.banks[1], C=10.0, L=20.0, D=20.0, E=10.0)

    @patch.object(interbank.Model, "do_shock", tests.interbank_testclass.mockedShock)
    def setUp(self):
        previous = interbank.Config.rho
        interbank.Config.rho = 0.3
        self.configureTest(N=2,T=1,
                           shocks=[
                               {"shock1": [0, 1], "shock2": [-3, 0], },
                           ], lc="Boltzmann")
        self.initialValues()
        self.doTest()
        interbank.Config.rho = previous

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[0], C=0, L=0.5900000000000007, D=0, E=0.5900000000000007,
                        paid_loan=0, bankrupted=False)
        self.assertBank(bank=self.model.banks[1], C=10.58, L=20, D=21, E=10)





if __name__ == '__main__':
    unittest.main()