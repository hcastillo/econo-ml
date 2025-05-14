# -*- coding: utf-8 -*-
import unittest
import interbank
import interbank_lenderchange
import tests.interbank_testclass
from mock import patch


class BalanceTestCase(tests.interbank_testclass.InterbankTest):
    """
    test borrower can pays the loan using C, no loan and no second shock
    """

    #       #0             #1
    #   -----------    -----------
    #   C=1.6| D=20    C=1.7| D=15
    #   L=20 | E=2     L=15 | E=2
    #   R=0.4|         R=0.3
    #
    #   shock1=-5      shock1=+5   --> #0 obtains from #1 a loan of 3.40
    #
    #   shock2=0       shock2=0    --> #0 has to return 3.468
    #                                  no enough C -> fire sales 3.468
    #                                  that costs 11.56 in L -> new E = -6.092 fails
    #                              --> #1 also fails
    def initialValues(self):
        self.setBank(bank=self.model.banks[0], C=2.0, L=20.0, D=20.0, E=2.0)
        self.assertBank(bank=self.model.banks[0], C=1.6, R=0.4)
        self.setBank(bank=self.model.banks[1], C=2.0, L=15.0, D=15.0, E=2.0)

    @patch.object(interbank.Model, "do_shock", tests.interbank_testclass.mockedShock)
    def setUp(self):
        self.configureTest(N=2, T=1,
                           shocks=[
                               {"shock1": [-5, 5], "shock2": [0, 0], },
                           ])
        self.model.config.lender_change = interbank_lenderchange.determine_algorithm("Boltzmann")
        self.initialValues()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[0], bankrupted=True)
        self.assertBank(bank=self.model.banks[1], bankrupted=True, active_borrowers={})
        # to ensure even both fail that B is correctly added:
        self.assertEqual(self.model.statistics.B[0],3.3000000000000003)



if __name__ == '__main__':
    unittest.main()
