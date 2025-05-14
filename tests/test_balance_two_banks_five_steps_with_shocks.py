# -*- coding: utf-8 -*-

import unittest
import interbank
import interbank_lenderchange
import tests.interbank_testclass
from mock import patch


class BalanceTestCase(tests.interbank_testclass.InterbankTest):
    """
    test with two banks during 5 steps with shocks
    """

    def initial_values(self):
        self.setBank(bank=self.model.banks[0], C=10.0, L=20.0,  D=20.0,  E=10.0)
        self.setBank(bank=self.model.banks[1], C=10.0, L=20.0,  D=20.0,  E=10.0)
        self.setBank(bank=self.model.banks[2], C=30.0, L=120.0, D=135.0, E=15.0)

    @patch.object(interbank.Model, "do_shock", tests.interbank_testclass.mockedShock)
    def setUp(self):
        self.configureTest( N=3, T=5, shocks=[
                                {"shock1": [-5, 5, 5], "shock2": [-3, 3,-3], },
                                {"shock1": [ 2, 5, 2], "shock2": [-5, 3,-5], },
                                {"shock1": [-5, 0, 0], "shock2": [-1, 0, 0], },
                                {"shock1": [-5, 5, 5], "shock2": [-3, 3,-3], },
                                {"shock1": [-3, 2, 2], "shock2": [ 5, 0, 5], },
                            ])
        self.model.config.lender_change = interbank_lenderchange.determine_algorithm("Boltzmann")
        self.initial_values()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[0], C=21.419999999999995, L=120, D=129, E=15, bankrupted=True)
        self.assertBank(bank=self.model.banks[1], C=35.080000000000005, L=20,  D=46,  E=10)
        self.assertBank(bank=self.model.banks[2], C=33.96, L=120, D=143, E=13.819999999999999)


if __name__ == '__main__':
    unittest.main()
