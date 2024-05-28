# -*- coding: utf-8 -*-

import unittest
import interbank
import tests.interbank_testclass
from mock import patch


class Balance3TestCase(tests.interbank_testclass.InterbankTest):
    """
    test borrower obtains a loan, and after does bankruptcy
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
        self.initial_values()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[0], C=24, L=120, D=129, E=15, bankrupted=True)
        self.assertBank(bank=self.model.banks[1], C=36, L=20,  D=46,  E=10)
        self.assertBank(bank=self.model.banks[2], C=35, L=120, D=143, E=12)


if __name__ == '__main__':
    unittest.main()
