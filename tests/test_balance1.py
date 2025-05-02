# -*- coding: utf-8 -*-
import unittest
import interbank
import tests.interbank_testclass
from mock import patch


class Balance1TestCase(tests.interbank_testclass.InterbankTest):
    """
    test borrower can pays the loan using C, no loan and no second shock
    """

    def initialValues(self):
        self.setBank(bank=self.model.banks[0], C=2.0, L=20.0, D=20.0, E=2.0)
        self.assertBank(bank=self.model.banks[0], C=1.6, R=0.4)
        self.setBank(bank=self.model.banks[1], C=10.0, L=15.0, D=15.0, E=10.0)

    @patch.object(interbank.Model, "do_shock", tests.interbank_testclass.mockedShock)
    def setUp(self):
        self.configureTest(N=2, T=1,
                           shocks=[
                               {"shock1": [-5, 5], "shock2": [0, 0], },
                           ])
        self.initialValues()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[0], C=27.3, R=2.7, L=120, D=135, E=15, bankrupted=True)
        self.assertBank(bank=self.model.banks[1], C=11.299999999999999, L=15.0, D=20.0, E=6.6, s=7.899999999999999)


if __name__ == '__main__':
    unittest.main()
