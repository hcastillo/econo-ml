# -*- coding: utf-8 -*-
import unittest
import interbank
import tests.interbank_testclass
from mock import patch


class Balance2TestCase(tests.interbank_testclass.InterbankTest):
    """
    test borrower obtains a loan, and after it pays it with a positive second shock
    """

    def initialValues(self):
        self.setBank(bank=self.model.banks[0], C=10.0, L=15.0, D=15.0, E=10.0)
        self.setBank(bank=self.model.banks[1], C=10.0, L=15.0, D=15.0, E=10.0)

    @patch.object(interbank.Model, "determine_shock_value", tests.interbank_testclass.determine_shock_value_mocked)
    def setUp(self):
        self.configureTest( N=2,T=1,
                            shocks=[
                                {"shock1": [-13, 13], "shock2": [20, -20], },
                            ])
        self.initialValues()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[0], C=16.499200000000002, L=15.0, D=22.0, E=9.9392,
                        paid_loan=3.100800000000001)
        self.assertBank(bank=self.model.banks[1], C=3.040000000000001, L=15, D=8.0,
                        E=10.0608, s=16.36)





if __name__ == '__main__':
    unittest.main()