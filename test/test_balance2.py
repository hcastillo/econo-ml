import unittest

from mock import patch

import interbank
from test import interbank_testclass


class Balance2TestCase(interbank_testclass.InterbankTest):
    """
    test borrower obtains a loan, and after it pays it with a positive second shock
    """

    def initialValues(self):
        self.setBank(bank=self.model.banks[0], C=10.0, L=15.0, D=15.0, E=10.0)
        self.setBank(bank=self.model.banks[1], C=10.0, L=15.0, D=15.0, E=10.0)

    @patch.object(interbank.Model, "do_shock", interbank_testclass.mockedShock)
    def setUp(self):
        self.configureTest( N=2,T=1,
                            shocks=[
                                {"shock1": [-13, 13], "shock2": [20, -20], },
                            ])
        self.initialValues()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[0], C=16.94, L=15.0, D=22.0, E=9.94, paidloan=3)
        self.assertBank(bank=self.model.banks[1], C=3.06, L=15.0, D=8.0, E=10.06, s=20)





if __name__ == '__main__':
    unittest.main()