import unittest

from mock import patch

import interbank
import interbank_testclass


class Balance3TestCase(interbank_testclass.InterbankTest):
    """
    test borrower obtains a loan, and after does bankruptcy
    """

    def initialValues(self):
        self.setBank(bank=self.model.banks[0], C=10.0, L=20.0, D=20.0, E=10.0)
        self.setBank(bank=self.model.banks[1], C=10.0, L=20.0, D=20.0, E=10.0)

    @patch.object(interbank.Model, "doShock", interbank_testclass.mockedShock)
    def setUp(self):
        self.configureTest( N=2,T=1,
                            shocks=[
                                {"shock1": [-15, 5], "shock2": [-3, 3], },
                            ])
        self.initialValues()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[0], paidloan=0, bankrupted=True)
        self.assertBank(bank=self.model.banks[1], C=13, L=20, D=28, E=5, B=5)





if __name__ == '__main__':
    unittest.main()