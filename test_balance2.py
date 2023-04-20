import unittest,bank_net
from bank_net_testclass import BankTest
from mock import patch, Mock

# 2. borrower obtains a loan, and after it pays it with a positive second shock
# -------------------

class BalanceTestCase(BankTest):

    def initialValues(self):
        self.setBank(bank=bank_net.Model.banks[0], C=10.0, L=15.0, D=15.0, E=10.0)
        self.setBank(bank=bank_net.Model.banks[1], C=10.0, L=15.0, D=15.0, E=10.0)


    def test_values_after_execution(self):
        self.assertBank(bank=bank_net.Model.banks[0], C=16.5, L=15.0, D=12.0, E=10.0, l=3)
        self.assertBank(bank=bank_net.Model.banks[1], C=13.0, L=15.0, D=18.0, E=10.0, s=30)

    @patch.object(bank_net, "doShock", BankTest.mockedShock)
    def setUp(self):
        self.configureTest( N=2,T=1,
                            shocks=[
                                {"shock1": [-13, 13], "shock2": [20, -20], },
                            ])
        self.initialValues()
        self.doTest()



