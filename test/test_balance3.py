import unittest,bank_net
from bank_net_testclass import BankTest
from mock import patch, Mock

# 3. borrower obtains a loan, and after does bankrupcy
# -------------------

class Balance3TestCase(BankTest):

    def initialValues(self):
        self.setBank(bank=bank_net.Model.banks[0], C=10.0, L=20.0, D=20.0, E=10.0)
        self.setBank(bank=bank_net.Model.banks[1], C=10.0, L=20.0, D=20.0, E=10.0)

    @patch.object(bank_net, "doShock", BankTest.mockedShock)
    def setUp(self):
        self.configureTest( N=2,T=1,
                            shocks=[
                                {"shock1": [-15, 5], "shock2": [-3, 3], },
                            ])
        self.initialValues()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=bank_net.Model.banks[0], paidloan=0, bankrupted=True)
        self.assertBank(bank=bank_net.Model.banks[1], C=13, L=20, D=28, E=5, B=5)





if __name__ == '__main__':
    unittest.main()