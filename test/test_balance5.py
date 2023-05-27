import unittest,bank_net
from bank_net_testclass import BankTest
from mock import patch, Mock

# 3. borrower obtains a loan, and after does bankrupcy
# -------------------

class Balance3TestCase(BankTest):

    def initialValues(self):
        self.setBank(bank=bank_net.Model.banks[0], C=10.0, L=20.0, D=20.0, E=10.0)
        self.setBank(bank=bank_net.Model.banks[1], C=10.0, L=20.0, D=20.0, E=10.0)
        self.setBank(bank=bank_net.Model.banks[2], C=30.0, L=120.0, D=135.0, E=15.0)

    @patch.object(bank_net.Model, "doShock", BankTest.mockedShock)
    def setUp(self):
        self.configureTest( N=3,T=5,
                            shocks=[
                                {"shock1": [-5, 5,5], "shock2": [-3, 3,-3], },
                                {"shock1": [ 2, 5, 2], "shock2": [-5, 3, -5], },
                                {"shock1": [-5, 0, 0], "shock2": [-1, 0, 0 ], },
                                {"shock1": [-5, 5, 5], "shock2": [-3, 3, -3], },
                                {"shock1": [-3, 2, 2], "shock2": [ 5, 0, 5], },
                            ])
        self.initialValues()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=bank_net.Model.banks[0], C=24, L=120, D=129, E=15, bankrupted=True)
        self.assertBank(bank=bank_net.Model.banks[1], C=36, L=20, D=46, E=10)
        self.assertBank(bank=bank_net.Model.banks[2], C=35, L=120, D=143, E=12)





if __name__ == '__main__':
    unittest.main()