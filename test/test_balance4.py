import unittest,bank_net
from bank_net_testclass import BankTest
from mock import patch, Mock

# 3. bank does firesale but it is covered ok
# -------------------

class Balance3TestCase(BankTest):

    def initialValues(self):
        bank_net.Config.ρ = 0.4 # fire sale cost
        self.setBank(bank=bank_net.Model.banks[0], C=0.0, L=8.0, D=3.0, E=5.0)
        self.setBank(bank=bank_net.Model.banks[1], C=10.0, L=20.0, D=20.0, E=10.0)

    @patch.object(bank_net, "doShock", BankTest.mockedShock)
    def setUp(self):
        self.configureTest( N=2,T=1,
                            shocks=[
                                {"shock1": [0, 0], "shock2": [-3, 0], },
                            ])
        self.initialValues()
        self.doTest()
        bank_net.Config.ρ = 0.3 # fire sale cost

    def test_values_after_execution(self):
        self.assertBank(bank=bank_net.Model.banks[0], C=0, L=0.5, D=0, E=0.5, paidloan=0, bankrupted=False)
        self.assertBank(bank=bank_net.Model.banks[1], C=10, L=20, D=20, E=10)





if __name__ == '__main__':
    unittest.main()