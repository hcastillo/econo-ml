import unittest,bank_net
import bank_net_testclass
from mock import patch, Mock


# 1. borrower can pays the loan using C, no loan and no second shock
# -------------------

class Balance1TestCase(bank_net_testclass.BankTest):

    def initialValues(self):
        self.setBank(bank=bank_net.Model.banks[0] ,C=10.0,L=15.0,D=15.0,E=10.0)
        self.setBank(bank=bank_net.Model.banks[1] ,C=10.0,L=15.0,D=15.0,E=10.0)

    @patch.object(bank_net, "doShock", bank_net_testclass.BankTest.mockedShock)
    def setUp(self):
        self.configureTest( N=2,T=1,
                            shocks=[
                                {"shock1": [-3, 3], "shock2": [0, 0], },
                            ])
        self.initialValues()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank( bank=bank_net.Model.banks[0], C=7.0, L=15.0, D=12.0, E=10.0 )
        self.assertBank( bank=bank_net.Model.banks[1], C=13.0, L=15.0, D=18.0, E=10.0,s=13)

if __name__ == '__main__':
    unittest.main()