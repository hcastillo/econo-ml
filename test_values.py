import unittest,bank_net

class ValuesTestCase(unittest.TestCase):
    def setUp(self):
        bank_net.Config.N = 5
        bank_net.Config.T = 10
        bank_net.Status.run()

    def test_values_after_execution(self):
        self.assertEqual( bank_net.Model.banks[1].C, 5.9168122666725935)
        self.assertEqual( bank_net.Model.banks[3].D, 141.37162622381376)
        self.assertEqual( bank_net.Model.banks[4].E, 15)
