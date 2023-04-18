import unittest,bank_net

class ValuesTestCase(unittest.TestCase):
    def setUp(self):
        bank_net.Config.N = 5
        bank_net.Config.T = 10
        bank_net.Status.run()

    def test_values_after_execution(self):
        self.assertEqual( bank_net.Model.banks[1].C, 0.13382435876260956)
        self.assertEqual( bank_net.Model.banks[3].D, 135)
        self.assertEqual( bank_net.Model.banks[4].E, 15)
