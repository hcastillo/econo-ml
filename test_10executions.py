import unittest,bank_net

class ValuesAfterExecutionTestCase(unittest.TestCase):
    def setUp(self):
        bank_net.Config.N = 5
        bank_net.Config.T = 10
        bank_net.Status.defineLog('DEBUG')
        bank_net.Model.initilize()

        bank_net.Statistics.reset()
        bank_net.Status.debugBanks()
        bank_net.Model.doSimulation()
        bank_net.Status.debugBanks()

    def test_values_after_execution(self):
        self.assertEqual( bank_net.Model.banks[1].C, 30 ) ## 139.53374153026374)
        self.assertEqual( bank_net.Model.banks[3].D, 151.85677679197377 )
        self.assertEqual( bank_net.Model.banks[4].E, 15)



if __name__ == '__main__':
    unittest.main()