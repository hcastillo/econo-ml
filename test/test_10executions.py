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

        # A=C+L or A=D+E
        self.assertEqual(bank_net.Model.banks[0].A, 99.8192837015739)
        self.assertEqual(bank_net.Model.banks[1].A, 150)
        self.assertEqual(bank_net.Model.banks[2].A, 135.2839370516004)
        self.assertEqual(bank_net.Model.banks[3].A, 166.85677679197377)
        self.assertEqual(bank_net.Model.banks[4].A, 225.8449975649632)

if __name__ == '__main__':
    unittest.main()