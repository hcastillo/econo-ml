import unittest,interbank

class ValuesAfterExecutionTestCase(unittest.TestCase):
    def setUp(self):
        interbank.Config.N = 5
        interbank.Config.T = 10
        interbank.Status.defineLog('DEBUG')
        interbank.Model.initialize()

        interbank.Statistics.reset()
        interbank.Status.debugBanks()
        interbank.Model.doFullSimulation()
        interbank.Status.debugBanks()

    def test_values_after_execution(self):
        self.assertEqual( interbank.Model.banks[1].C, 30 ) ## 139.53374153026374)
        self.assertEqual( interbank.Model.banks[3].D, 151.85677679197377 )
        self.assertEqual( interbank.Model.banks[4].E, 15)

        # A=C+L or A=D+E
        self.assertEqual(interbank.Model.banks[0].A, 99.8192837015739)
        self.assertEqual(interbank.Model.banks[1].A, 150)
        self.assertEqual(interbank.Model.banks[2].A, 135.2839370516004)
        self.assertEqual(interbank.Model.banks[3].A, 166.85677679197377)
        self.assertEqual(interbank.Model.banks[4].A, 225.8449975649632)

if __name__ == '__main__':
    unittest.main()