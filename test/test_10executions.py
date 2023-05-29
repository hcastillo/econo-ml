import unittest,interbank

class ValuesAfterExecutionTestCase(unittest.TestCase):
    def setUp(self):
        self.model = interbank.Model()
        self.model.configure( N=5, T=10 )
        self.model.log.defineLog('DEBUG')
        self.model.initialize()

        self.model.log.debugBanks()
        self.model.doFullSimulation()
        self.model.log.debugBanks()

    def test_values_after_execution(self):
        self.assertEqual( self.model.banks[1].C, 30)
        self.assertEqual( self.model.banks[3].D, 151.85677679197377)
        self.assertEqual( self.model.banks[4].E, 15)
        self.assertEqual(self.model.banks[0].A, 99.8192837015739)
        self.assertEqual(self.model.banks[1].A, 150)
        self.assertEqual(self.model.banks[2].A, 135.2839370516004)
        self.assertEqual(self.model.banks[3].A, 166.85677679197377)
        self.assertEqual(self.model.banks[4].A, 225.8449975649632)


if __name__ == '__main__':
    unittest.main()
