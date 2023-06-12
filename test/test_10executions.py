import unittest,interbank

class ValuesAfterExecutionTestCase(unittest.TestCase):
    def setUp(self):
        self.model = interbank.Model()
        self.model.configure( N=5, T=10 )
        self.model.log.define_log('DEBUG')
        self.model.initialize()

        self.model.log.debug_banks()
        self.model.simulate_full()
        self.model.log.debug_banks()

    def test_values_after_execution(self):
        self.assertEqual( self.model.banks[1].C, 36.10600655009327)
        self.assertEqual( self.model.banks[3].D, 112.18772359978848)
        self.assertEqual( self.model.banks[4].E, 15)
        self.assertEqual(self.model.banks[0].A, 128.2902482457455)
        self.assertEqual(self.model.banks[1].A, 156.10600655009327)
        self.assertEqual(self.model.banks[2].A, 168.16787536358308)
        self.assertEqual(self.model.banks[3].A, 117.73612363929549)
        self.assertEqual(self.model.banks[4].A, 165.36947609289194)


if __name__ == '__main__':
    unittest.main()
