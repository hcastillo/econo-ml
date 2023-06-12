import unittest
import interbank


class ValuesAfterExecutionTestCase(unittest.TestCase):
    def setUp(self):
        self.model = interbank.Model()
        self.model.configure(N=5, T=10)
        self.model.set_policy_recommendation(1)
        self.model.log.define_log('DEBUG')
        self.model.initialize()

        self.model.log.debug_banks()
        self.model.simulate_full()
        self.model.log.debug_banks()

    def test_values_after_execution(self):
        self.assertEqual(self.model.banks[1].C, 72.4384110807936)
        self.assertEqual(self.model.banks[3].D, 135)
        self.assertEqual(self.model.banks[4].E, 15)
        self.assertEqual(self.model.banks[0].A, 242.26886220650744)
        self.assertEqual(self.model.banks[1].A, 192.4384110807936)
        self.assertEqual(self.model.banks[2].A, 145.42985507645253)
        self.assertEqual(self.model.banks[3].A, 150)
        self.assertEqual(self.model.banks[4].A, 155.20113401303334)


if __name__ == '__main__':
    unittest.main()
