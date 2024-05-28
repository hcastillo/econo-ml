import unittest
import interbank


class ValuesAfterExecutionTestCase(unittest.TestCase):
    def setUp(self):
        self.model = interbank.Model()
        self.model.configure(N=5, T=10, µ=0.7, ω= 0.55)
        self.model.set_policy_recommendation(1)
        self.model.log.define_log('DEBUG')
        self.model.initialize()

        self.model.log.debug_banks()
        self.model.simulate_full()
        self.model.log.debug_banks()

    def test_values_after_execution(self):
        self.assertEqual(self.model.banks[1].C, 30)
        self.assertEqual(self.model.banks[3].D, 348.5986847105888)
        self.assertEqual(self.model.banks[4].E, 15)
        self.assertEqual(self.model.banks[0].A, 150)
        self.assertEqual(self.model.banks[1].A, 150)
        self.assertEqual(self.model.banks[2].A, 156.1348456314171)
        self.assertEqual(self.model.banks[3].A, 350.0463935682433)
        self.assertEqual(self.model.banks[4].A, 149.34872636893022)


if __name__ == '__main__':
    unittest.main()
