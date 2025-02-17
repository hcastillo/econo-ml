import unittest
import interbank


class ValuesAfterExecutionTestCase(unittest.TestCase):
    def setUp(self):
        self.model = interbank.Model()
        self.model.configure(N=5, T=10, mi=0.7, omega=0.55)
        self.model.set_policy_recommendation(1)
        self.model.log.define_log('DEBUG')
        self.model.initialize()

        self.model.log.debug_banks()
        self.model.simulate_full()
        self.model.log.debug_banks()

    def test_values_after_execution(self):
        self.assertEqual(self.model.banks[1].C, 29.4)
        self.assertEqual(self.model.banks[3].D, 247.37200851216474)
        self.assertEqual(self.model.banks[4].E, 15)
        self.assertEqual(self.model.banks[0].A, 149.4)
        self.assertEqual(self.model.banks[0].R, 2.7)
        self.assertEqual(self.model.banks[1].A, 149.4)
        self.assertEqual(self.model.banks[2].A, 123.11214378934406)
        self.assertEqual(self.model.banks[3].A, 253.89193080315928)
        self.assertEqual(self.model.banks[4].A, 173.02739558597938)
        self.assertEqual(self.model.banks[4].R, 3.18219174665264)
        self.assertEqual(self.model.banks[4].R, self.model.banks[4].D*self.model.config.reserves)


if __name__ == '__main__':
    unittest.main()
