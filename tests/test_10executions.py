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
        self.assertEqual(self.model.banks[1].C, 27.3)
        self.assertEqual(self.model.banks[3].D, 199.31236336553437)
        self.assertEqual(self.model.banks[4].E, 15)
        self.assertEqual(self.model.banks[0].A, 121.73103416054751)
        self.assertEqual(self.model.banks[0].R, 1.7310341605475106)
        self.assertEqual(self.model.banks[1].A, 150)
        self.assertEqual(self.model.banks[2].A, 123.13503987356722)
        self.assertEqual(self.model.banks[3].A, 214.31236336553437)
        self.assertEqual(self.model.banks[4].A, 174.109587332632)
        self.assertEqual(self.model.banks[4].R, 3.18219174665264)
        self.assertEqual(self.model.banks[4].R, self.model.banks[4].D*self.model.config.reserves)


if __name__ == '__main__':
    unittest.main()
