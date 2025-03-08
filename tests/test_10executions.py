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
        self.assertEqual(self.model.banks[3].D, 256.61549169583344)
        self.assertEqual(self.model.banks[4].E, 15)
        self.assertEqual(self.model.banks[0].A, 128.39220927509402)
        self.assertEqual(self.model.banks[0].R, 2.314126719899878)
        self.assertEqual(self.model.banks[1].A, 147.3)
        self.assertEqual(self.model.banks[2].A, 120.97014378934406)
        self.assertEqual(self.model.banks[3].A, 266.59294619950754)
        self.assertEqual(self.model.banks[4].A, 170.92739558597935)
        self.assertEqual(self.model.banks[4].R, 3.18219174665264)
        self.assertEqual(self.model.banks[4].R, self.model.banks[4].D*self.model.config.reserves)


if __name__ == '__main__':
    unittest.main()
