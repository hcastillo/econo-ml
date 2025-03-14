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
        self.assertEqual(self.model.banks[1].C, 0.6401971955167554)
        self.assertEqual(self.model.banks[3].D, 178.1135090231961)
        self.assertEqual(self.model.banks[4].E, 15)
        self.assertEqual(self.model.banks[0].A, 147.3)
        self.assertEqual(self.model.banks[0].R, 2.7)
        self.assertEqual(self.model.banks[1].A, 120.64019719551675)
        self.assertEqual(self.model.banks[2].A, 168.67420395631007)
        self.assertEqual(self.model.banks[3].A, 189.55123884273218)
        self.assertEqual(self.model.banks[4].A, 146.66175184155162)
        self.assertEqual(self.model.banks[4].R, 2.6869745273786045)
        self.assertEqual(self.model.banks[4].R, self.model.banks[4].D*self.model.config.reserves)


if __name__ == '__main__':
    unittest.main()
