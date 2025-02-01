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
        self.assertEqual(self.model.banks[1].C, 18.424695736853437)
        self.assertEqual(self.model.banks[3].D, 220.9207438929657)
        self.assertEqual(self.model.banks[4].E, 5.601193543445)
        self.assertEqual(self.model.banks[0].A, 129.69065371836356)
        self.assertEqual(self.model.banks[1].A, 138.42469573685344)
        self.assertEqual(self.model.banks[2].A, 177.10225776746478)
        self.assertEqual(self.model.banks[3].A, 225.20718284643277)
        self.assertEqual(self.model.banks[4].A, 106.57313363349286)


if __name__ == '__main__':
    unittest.main()
