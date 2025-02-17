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
        self.assertEqual(self.model.banks[1].C, 18.056201822116368)
        self.assertEqual(self.model.banks[3].D, 220.9207438929657)
        self.assertEqual(self.model.banks[4].E, 3.106061610472147)
        self.assertEqual(self.model.banks[0].A, 129.49684064399628)
        self.assertEqual(self.model.banks[0].R, 2.293813074367271)
        self.assertEqual(self.model.banks[1].A, 138.05620182211638)
        self.assertEqual(self.model.banks[2].A, 175.96021261211547)
        self.assertEqual(self.model.banks[3].A, 223.22851162676574)
        self.assertEqual(self.model.banks[4].A, 103.19011141849873)
        self.assertEqual(self.model.banks[4].R, 2.061650230234285)
        self.assertEqual(self.model.banks[4].R, self.model.banks[4].D*self.model.config.reserves)


if __name__ == '__main__':
    unittest.main()
