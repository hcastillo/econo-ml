import unittest
import interbank
import interbank_lenderchange


class ValuesAfterExecutionTestCase(unittest.TestCase):

    CONFIG = 'C_i0=30 D_i0=135 E_i0=15 L_i0=120 N=50 T=50 alfa=0.1 allow_replacement_of_bankrupted=True beta=5 chi=0.0015 mu=0.7 omega=0.55 phi=0.25 psi=0.0 r_i0=0.02 reintroduce_with_median=False reserves=0.02 rho=0.3 xi=0.9'
    SEED = 39393

    def setUp(self):
        self.model = interbank.Model()
        self.model.configure_json(self.CONFIG)
        self.model.configure(N=5, T=10)
        self.model.config.lender_change = interbank_lenderchange.determine_algorithm("Boltzmann")
        self.model.set_policy_recommendation(1)
        self.model.log.define_log('DEBUG')
        self.model.initialize(seed=self.SEED)

        self.model.log.debug_banks()
        self.model.simulate_full()
        self.model.log.debug_banks()

    def test_values_after_execution(self):
        self.assertEqual(self.model.banks[1].C,  20.921995464604404)
        self.assertEqual(self.model.banks[3].D, 108.85239242784516)
        self.assertEqual(self.model.banks[4].E, 7.566583939765853)
        self.assertEqual(self.model.banks[0].A, 191.24069178290375)
        self.assertEqual(self.model.banks[0].R, 3.7435422163620604)
        self.assertEqual(self.model.banks[1].A, 143.76877225737616)
        self.assertEqual(self.model.banks[2].A, 142.33328152717868)
        self.assertEqual(self.model.banks[3].A, 115.71456891025957)
        self.assertEqual(self.model.banks[4].A, 160.48384520203993)
        self.assertEqual(self.model.banks[4].R, 3.0608944625297334)
        self.assertEqual(self.model.banks[4].R, self.model.banks[4].D*self.model.config.reserves)


if __name__ == '__main__':
    unittest.main()
