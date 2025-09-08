import unittest
import interbank
import interbank_lenderchange


class ValuesAfterExecutionTestCase(unittest.TestCase):

    CONFIG = 'C_i0=30 D_i0=135 E_i0=15 L_i0=120 N=50 T=50 alfa=0.1 allow_replacement_of_bankrupted=True beta=5 chi=0.0015 mu=0.7 omega=0.55 phi=0.25 psi=0.0 r_i0=0.02 reintroduce_with_median=False reserves=0.02 rho=0.3 xi=0.9'

    def setUp(self):
        self.model = interbank.Model()
        self.model.configure_json(self.CONFIG)
        self.model.configure(N=5, T=10)
        self.model.config.lender_change = interbank_lenderchange.determine_algorithm("Boltzmann")
        self.model.set_policy_recommendation(1)
        self.model.log.define_log('DEBUG')
        self.model.initialize()

        self.model.log.debug_banks()
        self.model.simulate_full()
        self.model.log.debug_banks()

    def test_values_after_execution(self):
        self.assertEqual(self.model.banks[1].C,  0.90986389240916)
        self.assertEqual(self.model.banks[3].D, 146.68244545478308)
        self.assertEqual(self.model.banks[4].E, 2.463826448774782)
        self.assertEqual(self.model.banks[0].A, 159.22071365241473)
        self.assertEqual(self.model.banks[0].R, 2.8844142730482947)
        self.assertEqual(self.model.banks[1].A, 130.7063359949939)
        self.assertEqual(self.model.banks[2].A, 150.0)
        self.assertEqual(self.model.banks[3].A, 161.6824454547831)
        self.assertEqual(self.model.banks[4].A, 194.88837679373447)
        self.assertEqual(self.model.banks[4].R, 3.6724986890527784)
        self.assertEqual(self.model.banks[4].R, self.model.banks[4].D*self.model.config.reserves)


if __name__ == '__main__':
    unittest.main()
