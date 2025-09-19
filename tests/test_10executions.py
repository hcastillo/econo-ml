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
        print(self.model.banks[1].C)
        print(self.model.banks[3].D)
        print(self.model.banks[4].E)
        print(self.model.banks[0].A)
        print(self.model.banks[0].R)
        print(self.model.banks[1].A)
        print(self.model.banks[2].A)
        print(self.model.banks[3].A)
        print(self.model.banks[4].A)
        print(self.model.banks[4].R)
        self.assertEqual(self.model.banks[1].C,  14.905461675774493)
        self.assertEqual(self.model.banks[3].D, 119.97971468015639)
        self.assertEqual(self.model.banks[4].E, 7.439122075553262)
        self.assertEqual(self.model.banks[0].A, 272.84741396380343)
        self.assertEqual(self.model.banks[0].R, 5.156948279276068)
        self.assertEqual(self.model.banks[1].A, 137.55067922008195)
        self.assertEqual(self.model.banks[2].A, 142.33328152717868)
        self.assertEqual(self.model.banks[3].A, 134.97971468015638)
        self.assertEqual(self.model.banks[4].A, 185.20428278303106)
        self.assertEqual(self.model.banks[4].R, 3.5553032141495566)
        self.assertEqual(self.model.banks[4].R, self.model.banks[4].D*self.model.config.reserves)


if __name__ == '__main__':
    unittest.main()
