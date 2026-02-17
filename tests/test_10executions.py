import unittest
import interbank
import interbank_lenderchange


class ValuesAfterExecutionTestCase(unittest.TestCase):

    CONFIG = 'C_i0=30 D_i0=135 E_i0=15 L_i0=120 N=50 T=50 alfa=0.1 allow_replacement_of_bankrupted=True beta=5 chi=0.015 mu=0.7 omega=0.55 phi=0.25 psi=0.3 r_i0=0.02 reintroduce_with_median=False reserves=0.02 rho=0.3 xi=0.3'
    # CONFIG = 'C_i0=30 D_i0=135 E_i0=15 L_i0=120 N=50 T=1000 alfa=0.1 allow_replacement_of_bankrupted=True asset_i_avg_ir=0.0 asset_j_avg_ir=0.0 beta=5 c_avg_ir=0.0 chi=0.015 detailed_equity=False max_value_psi=0.99 mu=0.7 normalize_interest_rate_max=-2 omega=0.55 p_avg_ir=0.0 phi=0.025 psi=0.3 psi_endogenous=False r_i0=0.02 reintroduce_with_median=False reserves=0.02 rho=0.3 xi=0.3'
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
        # print(self.model.banks[1].C)
        # print(self.model.banks[3].D)
        # print(self.model.banks[4].E)
        # print(self.model.banks[0].A)
        # print(self.model.banks[0].R)
        # print(self.model.banks[1].A)
        # print(self.model.banks[2].A)
        # print(self.model.banks[3].A)
        # print(self.model.banks[4].A)
        # print(self.model.banks[4].R)
        self.assertEqual(self.model.banks[1].C,  29.151792021559686)
        self.assertEqual(self.model.banks[3].D, 176.21777404471345)
        self.assertEqual(self.model.banks[4].E, 15)
        self.assertEqual(self.model.banks[0].A, 176.70008308425264)
        self.assertEqual(self.model.banks[0].R, 3.2340016616850527)
        self.assertEqual(self.model.banks[1].A, 151.88958369546907)
        self.assertEqual(self.model.banks[2].A, 132.3365981860403)
        self.assertEqual(self.model.banks[3].A, 191.21777404471342)
        self.assertEqual(self.model.banks[4].A, 122.73199580871527)
        self.assertEqual(self.model.banks[4].R, 2.1546399161743053)
        self.assertEqual(self.model.banks[4].R, self.model.banks[4].D*self.model.config.reserves)


if __name__ == '__main__':
    unittest.main()
