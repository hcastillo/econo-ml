# -*- coding: utf-8 -*-
import unittest
import interbank
import tests.interbank_testclass
from mock import patch


class BalanceTestCase(tests.interbank_testclass.InterbankTest):
    """
    test borrower obtains a loan, and after it pays normally it with a positive second shock
    """
    CONFIG = 'C_i0=30 D_i0=135 E_i0=15 L_i0=120 N=50 T=1000 alfa=0.1 allow_replacement_of_bankrupted=True asset_i_avg_ir=0.0 asset_j_avg_ir=0.0 beta=5 c_avg_ir=0.0 chi=0.015 detailed_equity=False max_value_psi=0.99 mu=0.7 normalize_interest_rate_max=-2 omega=0.55 p_avg_ir=0.0 phi=0.025 psi=0.3 psi_endogenous=False r_i0=0.02 reintroduce_with_median=False reserves=0.02 rho=0.3 xi=0.3'

    def initialValues(self):
        self.setBank(bank=self.model.banks[0], C=10.0, L=15.0, D=15.0, E=10.0)
        self.setBank(bank=self.model.banks[1], C=10.0, L=15.0, D=15.0, E=10.0)

    @patch.object(interbank.Model, "determine_shock_value", tests.interbank_testclass.determine_shock_value_mocked)
    def setUp(self):
        self.configureTest( N=2,T=1,
                            shocks=[
                                {"shock1": [-13, 13], "shock2": [20, -20], },
                            ], lc="Boltzmann")
        self.initialValues()
        self.interest_rate_for_loan_of_bank0 = self.model.banks[0].get_loan_interest()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[0], C=16.499200000000002, L=15.0, D=22.0, E=9.9392,
                        paid_loan=3.040000000000001, paid_profits=0.06080000000000002)
        self.assertEqual(0.02, self.interest_rate_for_loan_of_bank0)
        self.assertEqual(0.06080000000000002, 3.040000000000001*self.interest_rate_for_loan_of_bank0)
        self.assertBank(bank=self.model.banks[1], C=22.439999999999998, L=14.333333333333323, D=8.0,
                        E=9.594133333333327, s=22.439999999999998)


if __name__ == '__main__':
    unittest.main()