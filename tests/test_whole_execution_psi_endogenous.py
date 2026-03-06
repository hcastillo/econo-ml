# -*- coding: utf-8 -*-

import unittest
import interbank
import interbank_lenderchange
import os


class WholeExecution(unittest.TestCase):
    N = 50
    T = 50

    ALGORITHM = interbank_lenderchange.ShockedMarket
    P = 0.5

    SEED = 39393

    CONFIG = ('C_i0=30 D_i0=135 E_i0=15 L_i0=120 alfa=0.1 psi_endogenous=True allow_replacement_of_bankrupted=True '
              'beta=5 chi=0.0015 mu=0.7 normalize_interest_rate_max=1 omega=0.55 phi=0.25 psi=0.0 r_i0=0.02 '
              'reintroduce_with_median=False reserves=0.02 rho=0.3 xi=0.9 ')

    RESULTS_INTEREST_RATE = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    RESULST_EQUITY = [610.4010260722748, 496.5640169826446, 604.8361114934108, 538.1191299269303, 423.71569065912166,
                      514.823487586185, 478.0748164690517, 471.34826526931465, 452.3886049668619, 460.8501150100257,
                      463.2374870874255, 491.6959162138815, 495.9958747840984, 538.1323347847906, 514.1414699808524,
                      506.0858908505811, 558.4993744408179, 507.856868808317, 625.4122451632364, 594.6507308565875,
                      598.2585605440861, 573.0030205276975, 581.5205830456579, 520.278156616728, 457.02080508071685,
                      473.8393614328817, 506.28069668777573, 482.4265387199394, 555.3221869913184, 523.2555232397771,
                      489.2026302328822, 552.0879119855673, 555.7935903034654, 521.4553615019646, 530.8365454049897,
                      456.0355928531461, 530.4493262208397, 583.6839390345864, 507.5392266150365, 562.2250740535183,
                      553.5860357533124, 523.3354945710921, 487.15008552153574, 533.1340296049074, 416.6257706859875,
                      514.2683522221573, 552.6297509953619, 439.05280145981646, 476.8822415412331, 423.73073327904]

    def setUp(self):
        model = interbank.Model()
        model.config.lender_change = self.ALGORITHM()
        model.log.define_log('ERROR')
        model.config.lender_change.set_parameter("p", self.P)
        model.configure_json(self.CONFIG)
        model.configure(N=self.N, T=self.T)
        model.initialize(seed=self.SEED, output_directory='tests',
                         export_datafile='test_whole_execution_shockedmarket')
        model.simulate_full()
        self.generated_now_values = model.finish()

    def test_values_after_execution(self):
        self.assertEqual(list(self.generated_now_values.interest_rate), self.RESULTS_INTEREST_RATE)
        self.assertEqual(list(self.generated_now_values.equity), self.RESULST_EQUITY)


if __name__ == '__main__':
    unittest.main()
