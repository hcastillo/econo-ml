# -*- coding: utf-8 -*-

import unittest
import interbank
import interbank_lenderchange
import os


class WholeExecution(unittest.TestCase):
    N = 50
    T = 50

    ALGORITHM = interbank_lenderchange.ShockedMarket3
    P = 0.5

    SEED = 39393

    CONFIG = ('C_i0=30 D_i0=135 E_i0=15 L_i0=120 alfa=0.1 allow_replacement_of_bankrupted=True '
              'beta=5 chi=0.0015 mu=0.7 normalize_interest_rate_max=1 omega=0.55 phi=0.25 '
              'psi=0.0 r_i0=0.02 reintroduce_with_median=False reserves=0.02 rho=0.3 xi=0.9 ')

    RESULTS_INTEREST_RATE = [0.020000000000000004,
                             0.020000000000000004,
                             0.02,
                             0.020000000000000004,
                             0.020000000000000004,
                             0.020000000000000004,
                             0.020000000000000007,
                             0.02,
                             0.02,
                             0.020000000000000004,
                             0.02,
                             0.020000000000000004,
                             0.02,
                             0.020000000000000004,
                             0.020000000000000004,
                             0.02,
                             0.020000000000000004,
                             0.02,
                             0.020000000000000004,
                             0.020000000000000004,
                             0.020000000000000004,
                             0.02,
                             0.02,
                             0.020000000000000004,
                             0.02,
                             0.020000000000000007,
                             0.020000000000000004,
                             0.020000000000000007,
                             0.020000000000000004,
                             0.020000000000000004,
                             0.019999999999999997,
                             0.02,
                             0.020000000000000004,
                             0.020000000000000004,
                             0.02,
                             0.020000000000000004,
                             0.020000000000000004,
                             0.02,
                             0.02,
                             0.02,
                             0.020000000000000007,
                             0.02,
                             0.020000000000000004,
                             0.020000000000000004,
                             0.020000000000000004,
                             0.020000000000000007,
                             0.020000000000000004,
                             0.02,
                             0.020000000000000004,
                             0.02]

    RESULTS_EQUITY = [498.775909726099, 377.56945300702597, 492.3113949309133, 521.1638964012063, 517.2670845179357,
                      449.1455711056486, 504.8073265164147, 579.0050884677444, 460.5347195508403, 366.55884128021506,
                      426.74446010175967, 330.14128904372467, 515.4347301003488, 386.0862295099382, 397.85251725283445,
                      491.88878929984236, 421.0983573603612, 518.662164738659, 427.09806183090575, 554.0702930078393,
                      445.4218934801575, 437.58031061296253, 529.9141102960268, 519.6856793522811, 404.3980300608121,
                      281.03799407894445, 445.49443944739346, 499.21827359900874, 505.72852561953846,
                      449.02951125695216, 397.6692407294271, 457.9589547900436, 390.12472291081605, 435.16778557418695,
                      473.4910533605306, 489.40515713060853, 494.53677825055854, 484.6057027875575, 506.7353124407164,
                      433.93606774023226, 433.92401300906545, 484.6751856392078, 487.47803997798377, 463.0055655029183,
                      497.9423404074396, 423.9749045057412, 540.9452938898153, 507.3143153393883, 387.08676502597064,
                      437.0980728457717]

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
        if os.path.exists('tests\\test_whole_execution_shockedmarket_erdos_renyi.json'):
            os.remove('tests\\test_whole_execution_shockedmarket_erdos_renyi.json')
        if os.path.exists('tests\\test_whole_execution_shockedmarket_erdos_renyi.png'):
            os.remove('tests\\test_whole_execution_shockedmarket_erdos_renyi.png')

    def test_values_after_execution(self):
        self.assertEqual(list(self.generated_now_values.ir), self.RESULTS_INTEREST_RATE)
        self.assertEqual(list(self.generated_now_values.equity), self.RESULTS_EQUITY)


if __name__ == '__main__':
    unittest.main()
