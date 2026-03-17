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

    RESULST_EQUITY = [596.659141736821, 487.8923977651297, 594.9474039552134, 446.21698920341817, 531.3393955487867,
                      478.7706445153492, 496.3270243490334, 446.605150911116, 530.217329184942, 554.8226632422686,
                      558.7565271802016, 510.34474685436237, 564.7598416984819, 526.9896387297555, 407.010857073206,
                      467.48657711805055, 386.28738183921735, 459.46883825692265, 535.5269910491648, 474.0904913695698,
                      425.0469194702448, 460.27033394661856, 464.4610868266459, 363.83138401620096, 472.60648910930024,
                      483.2187085260582, 530.6616784975965, 498.31830474313574, 506.16351892531435, 475.25803985584645,
                      523.8441192739092, 474.6635570887987, 547.339125911053, 532.0700035962061, 566.8953761313307,
                      472.4814826964991, 579.4693285925792, 491.93068664254895, 498.5335060865492, 524.6608875274159,
                      413.3475056344627, 419.4772820642311, 485.22678479813374, 530.7201724887133, 531.5716009793977,
                      514.2387459580415, 530.815535565524, 580.601106572504, 387.6781778906027, 431.3890959974844]

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
