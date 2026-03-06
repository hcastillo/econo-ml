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

    RESULTS_INTEREST_RATE = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    RESULTS_EQUITY = [530.7616598344503, 564.5696845798575, 446.43975045656657, 474.22843881280926, 547.1530668555607,
                      466.89440316516686, 497.35353669412217, 401.62247697754407, 495.8739222283853, 514.248507870857,
                      473.5118870927837, 397.55654390434336, 305.5336241853371, 511.4831474725823, 404.6592474790016,
                      499.61981127945387, 451.1721679586855, 544.5012240069241, 410.0789197190563, 441.36668732300444,
                      447.42205203529323, 476.883706721479, 403.4990718811843, 516.2234696512214, 433.9407128067145,
                      483.12590451480236, 508.77441562262516, 413.19804138220326, 513.2509576130761, 517.4734365228592,
                      493.5650106665125, 445.3526804516605, 528.9656066724717, 409.6387776028571, 410.9503110445452,
                      467.89557611311136, 502.31855848024765, 480.6709529775459, 531.3977313221808, 450.2411219051547,
                      394.9363337464178, 413.56721302071963, 452.42878299041666, 379.23605805863326, 501.5662842907196,
                      470.0202602331347, 494.6600848246751, 524.2479816148616, 432.33417143833833, 468.15036917563157]

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
        self.assertEqual(list(self.generated_now_values.interest_rate), self.RESULTS_INTEREST_RATE)
        self.assertEqual(list(self.generated_now_values.equity), self.RESULTS_EQUITY)


if __name__ == '__main__':
    unittest.main()
