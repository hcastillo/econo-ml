# -*- coding: utf-8 -*-

import unittest
import interbank
import interbank_lenderchange


class WholeExecution(unittest.TestCase):
    N = 50
    T = 50
    T_LONG = 1000

    maxDiff = None

    ALGORITHM = interbank_lenderchange.Boltzmann
    P = 0.5

    SEED = 39393

    CONFIG = ('C_i0=30 D_i0=135 E_i0=15 L_i0=120 N=50 T=50 alfa=0.1 allow_replacement_of_bankrupted=True beta=5'
              ' chi=0.0015 mu=0.7 normalize_interest_rate_max=1 omega=0.55 phi=0.25 psi=0.0 r_i0=0.02'
              ' reintroduce_with_median=False reserves=0.02 rho=0.3 xi=0.9')

    RESULTS_INTEREST_RATE = [0.020000000000000007, 0.020000000000000004, 0.020000000000000004, 0.02,
                             0.020000000000000004, 0.020000000000000004, 0.02, 0.020000000000000004,
                             0.020000000000000004, 0.02, 0.02, 0.020000000000000004, 0.020000000000000004,
                             0.020000000000000004, 0.020000000000000007, 0.020000000000000007, 0.02,
                             0.020000000000000004, 0.02, 0.02, 0.02, 0.020000000000000004, 0.020000000000000004, 0.02,
                             0.020000000000000004, 0.02, 0.02, 0.020000000000000004, 0.020000000000000007, 0.02,
                             0.020000000000000004, 0.020000000000000004, 0.02, 0.02, 0.02, 0.020000000000000004,
                             0.020000000000000004, 0.020000000000000004, 0.02, 0.020000000000000004, 0.02,
                             0.020000000000000004, 0.02, 0.020000000000000007, 0.020000000000000004,
                             0.020000000000000007, 0.020000000000000004, 0.02, 0.020000000000000004,
                             0.020000000000000004]

    RESULTS_EQUITY = [637.7392991883885, 483.21887876117694, 480.98866353771785, 489.73384994957127, 484.39279387922903,
                      449.24543690991374, 517.7193678999982, 540.5644183286074, 508.3936098903543, 558.6475939874765,
                      462.9642215122804, 574.8472610816879, 552.4782523587612, 489.3763556509862, 528.6024025568454,
                      529.8640632105348, 500.3978942109184, 574.7127348360277, 461.2239214498258, 539.889913405503,
                      523.6017481933925, 502.3826144188803, 517.8577989375933, 525.4095731170991, 511.4318300786891,
                      504.9702762205871, 531.9804976437439, 480.46860864409535, 439.06237897613886, 512.454584871297,
                      429.51228313652075, 500.6790643437659, 495.09958915193107, 503.03497210279374, 565.8203761462231,
                      481.87454233984045, 471.0647039493076, 469.13047744216, 554.5795254777447, 529.299051988041,
                      558.0615154465614, 523.9157498445642, 543.9789539032846, 539.0015200020633, 514.7422150454514,
                      443.97691379025497, 502.6796829656364, 413.22074475481617, 427.4841150920533, 468.0051887944535]

    def setUp(self):
        model = interbank.Model()
        model.log.define_log('ERROR')
        model.config.lender_change = self.ALGORITHM()
        model.configure_json(self.CONFIG)
        model.configure(N=self.N, T=self.T)
        model.configure(normalize_interest_rate_max=0)
        model.initialize(seed=self.SEED, output_directory='tests',
                         export_datafile='test_whole_execution_boltzmann')
        model.simulate_full()
        self.generated_now_values = model.finish()

    def test_values_after_execution(self):
        self.assertEqual(list(self.generated_now_values.interest_rate), self.RESULTS_INTEREST_RATE)
        self.assertEqual(list(self.generated_now_values.equity), self.RESULTS_EQUITY)


if __name__ == '__main__':
    unittest.main()
