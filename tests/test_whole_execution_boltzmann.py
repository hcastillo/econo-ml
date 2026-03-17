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
                             0.020000000000000004, 0.020000000000000004, 0.020000000000000004, 0.020000000000000004,
                             0.020000000000000004, 0.02, 0.02, 0.020000000000000004, 0.02, 0.020000000000000004,
                             0.020000000000000004, 0.020000000000000004, 0.02, 0.020000000000000004, 0.02, 0.02, 0.02,
                             0.020000000000000007, 0.02, 0.02, 0.020000000000000004, 0.02, 0.02, 0.020000000000000007,
                             0.020000000000000004, 0.020000000000000007, 0.02, 0.020000000000000004, 0.02, 0.02, 0.02,
                             0.020000000000000004, 0.020000000000000004, 0.020000000000000004, 0.02,
                             0.020000000000000004, 0.02, 0.02, 0.020000000000000004, 0.02, 0.02, 0.020000000000000004,
                             0.02, 0.020000000000000004, 0.020000000000000004, 0.02]

    RESULTS_EQUITY = [637.7392991883885, 483.21887876117694, 484.0578223706821, 492.8030087825355, 495.7619580020998,
                      467.73112313531, 476.48334051965554, 492.14190689593744, 491.1083594239135, 572.3503925484902,
                      478.1329707309962, 521.1434954224121, 474.8001154834501, 531.746628703412, 415.77967666999695,
                      530.1208442630835, 540.2693856876919, 464.7838240171506, 472.8426274966612, 528.960568319776,
                      508.96305075315655, 529.7386395507762, 570.0371960850808, 546.3719147083796, 457.79890915136514,
                      522.9071324053336, 480.92344604173195, 433.20881605383596, 426.0448456707967, 503.8526573535246,
                      553.4622995767317, 502.8235114905388, 548.7378580623755, 477.8472517566348, 480.1113157062855,
                      490.8609546337674, 525.36809473891, 431.09167620491905, 549.1899013232501, 493.3888066278605,
                      544.1299398963629, 517.9760872838534, 510.99356584484946, 545.5921194654251, 541.9540730656994,
                      483.85304756690647, 467.48692895768176, 489.3067320467585, 553.9105040929516, 506.9585621672632]

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
