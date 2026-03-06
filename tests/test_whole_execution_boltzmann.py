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

    RESULTS_INTEREST_RATE = [0.02, 0.02, 0.02, 0.020000000000000004, 0.020000000000000004, 0.020000000000000004, 0.02,
                             0.02, 0.020000000000000004, 0.020000000000000004, 0.02, 0.02, 0.02, 0.020000000000000007,
                             0.020000000000000004, 0.02, 0.020000000000000004, 0.020000000000000007,
                             0.020000000000000004, 0.020000000000000004, 0.02, 0.020000000000000004,
                             0.020000000000000004, 0.02, 0.02, 0.02, 0.020000000000000004, 0.020000000000000004,
                             0.020000000000000004, 0.020000000000000007, 0.02, 0.02, 0.02, 0.020000000000000004,
                             0.020000000000000004, 0.020000000000000004, 0.02, 0.02, 0.020000000000000007, 0.02, 0.02,
                             0.02, 0.020000000000000004, 0.02, 0.02, 0.020000000000000004, 0.020000000000000004,
                             0.020000000000000004, 0.020000000000000004, 0.020000000000000004]
    RESULTS_EQUITY = [500.43539048668686, 525.2324757589863, 487.6721762810356, 382.8357965601082, 430.98243194195993,
                      448.1262284516474, 475.22702235963374, 404.30643826206483, 516.8961735650529, 447.24740252656795,
                      472.41418105349527,
                      513.7297989399133, 562.0935799346973, 517.7750298725716, 487.7890853508584, 543.2642524001762,
                      461.63481062053734,
                      445.8560825395012, 524.922826176302, 462.41230215243183, 483.8725425818347, 563.7105619830324,
                      539.6925450248448,
                      504.3480808316068, 483.6982953357896, 501.39134305319766, 529.9504181311372, 470.3221434255823,
                      470.41517136560566,
                      500.342569390698, 536.7580628258079, 486.0785808340767, 477.8028752512214, 446.6410875982249,
                      512.2354282933537,
                      459.3340113334463, 497.82668131171476, 534.4903211910678, 533.4699849952675, 574.179632713974,
                      495.9126103200871,
                      531.6875699068819, 546.940869966341, 516.1360994621707, 576.9189479047054, 488.9978436407511,
                      545.8821357861525,
                      493.46097160542547, 514.0195512429442, 455.0538388035971]

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
