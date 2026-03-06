# -*- coding: utf-8 -*-

import unittest
import interbank
import interbank_lenderchange
import os


class WholeExecution(unittest.TestCase):
    N = 50
    T = 50

    ALGORITHM = interbank_lenderchange.RestrictedMarket
    P = 0.5

    SEED = 39393

    CONFIG = ('C_i0=30 D_i0=135 E_i0=15 L_i0=120 alfa=0.1 allow_replacement_of_bankrupted=True beta=5 '
              'chi=0.0015 mu=0.7 normalize_interest_rate_max=1 omega=0.55 phi=0.25 psi=0.0 r_i0=0.02 '
              'reintroduce_with_median=False reserves=0.02 rho=0.3 xi=0.9 ')

    RESULTS_INTEREST_RATE = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    RESULTS_EQUITY = [610.4010260722748, 476.3372784422315, 475.0746049022757, 556.6730377997383,
                      524.9401927211318, 469.8016882764989, 435.7858850652228, 479.47164114184613,
                      531.6186444811356, 517.3307193223334, 500.686093562556, 413.64034910979206,
                      510.06616874094243, 474.97791383183204, 526.8120922005734,
                      524.0787497276858, 511.34520566234335, 500.50515341590756, 491.50226082731734,
                      491.07796202297715, 513.6329732658808, 546.55092738448,
                      619.7325410134279, 559.8440754091225, 523.6113620236176,
                      552.8205958661852, 500.2154944210776, 562.8901438290887,
                      589.5996267310591, 538.3158987690747, 611.8107470800359, 505.9481398201678,
                      518.7331049330506, 487.19753054971727, 499.3902217349186,
                      538.7415999615944, 544.3417776939391, 542.1936536352732, 490.3693191154022,
                      558.4766234986241, 473.36238127434456, 609.675553672432, 581.4357161790335,
                      609.2133570554037, 557.1812117306753, 608.7444975437436, 513.3938182488528, 588.5883894093665,
                      585.7095041673538, 571.7101612235043]

    def setUp(self):
        model = interbank.ModelOptimized()
        model.config.lender_change = self.ALGORITHM()
        model.log.define_log('ERROR')
        model.config.lender_change.set_parameter("p", self.P)
        model.configure_json(self.CONFIG)
        model.configure(N=self.N, T=self.T)
        model.initialize(seed=self.SEED, output_directory='tests',
                         export_datafile='test_whole_execution_restrictedmarket')
        model.simulate_full()
        self.generated_now_values = model.finish()
        if os.path.exists('tests\\test_whole_execution_restrictedmarket_erdos_renyi_0.json'):
            os.remove('tests\\test_whole_execution_restrictedmarket_erdos_renyi_0.json')
        if os.path.exists('tests\\test_whole_execution_restrictedmarket_erdos_renyi_0.png'):
            os.remove('tests\\test_whole_execution_restrictedmarket_erdos_renyi_0.png')

    def test_values_after_execution(self):
        self.assertEqual(list(self.generated_now_values.interest_rate), self.RESULTS_INTEREST_RATE)
        self.assertEqual(list(self.generated_now_values.equity), self.RESULTS_EQUITY)


if __name__ == '__main__':
    unittest.main()
