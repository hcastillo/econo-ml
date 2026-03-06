# -*- coding: utf-8 -*-

import unittest
import interbank
import interbank_lenderchange
import os


class WholeExecution(unittest.TestCase):
    N = 50
    T = 50

    ALGORITHM = interbank_lenderchange.Preferential
    M = 25

    SEED = 39393

    CONFIG = ('C_i0=30 D_i0=135 E_i0=15 L_i0=120 alfa=0.1 allow_replacement_of_bankrupted=True beta=5 chi=0.0015 '
              'mu=0.7 normalize_interest_rate_max=1 omega=0.55 phi=0.25 psi=0.0 r_i0=0.02 reintroduce_with_median=False '
              'reserves=0.02 rho=0.3 xi=0.9 ')

    RESULTS_INTEREST_RATE = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    RESULTS_EQUITY = [735.0, 733.7017682127106, 646.900905103199, 636.3043843972137, 596.0497528632375,
                      612.9671102419156, 576.4225239450066, 527.7336551263518, 539.7822415972257, 581.2842802056703,
                      525.3822656592334, 589.5961589589351, 510.4650980772452, 564.1275151904006, 501.70564163205563,
                      512.9329966223261, 540.1833553891358, 532.8500717279771, 618.1757009002391, 558.154089906433,
                      503.89544910151255, 384.7552298857946, 492.4348598693322, 525.1977943431061, 554.7291536239853,
                      512.7987862284072, 508.0082863796644, 569.0699177183831, 545.4413411678094, 515.3358018458728,
                      506.86317787051684, 495.44695752059886, 626.1754534436311, 525.9581805915401, 536.799961129779,
                      473.05656965513396, 529.9299315738111, 409.2060283623504, 504.3101648173292, 514.1833373981068,
                      583.1727377186365, 490.7972577427058, 508.8984320619357, 571.8365196553721, 435.749015203538,
                      457.79113325768697, 460.86190152903583, 434.77562331170816, 452.1203946910257, 516.8159844188447]

    def setUp(self):
        model = interbank.Model()
        model.log.define_log('ERROR')
        model.config.lender_change = self.ALGORITHM()
        model.config.lender_change.set_parameter("m", self.M)
        model.configure_json(self.CONFIG)
        model.configure(N=self.N, T=self.T)
        model.initialize(seed=self.SEED, output_directory='tests',
                         export_datafile='test_whole_execution_preferential')
        model.simulate_full()
        self.generated_now_values = model.finish()
        if os.path.exists('tests\\test_whole_execution_preferential_barabasi_pref.json'):
            os.remove('tests\\test_whole_execution_preferential_barabasi_pref.json')
        if os.path.exists('tests\\test_whole_execution_preferential_barabasi_pref.png'):
            os.remove('tests\\test_whole_execution_preferential_barabasi_pref.png')

    def test_values_after_execution(self):
        self.assertEqual(list(self.generated_now_values.interest_rate), self.RESULTS_INTEREST_RATE)
        self.assertEqual(list(self.generated_now_values.equity), self.RESULTS_EQUITY)


if __name__ == '__main__':
    unittest.main()
