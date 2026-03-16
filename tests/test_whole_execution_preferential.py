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
              'mu=0.7 normalize_interest_rate_max=1 omega=0.55 phi=0.25 psi=0.0 r_i0=0.02 reintroduce_with_median=False'
              ' reserves=0.02 rho=0.3 xi=0.9')

    RESULTS_INTEREST_RATE = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    RESULTS_EQUITY = [735.0, 703.0227952377907, 713.7509802382801, 620.9724234496186, 616.2163160902595,
                      661.4956246046122, 614.5677002298885, 560.1800305068501, 594.0321654095042, 588.6613079722583,
                      565.8191102249601, 492.16673221974236, 533.3021136507743, 495.5437565583639, 486.4482529653551,
                      542.2878241339195, 560.847935461598, 479.66290810493723, 530.4899265644849, 540.5018370292272,
                      475.0454766141065, 466.1488479438723, 482.6402763923509, 543.0214608517981, 558.7541979724015,
                      541.0145045154642, 556.639288947029, 527.7912904473898, 473.1210911633661, 474.8929960471938,
                      556.3066749121582, 557.6965588838507, 529.8591066276053, 556.2376821502398, 569.0671398101056,
                      611.3838396576506, 559.5983153905502, 536.5343051667531, 577.463645178276, 575.0839613041592,
                      572.7607732295724, 508.37670434715625, 490.7140292106314, 570.221343158346, 541.5349527844845,
                      512.1648737061898, 506.00822475101245, 523.4642304429053, 507.0157917504977, 495.1935682148372]

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
