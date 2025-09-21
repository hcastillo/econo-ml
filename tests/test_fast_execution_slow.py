# -*- coding: utf-8 -*-

import unittest
import interbank
import interbank_lenderchange
import os

class WholeExecution(unittest.TestCase):
    N = 50
    T = 200


    ALGORITHM = interbank_lenderchange.ShockedMarket3
    P = 0.5

    SEED = 39393

    CONFIG = 'C_i0=30 D_i0=135 E_i0=15 L_i0=120 alfa=0.1 allow_replacement_of_bankrupted=True beta=5 chi=0.0015 mu=0.7 omega=0.55 phi=0.25 psi=0.0 r_i0=0.02 reintroduce_with_median=False reserves=0.02 rho=0.3 xi=0.9 '

    maxDiff = None

    def setUp(self):
        model_slow = interbank.Model()
        model_slow.config.lender_change = self.ALGORITHM()
        model_slow.log.define_log('ERROR')
        model_slow.config.lender_change.set_parameter("p", self.P)
        model_slow.configure_json(self.CONFIG)
        model_slow.configure(N=self.N, T=self.T)
        model_slow.initialize(seed=self.SEED, output_directory='tests', export_datafile='test_slow')
        model_slow.simulate_full()
        self.slow_model = model_slow.finish()

        model_fast = interbank.ModelOptimized()
        model_fast.config.lender_change = self.ALGORITHM()
        model_fast.log.define_log('ERROR')
        model_fast.config.lender_change.set_parameter("p", self.P)
        model_fast.configure_json(self.CONFIG)
        model_fast.configure(N=self.N, T=self.T)
        model_fast.initialize(seed=self.SEED, output_directory='tests', export_datafile='test_fast')
        model_fast.simulate_full()
        self.fast_model = model_fast.finish()
        if os.path.exists('tests\\test_slow.gdt'):
            os.remove('tests\\test_slow.gdt')
            os.remove('tests\\test_slow_erdos_renyi.png')
            os.remove('tests\\test_slow_erdos_renyi.json')
        if os.path.exists('tests\\test_fast.gdt'):
            os.remove('tests\\test_fast.gdt')
            os.remove('tests\\test_fast_erdos_renyi.png')
            os.remove('tests\\test_fast_erdos_renyi.json')


    def test_values_after_execution(self):
        self.assertEqual(self.slow_model.to_json(), self.fast_model.to_json())


if __name__ == '__main__':
    unittest.main()
