import unittest

import numpy as np

import interbank
import interbank_lenderchange

P = 0.2
SEED= 1234
class ValuesAfterExecutionTestCase(unittest.TestCase):
    def setUp(self):
        self.model1 = interbank.Model()
        self.model1.configure(N=50, T=50)
        self.model1.config.lender_change = interbank_lenderchange.determine_algorithm("ShockedMarket3")
        self.model1.config.lender_change.set_parameter("p", P)
        self.model1.initialize(seed=SEED)
        #self.model1.log.define_log('DEBUG')
        self.model1.simulate_full()
        self.model2 = interbank.Model()
        self.model2.configure(N=50, T=50)
        self.model2.config.lender_change = interbank_lenderchange.determine_algorithm("ShockedMarket3")
        self.model2.config.lender_change.set_parameter("p", P)
        self.model2.initialize(seed=SEED)
        #self.model2.log.define_log('DEBUG')
        self.model2.simulate_full()

    def test_values_after_execution(self):
        self.assertTrue(np.array_equal(self.model1.statistics.B, self.model2.statistics.B))
        self.assertTrue(np.array_equal(self.model1.statistics.interest_rate, self.model2.statistics.interest_rate))

if __name__ == '__main__':
    unittest.main()
