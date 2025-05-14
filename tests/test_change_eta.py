import unittest

import numpy as np

import interbank
import interbank_lenderchange


class ValuesAfterExecutionTestCase(unittest.TestCase):
    def setUp(self):
        self.model1 = interbank.Model()
        self.model1.configure(N=5, T=30)  # under t<20 we force a fixed P and no changes
        self.model1.config.lender_change = interbank_lenderchange.determine_algorithm("Boltzmann")
        self.model1.initialize()
        #self.model1.log.define_log('DEBUG')
        self.model1.set_policy_recommendation(1)
        self.model1.simulate_full()

        self.model2 = interbank.Model()
        self.model2.configure(N=5, T=30)
        self.model2.config.lender_change = interbank_lenderchange.determine_algorithm("Boltzmann")
        self.model2.initialize()
        #self.model1.log.define_log('DEBUG')
        self.model2.set_policy_recommendation(0)
        self.model2.simulate_full()

    def test_values_after_execution(self):
        # model1 and model2 have different eta and so they will evolve different:
        self.assertNotEqual(self.model1.banks[0].get_id(), self.model2.banks[0].get_id())
        self.assertFalse(np.array_equal(self.model1.statistics.B, self.model2.statistics.B))
        self.assertFalse(np.array_equal(self.model1.statistics.interest_rate, self.model2.statistics.interest_rate))


if __name__ == '__main__':
    unittest.main()
