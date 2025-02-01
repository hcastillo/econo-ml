# -*- coding: utf-8 -*-

import unittest
import interbank
import interbank_lenderchange


BANKS = 50
T = 5
P = 0.3

class RestrictedMarketTestCase(unittest.TestCase):
    def setUp(self):
        self.model = interbank.Model()
        self.model.configure(N=BANKS)
        self.model.config.lender_change = interbank_lenderchange.determine_algorithm("RestrictedMarket")
        self.model.config.lender_change.set_parameter("p", P)
        self.model.config.lender_change.initialize_bank_relationships(self.model)
        self.graphs = []

        # we execute T times the model:
        for t in range(T):
            self.graphs.append(self.model.config.lender_change.banks_graph)
            self.model.forward()

    def test_values_after_execution(self):
        self.assertEqual(self.model.config.lender_change.parameter['p'], P)
        self.assertIsInstance(self.model.config.lender_change, interbank_lenderchange.RestrictedMarket)

        for t in range(T):
            self.assertEqual(self.graphs[t].is_directed(), True)
            # only one incoming link for each node: only one lender, so 0 or 1 the value:
            for i in range(BANKS):
                self.assertLessEqual(len(self.graphs[t].in_edges(i)), 1)
            # the graph should be same in all the steps:
            self.assertEqual(self.graphs[t], self.graphs[0])

if __name__ == '__main__':
    unittest.main()

