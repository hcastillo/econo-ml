# -*- coding: utf-8 -*-

import unittest
import interbank
import interbank_lenderchange


BANKS = 50
T = 5
P = 0.6

class RestrictedMarketTestCase(unittest.TestCase):
    def setUp(self):
        self.model = interbank.Model()
        self.model.configure(N=BANKS)
        self.model.config.lender_change = interbank_lenderchange.determine_algorithm("ShockedMarket3")
        self.model.config.lender_change.set_parameter("p", P)
        self.model.initialize()
        self.model.config.lender_change.initialize_bank_relationships(self.model)
        self.graphs = []


        # we execute T times the model:
        for t in range(T):
            self.graphs.append(self.model.config.lender_change.banks_graph)
            self.model.forward()

    def test_values_after_execution(self):
        self.assertEqual(self.model.config.lender_change.parameter['p'], P)
        self.assertIsInstance(self.model.config.lender_change, interbank_lenderchange.ShockedMarket3)

        for t in range(T):
            self.assertEqual(self.graphs[t].is_directed(), False)
            # many undirected links for each node (as p=0.6):
            for i in range(BANKS):
                self.assertGreater(len(self.graphs[t].edges(i)), 1)
            # the graph should be different in each step:
            self.assertNotEqual(self.graphs[t], self.graphs[t-1])

if __name__ == '__main__':
    unittest.main()

