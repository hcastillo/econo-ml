# -*- coding: utf-8 -*-

import unittest
import exp_runner
import interbank
import interbank_lenderchange
import numpy as np
import pandas as pd
from pathlib import Path
import os


class ExpRunnerTestCase(unittest.TestCase):

    def setUp(self):
        if os.path.exists('output/results.csv'):
            os.remove("output/results.csv")
        self.runner = MockedRunner()
        self.runner.do()

    def test_values_after_execution(self):
        self.assertIsInstance(self.runner, exp_runner.ExperimentRun)
        self.assertEqual(len(self.runner.parameters['p']), 10)
        self.assertEqual(self.runner.parameters['p'][1], 0.11111111111111112)
        self.assertTrue(Path("output/results.csv").is_file())


if __name__ == '__main__':
    unittest.main()


class MockedRunner(exp_runner.ExperimentRun):
    N = 5
    T = 100
    MC = 1

    ALGORITHM = interbank_lenderchange.RestrictedMarket
    OUTPUT_DIRECTORY = "output"

    parameters = {  # items should be iterable:
        "p": np.linspace(0.1, 0.2, num=10),
    }

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = interbank.Model()
        model.config.lender_change = self.ALGORITHM()
        model.config.lender_change.set_parameter("p", execution_parameters["p"])
        return pd.DataFrame({'seed': [seed_random]})
