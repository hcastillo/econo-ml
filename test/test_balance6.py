import unittest
import interbank
import interbank_testclass
from mock import patch, Mock


class Balance1TestCase(interbank_testclass.InterbankTest):
    """
    test borrower can pay the loan using C, no loan and no second shock
    """
    def initial_values(self):
        self.setBank(bank=self.model.banks[0], C=10.0, L=15.0, D=13.0, E=12.0)
        self.setBank(bank=self.model.banks[1], C=10.0, L=15.0, D=14.0, E=11.0)
        self.setBank(bank=self.model.banks[2], C=10.0, L=15.0, D=16.0, E=9.0)

    @patch.object(interbank.Model, "doShock", interbank_testclass.mockedShock)
    def setUp(self):
        self.configureTest(N=3, T=10, shocks=[
            {"shock1": [-5, 3, -3], "shock2": [0, 0, -3], },
            {"shock1": [-5, 3, 0], "shock2": [0, 0, -6], },
            {"shock1": [1, -5, 2], "shock2": [0, -5, 2], },
            {"shock1": [0, 3, 0], "shock2": [0, 0, -2], },
            {"shock1": [1, 3, 2], "shock2": [0, -5, 2], },
            {"shock1": [2, 3, 1], "shock2": [0, -3, 0], },
            {"shock1": [-3, 3, -3], "shock2": [0, 0, -2], },
            {"shock1": [0, 3, 0], "shock2": [0, 0, -2], },
            {"shock1": [1, 3, 2], "shock2": [0, -5, 2], },
            {"shock1": [2, 3, 1], "shock2": [0, -3, 0], },
        ])
        self.initial_values()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[0], C=4.0, L=15.0, D=7.0, E=12.0)
        self.assertBank(bank=self.model.banks[1], C=11.0, L=15.0, D=15.0, E=11.0, s=14)
        self.assertBank(bank=self.model.banks[2], C=5.0, L=8.333333333333332, D=9.0, E=4.33333333333333333, s=5)


if __name__ == '__main__':
    unittest.main()
