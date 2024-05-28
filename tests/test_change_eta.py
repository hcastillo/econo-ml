import unittest
import interbank


class ValuesAfterExecutionTestCase(unittest.TestCase):
    def setUp(self):
        self.model1 = interbank.Model()
        self.model1.configure(N=5, T=10)
        self.model1.set_policy_recommendation(1)
        self.model1.initialize()
        self.model1.simulate_full()

        self.model2 = interbank.Model()
        self.model2.configure(N=5, T=10)
        self.model2.set_policy_recommendation(0)
        self.model2.initialize()
        self.model2.simulate_full()

    def test_values_after_execution(self):
        self.assertNotEqual(self.model1.banks[1].C, self.model2.banks[1].C)
        self.assertNotEqual(self.model1.banks[3].C, self.model2.banks[3].C)
        self.assertNotEqual(self.model1.banks[4].C, self.model2.banks[4].C)
        self.assertNotEqual(self.model1.banks[2].C, self.model2.banks[2].C)


if __name__ == '__main__':
    unittest.main()
