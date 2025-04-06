import unittest
import interbank


class BankTestCase(unittest.TestCase):
    def setUp(self):
        self.model = interbank.Model()
        self.model.test = True
        self.model.configure(N=10)
        self.model.initialize()

    def test_new_lender(self):
        for i in range(10):
            current = self.model.banks[i].lender
            new = self.model.config.lender_change.new_lender(self.model,self.model.banks[i])
            self.assertNotEqual(current, self.model.banks[i].get_id(),
                                f"new_lender() of {i} is not the same bank that borrows")
            self.assertNotEqual(new, current, f"new_lender() of {i} is not the previous")
