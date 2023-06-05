import unittest
import interbank


class BankTestCase(unittest.TestCase):
    def setUp(self):
        self.model = interbank.Model()
        self.model.test = True
        self.model.configure(N=10)

    def test_new_lender(self):
        for i in range(10):
            current = self.model.banks[i].lender
            new = self.model.banks[i].newLender()
            self.assertNotEqual(current, self.model.banks[i].getId(),
                                f"newLender() of {i} is not the same bank that borrows")
            self.assertNotEqual(new, current, f"newLender() of {i} is not the previous")
