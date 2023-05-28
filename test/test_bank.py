import unittest,interbank


class BankTestCase(unittest.TestCase):
    def setUp(self):
        interbank.Config.N = 10
        self.bank1 = interbank.Bank(6)
        self.bank1.lender = self.bank1.newLender()
        self.bank2 = interbank.Bank(0)
        self.bank2.lender = self.bank2.newLender()
        self.bank3 = interbank.Bank(9)
        self.bank3.lender = self.bank3.newLender()

    def test_at_beginning_of_list(self):
        for i in range(10):
            current = self.bank2.lender
            new     = self.bank2.newLender()
            self.assertNotEqual(current, self.bank2.getId(),
                            'newLender() is not the same bank that borrows')
            self.assertNotEqual(new, current, 'newLender() is not the previous')

    def test_at_end_of_list(self):
        for i in range(10):
            current = self.bank3.lender
            new     = self.bank3.newLender()
            self.assertNotEqual(current, self.bank3.getId(),
                            'newLender() is not the same bank that borrows')
            self.assertNotEqual(new, current, 'newLender() is not the previous')

    def test_at_middle_of_list(self):
        for i in range(10):
            current = self.bank1.lender
            new     = self.bank1.newLender()
            self.assertNotEqual(current, self.bank1.getId(),
                            'newLender() is not the same bank that borrows')
            self.assertNotEqual(new, current, 'newLender() is not the previous')


