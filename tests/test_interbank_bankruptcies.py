import unittest
import interbank
import interbank_lenderchange
from mock import patch


class BankruptedTestCase(unittest.TestCase):
    """
    It tests when a bank is bankrupted, but not replaced (experiment exp_surviving)
    the mecanism of removing it from the array of banks
    """

    BANKS = 10
    T = 5
    P = 0.001

    def setUp(self):
        self.model = interbank.Model()
        self.model.test = True
        self.model.configure(N=self.BANKS, T=self.T)
        self.model.log.define_log(log='DEBUG', script_name=self.id().split('.')[0])
        self.model.initialize()

        # bank#0 and #3 as failed, and #1 is lender with #2 and #3 as borrowers:
        self.model.banks[0].failed = True
        self.model.banks[2].lender = 1
        self.model.banks[4].lender = 1
        self.model.banks[1].active_borrowers[2] = 200
        self.model.banks[1].active_borrowers[3] = 300
        self.model.banks[3].failed = True
        # bank#9 is lender and #8 its borrower:
        self.model.banks[9].active_borrowers[8] = 10
        self.model.banks[8].lender = 9
        self.model.config.allow_replacement_of_bankrupted = False
        # to recognize banks, we force the C to a value (id*100):
        for i in range(self.BANKS):
            self.model.banks[i].C = i*100
        # call to the function of removing them:
        self.model.replace_bankrupted_banks()

    def test_values_after_execution(self):
        # 10 banks - 3 = 7 still active
        self.assertEqual(len(self.model.banks),8)
        # #1 now is #0, borrowers were #2 and #3 and now it is #1 (#3 was failed):
        self.assertEqual(self.model.banks[0].C, 100)
        #self.assertEqual(len(self.model.banks[0].active_borrowers), 1)
        #self.assertEqual(self.model.banks[0].active_borrowers.keys()[0], 1)
        # #2 now is #1,
        self.assertEqual(self.model.banks[1].C, 200)
        #self.assertEqual(self.model.banks[1].lender, 0)
        # #4 now is #2
        self.assertEqual(self.model.banks[2].C, 400)
        # #8 and #9 are now #6 and #7:
        self.assertEqual(self.model.banks[6].C, 800)
        self.assertEqual(self.model.banks[7].C, 900)
        #self.assertEqual(len(self.model.banks[6].active_borrowers), 1)
        #self.assertEqual(self.model.banks[7].active_borrowers.keys()[0], 6)
        #self.assertEqual(self.model.banks[6].lender, 7)


if __name__ == '__main__':
    unittest.main()
