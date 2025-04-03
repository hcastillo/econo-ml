# -*- coding: utf-8 -*-

import unittest
from mock import patch
import interbank
import tests.interbank_testclass


class Balance3TestCase(tests.interbank_testclass.InterbankTest):
    """
    test lender gives two loans: one of it could be paid correctly and the second one
         makes fails to the lender
    """

    #       #0             #1             #2
    #   -----------     ------------    -----------
    #   C=9.6| D=20     C=19.6| D=30    C=9.6| D=20
    #   L=20 | E=10     L=14  | E=4     L=20 | E=10
    #   R=0.4|          R=0.4           R=0.4
    #
    #   shock1=-15     shock1=+5        shock=-20 --> #0 obtains from #1 a loan of 5.40
    #                                                 #2 obtains from #2 a loan of 10.4
    #   shock2=-3      shock2=3         shock= 0  --> #0 returns loan
    #                                                 #2 fails and affects E in #1
    #                                                 #1 fails due to loss of E
    #
    #       #0             #1             #2
    #   ------------    ------------    -----------
    #   C=9.49| D=20    failed          failed
    #   L=20  | E=9.89
    #   R=0.4 |

    def initialValues(self):
        self.setBank(bank=self.model.banks[0], C=10.0, L=20.0, D=20.0, E=10.0, lender=1)
        self.setBank(bank=self.model.banks[1], C=20.0, L=14.0, D=30.0, E=4.0)
        self.setBank(bank=self.model.banks[2], C=10.0, L=20.0, D=20.0, E=10.0, lender=1)

    @patch.object(interbank.Model, "do_shock", tests.interbank_testclass.mockedShock)
    def setUp(self):
        self.configureTest(N=3, T=1,
                           shocks=[
                               {"shock1": [-15, 5,-20], "shock2": [15, 3, 0], },
                           ])
        self.initialValues()
        self.doTest()

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[2], paid_loan=0, bankrupted=True)
        self.assertBank(bank=self.model.banks[1], bankrupted=True)
        self.assertBank(bank=self.model.banks[0], paid_loan=0, bankrupted=False)
        # even #1 has failed, the bad debt of #1 is taken into account:
        # the loan of #2 not returned partially, generates 4.4 in bad debt
        self.assertEqual(self.model.statistics.B[0],4.4)


if __name__ == '__main__':
    unittest.main()
