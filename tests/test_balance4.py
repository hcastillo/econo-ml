# -*- coding: utf-8 -*-

import unittest
from mock import patch
import interbank
import tests.interbank_testclass


# 3. bank does fire sales but it is covered ok
# -------------------

class Balance3TestCase(tests.interbank_testclass.InterbankTest):

    def initialValues(self):
        interbank.Config.ro = 0.4 # fire sale cost
        self.setBank(bank=self.model.banks[0], C=0.0, L=8.0, D=3.0, E=5.0)
        self.setBank(bank=self.model.banks[1], C=10.0, L=20.0, D=20.0, E=10.0)

    @patch.object(interbank.Model, "do_shock", tests.interbank_testclass.mockedShock)
    def setUp(self):
        self.configureTest( N=2,T=1,
                            shocks=[
                                {"shock1": [0, 0], "shock2": [-3, 0], },
                            ])
        self.initialValues()
        self.doTest()
        interbank.Config.ro = 0.3 # fire sale cost

    def test_values_after_execution(self):
        self.assertBank(bank=self.model.banks[0], C=0, L=0.5, D=0, E=0.5, paid_loan=0, bankrupted=False)
        self.assertBank(bank=self.model.banks[1], C=10, L=20, D=20, E=10)





if __name__ == '__main__':
    unittest.main()