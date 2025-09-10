import unittest
import interbank
import interbank_lenderchange


class BankTestCase(unittest.TestCase):
    def setUp(self):
        self.model = interbank.Model()
        self.model.test = True
        self.model.config.lender_change = interbank_lenderchange.determine_algorithm("Boltzmann")
        self.model.configure(N=10)
        self.model.initialize()
        self.model.enable_backward()
        self.model.forward()
        self.model.log.define_log(log='DEBUG')

    def test_backwards(self):
        before_C = []
        before_D = []
        temporary_C = []
        temporary_D = []
        banks_ids = []
        for i in range(10):
            before_C.append( self.model.banks[i].C )
            before_D.append( self.model.banks[i].D )
            banks_ids.append(self.model.banks[i].get_id())

        self.model.forward()
        # self.model.log.debug_banks()
        for i in range(10):
            temporary_C.append( self.model.banks[i].C )
            temporary_D.append( self.model.banks[i].D )
        self.model.backward()
        self.model.log.debug_banks()
        for i in range(10):
            if banks_ids[i] == self.model.banks[i].get_id():
                bankd_id = self.model.banks[i].get_id()
                self.assertNotEqual(before_D[i], temporary_D[i],
                                    f"r of #{bankd_id} should change from t={self.model.t} to t={self.model.t+1}")
                self.assertEqual(before_D[i], self.model.banks[i].D,
                                 f"r of #{bankd_id} should be the same as t={self.model.t} if we go backward")
