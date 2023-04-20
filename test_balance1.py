import unittest,bank_net
from mock import patch, Mock

class ValuesTestCase(unittest.TestCase):

    shocks = [
        # t=0, for each bank
        {  "shock1" :[ -15, 5 ], "shock2": [ -10, 10 ], },
        # t=1... etc
        {  "shock1" :[ -15, 5 ], "shock2": [ -10, 10 ], },
    ]

    def mockedShock(whichShock):
        for bank in bank_net.Model.banks:
            bank.newD = bank.D + ValuesTestCase.shocks[ bank_net.Model.t ][whichShock][bank.id]
            bank.ΔD = bank.newD - bank.D
            bank.D = bank.newD
            if bank.ΔD > 0:
                bank.C += bank.ΔD
            bank_net.Statistics.incrementD[bank_net.Model.t] += bank.ΔD
        bank_net.Status.debugBanks(details=False, info=whichShock)

    @patch.object(bank_net, "doShock",mockedShock)
    def setUp(self):
        bank_net.Config.N = 2
        bank_net.Config.T = 1

        bank_net.Status.defineLog('DEBUG')
        bank_net.Model.initilize()
        bank_net.Statistics.reset()
        bank_net.Status.debugBanks()

        bank_net.Model.doSimulation()

        bank_net.Status.debugBanks()


    def test_values_after_execution(self):
        #self.assertEqual( bank_net.Model.banks[1].C, 0.13382435876260956)
        #self.assertEqual( bank_net.Model.banks[3].D, 135)
        #self.assertEqual( bank_net.Model.banks[4].E, -13.26847831510539)
        pass
