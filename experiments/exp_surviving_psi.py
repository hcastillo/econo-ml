#!/usr/bin/env python
# coding: utf-8
"""
Executor for the interbank model using different values for the lc RestrictedMarket
   to determine number of survivings banks (no replacement of bankrupted)
   depending on p

@author: hector@bith.net
"""
import exp_runner_surviving


class RestrictedMarketSurvivingRun(exp_runner_surviving.SurvivingRun):
    N = 50
    T = 1000
    MC = 15

    OUTPUT_DIRECTORY = "/experiments/251123.surviving.psi"
    DESCRIPTION_TITLE = "\\psi"

    parameters = {
        "p": [0.1],
    }
    config = {
        "psi": [0, 0.3333, 0.6666, 0.999]
    }
    LENGTH_FILENAME_PARAMETER = 3
    LENGTH_FILENAME_CONFIG = 5

    SEED_FOR_EXECUTION = 318994
    COLORS_VARIABLE = 'psi'

    COMPARING_DATA_IN_SURVIVING = False

    def plot_surviving(self):
        print("Plotting surviving data...")
        self.save_surviving_csv(self.data_of_surviving_banks_avg, '_surviving', self.max_t)
        self.save_surviving_csv(self.data_of_failures_rationed_avg, '_failures_rationed', self.max_t)
        self.generate_plot(f"Failures rationed accum p={self.parameters['p']}",
                           "_failures_rationed_accum_"+self.get_filename_for_parameters(self.parameters)+".png",
                           self.data_of_failures_rationed_accum_avg, self.all_models, self.max_t,
                           data_comparing_data_surviving=self.data_of_failures_accum_avg)
        self.save_surviving_csv(self.data_of_failures_rationed_accum_avg,
                                '_failures_rationed_accum_'+self.get_filename_for_parameters(self.parameters),
                                self.max_t)


if __name__ == "__main__":
    runner = exp_runner_surviving.Runner(RestrictedMarketSurvivingRun)
    experiment = runner.do()
    if experiment:
        experiment.generate_data_surviving()
        experiment.plot_surviving()

