# -*- coding: utf-8 -*-
"""
Runs the Interbank model using montecarlo to determine the precision of RL model
@author: hector@bith.net

@date:   06/2023
"""

import interbank
import numpy as np
import argparse
import sys
import tqdm
from scipy.stats import bernoulli

NUM_SIMULATIONS = 50


class Montecarlo:
    """
    Create self.simulations of Interbank.model, using Montecarlo MCMC
    """

    simulations = NUM_SIMULATIONS

    def __init__(self, environment, simulations: int = None, fixed_eta: int = None):
        self.interbank_model = environment
        self.data = []
        self.summary = {}
        self.fixed_eta = fixed_eta
        if simulations:
            self.simulations = simulations

    def do_one_simulation(self, iteration):
        """
        Set to the initial state the Interbank.Model and run a new simulation, using each time a different policy
        recommendation
        """
        self.interbank_model.initialize(dont_seed=(iteration > 1))
        if self.fixed_eta is not None:
            self.interbank_model.set_policy_recommendation(self.fixed_eta)
        else:
            bernoulli_policy = bernoulli(0.5)
            policies = bernoulli_policy.rvs(self.interbank_model.config.T)
        for i in range(self.interbank_model.config.T):
            if self.fixed_eta is None:
                self.interbank_model.set_policy_recommendation(policies[i])
            self.interbank_model.forward()
        self.interbank_model.finish()
        return self.interbank_model.statistics.get_data()

    def run(self):
        self.interbank_model.limit_to_two_policies()
        for i in tqdm.tqdm(range(self.simulations), total=self.simulations):
            self.data.append(self.do_one_simulation(i))

    def save_column(self, prefix, name, column):
        filename = interbank.Statistics.get_export_path(f"{prefix}_{name}.txt")
        total = np.zeros(self.simulations, dtype=float)
        with open(filename, 'w', encoding="utf-8") as savefile:
            head = "# t"
            for i in range(self.simulations):
                head += f"\t{i:18}#"
            head += f"\n# {name}\n"
            savefile.write(head)

            for j in range(self.interbank_model.config.T):
                line = f"{j:3}"
                for i in range(self.simulations):
                    line += f"\t{self.data[i][column][j]:19}"
                    total[i] += self.data[i][column][j]
                savefile.write(line + "\n")
        self.summary[name] = total

    def save_summary(self, filename):
        filename = interbank.Statistics.get_export_path(f"{filename}.txt")
        with open(filename, 'w', encoding="utf-8") as savefile:
            head = "# n"
            for item in self.summary:
                head += f"\tsum_{item}"
            savefile.write(head+"\n")
            for i in range(self.simulations):
                line = f"{i:3}"
                for j in self.summary:
                    line += f"\t{self.summary[j][i]:19}"
                savefile.write(line + "\n")

    def save(self, filename):
        for column in interbank.DataColumns:
            self.save_column(filename, column.name.lower(), column.value)
        self.save_summary(filename)


def run_interactive():
    """
        Run interactively the model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default='ERROR', help="Log level messages of Interbank model")
    parser.add_argument("--modules", default=None, help=f"Log only this modules (Interbank model)")
    parser.add_argument("--logfile", default=None, help="File to send logs to (Interbank model)")
    parser.add_argument("--n", type=int, default=interbank.Config.N,
                        help=f"Number of banks in Interbank model")
    parser.add_argument("--t", type=int, default=interbank.Config.T,
                        help=f"Time repetitions of Interbank model")
    parser.add_argument("--simulations", type=int, default=NUM_SIMULATIONS,
                        help=f"Number of MonteCarlo simulations")
    parser.add_argument("--fixed_eta", type=int, default=None, help="Fix the eta ŋ to this value (0,1)")
    parser.add_argument("--save", default=None, help=f"Saves the output of this execution")
    args = parser.parse_args()

    environment = interbank.Model(T=args.t, N=args.n)
    environment.log.define_log(log=args.log, logfile=args.logfile, modules=args.modules, script_name=sys.argv[0])
    simulation = Montecarlo(environment=environment, simulations=args.simulations, fixed_eta=args.fixed_eta)
    simulation.run()
    simulation.save(args.save)


if __name__ == "__main__":
    run_interactive()