# -*- coding: utf-8 -*-
"""
Runs the Interbank model using montecarlo to determine the precision of RL model
@author: hector@bith.net

@date:   06/2023
"""

import interbank
import numpy as np
import typer
import sys
import tqdm
from scipy.stats import bernoulli

NUM_SIMULATIONS = 50


class Montecarlo:
    """
    Create self.simulations of Interbank.model, using Montecarlo MCMC
    """

    simulations = NUM_SIMULATIONS

    def __init__(self, environment, simulations: int = None):
        self.environment = environment
        self.data = []
        self.summary = {}
        if simulations:
            self.simulations=simulations

    def do_one_simulation(self,iteration):
        """
        Set to the initial state the Interbank.Model and run a new simulation, using each time a different policy
        recommendation
        """
        self.environment.initialize(dont_seed=(iteration>1))
        X = bernoulli(0.5)
        policies = X.rvs(self.environment.config.T)
        for i in range(self.environment.config.T):
            self.environment.set_policy_recommendation(policies[i])
            self.environment.forward()
        self.environment.finish()
        return self.environment.statistics.get_data()

    def run(self):
        for i in tqdm.tqdm(range(self.simulations), total=self.simulations, desc="mc"):
            self.data.append(self.do_one_simulation())

    def save_column(self, prefix, name, column):
        filename = interbank.Statistics.get_export_path(f"{prefix}_{name}.txt")
        print(filename)
        total = np.zeros(self.simulations, dtype=float)
        with open(filename, 'w', encoding="utf-8") as savefile:
            head = "# t"
            for i in range(self.simulations):
                head += f"\t{i:18}#"
            head += f"\n# {name}\n"
            savefile.write(head)
            for j in range(self.environment.config.T):
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
        self.save_column(filename, "policy", interbank.Statistics.DATA_POLICY)
        self.save_column(filename, "ir", interbank.Statistics.DATA_IR)
        self.save_column(filename, "fitness", interbank.Statistics.DATA_FITNESS)
        self.save_column(filename, "liquidity", interbank.Statistics.DATA_LIQUIDITY)
        self.save_column(filename, "bankruptcy", interbank.Statistics.DATA_BANKRUPTCY)
        self.save_column(filename, "best_lender", interbank.Statistics.DATA_BEST_LENDER)
        self.save_column(filename, "best_clients", interbank.Statistics.DATA_BEST_LENDER_CLIENTS)
        self.save_column(filename, "credit_channels", interbank.Statistics.DATA_CREDIT_CHANNELS)
        self.save_summary(filename)


def run_interactive(log: str = typer.Option('ERROR', help="Log level messages of Interbank model"),
                    modules: str = typer.Option(None, help=f"Log only this modules (Interbank model)"),
                    logfile: str = typer.Option(None, help="File to send logs to (Interbank model)"),
                    n: int = typer.Option(interbank.Config.N, help=f"Number of banks in Interbank model"),
                    t: int = typer.Option(interbank.Config.T, help=f"Time repetitions of Interbank model"),
                    simulations: int = typer.Option(NUM_SIMULATIONS, help=f"Number of MonteCarlo simulations"),
                    save: str = typer.Option(None, help=f"Saves the output of this execution")):
    """
        Run interactively the model
    """
    environment = interbank.Model(T=t, N=n)
    environment.log.define_log(log=log, logfile=logfile, modules=modules, script_name=sys.argv[0])
    simulation = Montecarlo(environment=environment, simulations=simulations)
    simulation.run()
    simulation.save(save)


if __name__ == "__main__":
    typer.run(run_interactive)