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
import random
import tqdm

NUM_SIMULATIONS = 50

class Metropolis:
    def metropolis_hastings( self, likelihood, proposal_distribution, initial_state,
            num_samples, stepsize=0.5, burnin=0.2):
        """ Compute the Markov Chain Monte Carlo
            https://exowanderer.medium.com/metropolis-hastings-mcmc-from-scratch-in-python-c21e53c485b7
        Args:
            likelihood (function): a function handle to compute the likelihood
            proposal_distribution (function): a function handle to compute the
              next proposal state
            initial_state (list): The initial conditions to start the chain
            num_samples (integer): The number of samples to compte,
              or length of the chain
            burnin (float): a float value from 0 to 1.
              The percentage of chain considered to be the burnin length

        Returns:
            samples (list): The Markov Chain,
              samples from the posterior distribution
        """
        samples = []

        # The number of samples in the burn in phase
        idx_burnin = int(burnin * num_samples)

        # Set the current state to the initial state
        curr_state = initial_state
        curr_likeli = self.likelihood(curr_state)

        for i in range(num_samples):
            # The proposal distribution sampling and comparison
            #   occur within the mcmc_updater routine
            curr_state, curr_likeli = self.mcmc_updater(
                curr_state=curr_state,
                curr_likeli=curr_likeli,
                likelihood=likelihood,
                proposal_distribution=proposal_distribution
            )

            # Append the current state to the list of samples
            if i >= idx_burnin:
                # Only append after the burnin to avoid including
                #   parts of the chain that are prior-dominated
                samples.append(curr_state)
        return samples

    def mcmc_updater(self, curr_state, curr_likeli,
                     likelihood, proposal_distribution):
        """ Propose a new state and compare the likelihoods

        Given the current state (initially random),
          current likelihood, the likelihood function, and
          the transition (proposal) distribution, `mcmc_updater` generates
          a new proposal, evaluate its likelihood, compares that to the current
          likelihood with a uniformly samples threshold,
        then it returns new or current state in the MCMC chain.

        Args:
            curr_state (float): the current parameter/state value
            curr_likeli (float): the current likelihood estimate
            likelihood (function): a function handle to compute the likelihood
            proposal_distribution (function): a function handle to compute the
              next proposal state

        Returns:
            (tuple): either the current state or the new state
              and its corresponding likelihood
        """
        # Generate a proposal state using the proposal distribution
        # Proposal state == new guess state to be compared to current
        proposal_state = proposal_distribution(curr_state)

        # Calculate the acceptance criterion
        prop_likeli = likelihood(proposal_state)
        accept_crit = prop_likeli / curr_likeli

        # Generate a random number between 0 and 1
        accept_threshold = np.random.uniform(0, 1)

        # If the acceptance criterion is greater than the random number,
        # accept the proposal state as the current state
        if accept_crit > accept_threshold:
            return proposal_state, prop_likeli

        # Else
        return curr_state, curr_likeli

    def likelihood(self, x):
        # Standard Normal Distribution
        # An underlying assumption of linear regression is that the residuals
        # are Gaussian Normal Distributed; often, Standard Normal distributed
        return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

    def proposal_distribution(self, x, stepsize=0.5):
        # Select the proposed state (new guess) from a Gaussian distribution
        #  centered at the current state, within a Guassian of width `stepsize`
        return np.random.normal(x, stepsize)


    def run(self):
        np.random.seed(42)
        initial_state = 0  # Trivial case, starting at the mode of the likelihood
        num_samples = int(20)
        burnin = 0.2

        samples = self.metropolis_hastings(
            self.likelihood,
            self.proposal_distribution,
            initial_state,
            num_samples,
            burnin=burnin
        )
        print(samples)


class MonteCarlo:
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

    def do_one_simulation(self, seed=None):
        """
        Set to the initial state the Interbank.Model and run a new simulation, using each time a different policy
        recommendation
        """
        self.environment.initialize(seed=seed)
        random.seed(seed+1e2)
        for i in range(self.environment.config.T):
            self.environment.set_policy_recommendation(random.randint(0,2))
            self.environment.forward()
        self.environment.finish()
        return self.environment.statistics.get_data()

    def run(self):
        srd = np.random.randn(self.simulations)
        for i in tqdm.tqdm(range(self.simulations), total=self.simulations, desc="mc"):
            self.data.append(self.do_one_simulation(abs(int(srd[i]*10000))))

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
        self.save_column(filename, "best_lender", interbank.Statistics.DATA_BESTLENDER)
        self.save_column(filename, "best_clients", interbank.Statistics.DATA_BESTLENDER_CLIENTS)
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
    global simulation, environment
    environment = interbank.Model(T=t, N=n)
    environment.log.define_log(log=log, logfile=logfile, modules=modules, script_name=sys.argv[0])
    simulation = MonteCarlo(environment, simulations=simulations)
    simulation.run()
    simulation.save(save)

if __name__ == "__main__":
    typer.run(run_interactive)