# -*- coding: utf-8 -*-
"""
Runs the RL PPO to estimate the policy recommendation training (--training)
or predicts the next steps using the previous saved training (--predict)

@author: hector@bith.net
@date:   05/2023
"""

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import interbank_agent
import interbank
import typer
import time
import sys
import run_mc
import os

# we run STEPS_BEFORE_TRAINING times the Interbank.model() before train
STEPS_BEFORE_TRAINING: int = 5

# we train the number of times the tuple we have and using a different seed each time
SEEDS_FOR_TRAINING: tuple = (1979, 1880, 1234, 6125, 1234, 9999)

# we train the number of times the tuple we have and using a different seed each time
OUTPUT_TRAINING: str = "ppo.log"
MODELS_DIRECTORY = "models"

# same as MC model repetitions, to compare
NUM_SIMULATIONS = 50


class PPOSimulation(run_mc.Montecarlo):
    """
    Create self.simulations of Interbank model, using an agent
    """
    simulations = NUM_SIMULATIONS    

    def __init__(self, model, env, simulations: int = None):
        self.env = env
        self.interbank_model = self.env.interbank_model
        self.model = model
        self.data = []
        self.summary = {}
        if simulations:
            self.simulations = simulations

    def do_one_simulation(self, iteration):
        """
        Set to the initial state the Interbank.Model and run a new simulation, using the recommendation suggested
        by the model
        """
        done = False
        observations, _info = self.env.reset(dont_seed=(iteration > 1))
        while not done:
            action, _states = self.model.predict(observations)
            observations, reward, done, _truncated, _info = self.env.step(action)
            if self.simulations == 1:
                self.env.render()
        self.env.close()
        return self.interbank_model.statistics.get_data()
    
    @staticmethod
    def get_models_path(filename):
        if not filename.startswith(MODELS_DIRECTORY):
            filename = f"{MODELS_DIRECTORY}/{filename}"
        return filename if filename.endswith('.zip') else f"{filename}.zip"


def training(verbose, times, env, logs):
    if not env:
        env = interbank_agent.InterbankAgent()
    model = PPO(MlpPolicy, env, verbose=int(verbose), tensorboard_log=logs)
    for seed in SEEDS_FOR_TRAINING:
        env.reset(seed=seed)
        for j in range(STEPS_BEFORE_TRAINING):
            env.interbank_model.forward()
        model.learn(total_timesteps=times, reset_num_timesteps=False,
                    tb_log_name=interbank.Statistics.get_export_path(OUTPUT_TRAINING))
        env.close()
    return model


def run_interactive(log: str = typer.Option('ERROR', help="Log level messages of Interbank model"),
                    modules: str = typer.Option(None, help=f"Log only this modules (Interbank model)"),
                    logfile: str = typer.Option(None, help="File to send logs to (Interbank model)"),
                    n: int = typer.Option(interbank.Config.N, help=f"Number of banks in Interbank model"),
                    t: int = typer.Option(interbank.Config.T, help=f"Time repetitions of Interbank model"),
                    simulations: int = typer.Option(NUM_SIMULATIONS, help=f"Number of simulations"),
                    save: str = typer.Option(None, help=f"Saves the output of this execution"),
                    verbose: bool = typer.Option(False, help="Verbosity of RL model"),
                    train: str = typer.Option(None, help=f"Trains the model and saves it this file"),
                    load: str = typer.Option(None, help=f"Loads the trained model and runs it"),
                    logs: str = typer.Option("logs", help=f"Log dir for Tensorboard")):
    """
        Run interactively the model
    """
    env = interbank_agent.InterbankAgent(T=t, N=n)
    env.interbank_model.log.define_log(log=log, logfile=logfile, modules=modules, script_name=sys.argv[0])
    if not os.path.isdir(logs):
        os.mkdir(logs)
    description = f"{type(env).__name__} T={env.interbank_model.config.T}" + \
                  f"N={env.interbank_model.config.N} env={load if load else '-'}"

    # train ------------------------
    if train:
        t1 = time.time()
        model = training(verbose, t, env, logs)
        env.define_savefile(save, description)
        if verbose:
            print(f"-- total time of execution of training: {time.time() - t1:.2f} secs")
        model.save(PPOSimulation.get_models_path(train))
    # run -------------------------
    else:
        if load:
            model = PPO.load(PPOSimulation.get_models_path(load))
        else:
            model = training(verbose, t, env, logs)
        if simulations == 1:
            env.define_savefile(save, description)
        simulation = PPOSimulation(model, env, simulations=simulations)
        simulation.run()
        simulation.save(save)


if __name__ == "__main__":
    if not os.path.isdir(MODELS_DIRECTORY):
        os.mkdir(MODELS_DIRECTORY)
    typer.run(run_interactive)
