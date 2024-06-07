# -*- coding: utf-8 -*-
"""
Runs the RL PPO to estimate the policy recommendation training (--training)
or predicts the next steps using the previous saved training (--predict)

@author: hector@bith.net
@date:   05/2023
"""

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from interbank_agent import InterbankAgent
import numpy as np
import interbank
import time
import sys
import run_mc
import os
import argparse

# we run STEPS_BEFORE_TRAINING times the Interbank.model() before train
STEPS_BEFORE_TRAINING: int = 5

# we train the number of times the tuple we have and using a different seed each time
SEEDS_FOR_TRAINING: tuple = (1979, 1880, 1234, 6125, 1234, 9999)

OUTPUT_TRAINING: str = "td3.txt"
MODELS_DIRECTORY = "models"

# same as MC model repetitions, to compare
NUM_SIMULATIONS = 50

aux = None


class TD3Simulation(run_mc.Montecarlo):
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
        return self.env.interbank_model.statistics.get_data()


def training(verbose, times, env, logs):
    if not env:
        env = InterbankAgent()
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=int(verbose), tensorboard_log=logs)
    for seed in SEEDS_FOR_TRAINING:
        env.reset(seed=seed)
        for j in range(STEPS_BEFORE_TRAINING):
            env.interbank_model.forward()
        model.learn(total_timesteps=times, log_interval=10,
                    tb_log_name=env.interbank_model.statistics.get_export_path(OUTPUT_TRAINING))
        env.close()
    return model

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
                        help=f"Number of MC simulations")
    parser.add_argument("--save", default=None, help=f"Saves the output of this execution")
    parser.add_argument("--verbose", default=False, help="Verbosity of RL model",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--train", default=None, help=f"Trains the model and saves it this file")
    parser.add_argument("--load", default=None, help=f"Loads the trained model and runs it")
    parser.add_argument("--logs", default="logs", help=f"Log dir for Tensorboard")
    args = parser.parse_args()
    env = InterbankAgent(T=args.t, N=args.n)
    env.interbank_model.log.define_log(log=args.log, logfile=args.logfile,
                                       modules=args.modules, script_name=sys.argv[0])
    if not os.path.isdir(args.logs):
        os.mkdir(args.logs)
    description = f"{type(env).__name__} T={env.interbank_model.config.T}" + \
                  f"N={env.interbank_model.config.N} env={args.load if args.load else '-'}"

    # train ------------------------
    if args.train:
        t1 = time.time()
        model = training(args.verbose, args.t, env, args.logs)
        env.define_savefile(args.save, description)
        if args.verbose:
            print(f"-- total time of execution of training: {time.time() - t1:.2f} secs")
        model.save(f"{MODELS_DIRECTORY}/{args.train}" if not args.train.startswith(MODELS_DIRECTORY) else args.train)
    # run -------------------------
    else:
        if args.load:
            model = TD3.load(f"{MODELS_DIRECTORY}/{args.load}"
                             if not args.load.startswith(MODELS_DIRECTORY) else args.load)
        else:
            model = training(args.verbose, args.t, env, args.logs)
        if args.simulations == 1:
            env.define_savefile(args.save, description)
        simulation = TD3Simulation(model, env, simulations=args.simulations)
        simulation.run()
        simulation.save(args.save)


if __name__ == "__main__":
    if not os.path.isdir(MODELS_DIRECTORY):
        os.mkdir(MODELS_DIRECTORY)
    run_interactive()