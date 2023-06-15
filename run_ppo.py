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
import interbank_agent_ppo
import interbank
import typer
import time
import sys
import os

# we run STEPS_BEFORE_TRAINING times the Interbank.model() before train
STEPS_BEFORE_TRAINING: int = 5

# we train the number of times the tuple we have and using a different seed each time
SEEDS_FOR_TRAINING: tuple = (1979, 1880, 1234, 6125, 1234)
OUTPUT_PPO_TRAINING: str = "ppo.log"
MODELS_DIRECTORY = "models"


def training(verbose, times, env):
    if not env:
        env = interbank_agent_ppo.InterbankPPO()
    model = PPO(MlpPolicy, env, verbose=int(verbose))
    for seed in SEEDS_FOR_TRAINING:
        env.reset(seed)
        for j in range(STEPS_BEFORE_TRAINING):
            env.environment.forward()
        model.learn(total_timesteps=times, reset_num_timesteps=False,
                    tb_log_name=interbank.Statistics.get_export_path(OUTPUT_PPO_TRAINING) )
        env.close()
    # PPO(MlpPolicy, env, verbose=int(verbose), device="cuda") > it doesn't perform better
    return model


def run(model, env: interbank_agent_ppo.InterbankPPO = interbank_agent_ppo.InterbankPPO(), verbose: bool = False):
    done = False
    observations = env.reset()
    while not done:
        action, _states = model.predict(observations)
        observations, reward, done, info = env.step(action)
        env.render()
    env.close()


def run_interactive(log: str = typer.Option('ERROR', help="Log level messages of Interbank model"),
                    modules: str = typer.Option(None, help=f"Log only this modules (Interbank model)"),
                    logfile: str = typer.Option(None, help="File to send logs to (Interbank model)"),
                    n: int = typer.Option(interbank.Config.N, help=f"Number of banks in Interbank model"),
                    t: int = typer.Option(interbank.Config.T, help=f"Time repetitions of Interbank model"),
                    times: int = typer.Option(interbank.Config.T, help=f"Training model running times"),
                    verbose: bool = typer.Option(False, help="Verbosity of RL model"),
                    save: str = typer.Option(None, help=f"Saves the output of this execution"),
                    train: str = typer.Option(None, help=f"Trains the model and saves it this file"),
                    load: str = typer.Option(None, help=f"Loads the trained model and runs it")):
    """
        Run interactively the model
    """
    env = interbank_agent_ppo.InterbankPPO(T=t, N=n)
    env.define_log(log=log, logfile=logfile, modules=modules, script_name=sys.argv[0])
    if save:
        description = f"{type(env).__name__} T={env.environment.config.T}" + \
                      f"N={env.environment.config.N} env={load if load else '-'}"
        env.define_savefile(save, description)

    if train:
        t1 = time.time()
        model = training(verbose, times, env)
        if verbose:
            print(f"-- total time of execution of training: {time.time()-t1:.2f} secs")
        model.save(f"{MODELS_DIRECTORY}/{train}" if not train.startswith(MODELS_DIRECTORY) else train)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
        print(f"-- mean_reward={mean_reward:.2f} +/- {std_reward}")
    else:
        if load:
            model = PPO.load(f"{MODELS_DIRECTORY}/{load}" if not load.startswith(MODELS_DIRECTORY) else load)
        else:
            model = training(verbose, times, env)
        run(model, env, verbose=verbose)


if __name__ == "__main__":
    if not os.path.isdir(MODELS_DIRECTORY):
        os.mkdir(MODELS_DIRECTORY)
    typer.run(run_interactive)
