# -*- coding: utf-8 -*-
"""
Runs the Interbank model using montecarlo to determine the precision of RL model
@author: hector@bith.net

@date:   06/2023
"""

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import interbank_agent_ppo
import interbank
import typer
import time
import sys


def training(verbose, times, env):
    if not env:
        env = interbank_agent_ppo.InterbankPPO()
    model = PPO(MlpPolicy, env, verbose=int(verbose))
    for seed in SEEDS_FOR_TRAINING:
        env.reset(seed)
        for j in range(STEPS_BEFORE_TRAINING):
            env.environment.forward()
        model.learn(total_timesteps=times, progress_bar=verbose)
        env.close()
    # PPO(MlpPolicy, env, verbose=int(verbose), device="cuda") > it doesn't perform better
    return model


def run(model, env: interbank_agent_ppo.InterbankPPO = interbank_agent_ppo.InterbankPPO() ):
    done = False
    observations, _ = env.reset()
    while not done:
        action, _states = model.predict(observations)
        (liquidity, ir), reward, done, truncated, info = env.step(action)
        env.render(liquidity=liquidity, ir=ir)
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
        training(verbose, times, env).save(f"models/{train}")
        if verbose:
            print(f"-- total time of execution: {time.time()-t1:.2f} secs")
    else:
        if load:
            model = PPO.load(f"models/{load}")
            run(model, env)
        else:
            model = training(verbose, times, env)
            run(model, env)
    # Evaluate the trained agent
    #mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    #https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb#scrollTo=ygl_gVmV_QP7
    #print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

if __name__ == "__main__":
    typer.run(run_interactive)
