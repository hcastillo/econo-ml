# -*- coding: utf-8 -*-
"""
Runs the RL PPO to estimate the policy reccomendation training (--training)
or predicts the next steps using the previous saved training (--predict)

@author: hector@bith.net
@date:   05/2023
"""

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import interbank_agent_ppo
import interbank
import typer


def training(verbose, times, env):
    if not env:
        env = interbank_agent_ppo.InterbankPPO()
    return PPO(MlpPolicy, env, verbose=int(verbose)).learn(total_timesteps=times, progress_bar=verbose)


def run(model, env):
    if not env:
        env = interbank_agent_ppo.InterbankPPO()
    rewards = []
    num_episodes = 5
    for i in range(num_episodes):
        rewards_step = []
        done = False
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # env.render()
            rewards_step.append(reward)
        env.render()
        # print(info)
        rewards.append(sum(rewards_step))
    return rewards


def run_interactive(log: str = typer.Option('ERROR', help="Log level messages of Interbank model"),
                    verbose: bool = typer.Option(False,help="Verbosity of RL model"),
                    modules: str = typer.Option(None, help=f"Log only this modules (Interbank model)"),
                    logfile: str = typer.Option(None, help="File to send logs to (Interbank model)"),
                    n: int = typer.Option(interbank.Config.N, help=f"Number of banks in Interbank model"),
                    t: int = typer.Option(interbank.Config.T, help=f"Time repetitions of Interbank model"),
                    times: int = typer.Option(interbank.Config.T, help=f"Training model running times"),
                    save: str = typer.Option(None,help=f"Trains the model and saves it this file"),
                    load: str = typer.Option(None, help=f"Loads the trained model and runs it"),
                   ):
        """
            Run interactively the model
        """
        env = interbank_agent_ppo.InterbankPPO(T=t, N=n)
        env.define_log(log, logfile, modules)
        if save:
            training(verbose, times, env).save( save )
        else:
            if load:
                run( PPO.load(load, env) , env)
            else:
                model = training(verbose, times, env)
                run(model, env)


if __name__ == "__main__":
    typer.run(run_interactive)
