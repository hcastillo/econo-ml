# econo-ml: Reinforce Learning Policy Recommendation for Interbank Network Stability

- Auxiliary files:
  - *requirements.txt*: list of the necessary python packages


## - Interbank model

  - *interbank.py*: use to execute standalone the Interbank simulation.
    - It accepts command line options. For instance, you can execute this:
    
          interbank.py --log DEBUG --n 150 --t 2000
    - When it is used as a package, the sequence should be:

          import interbank
          model = interbank.Model()
          model.config.configure( param=x )
          model.forward()
          μ = model.get_current_fitness()
          model.set_policy_recommendation( ŋ=0.5 )


  - *interbank.ipynb*: Notebook version of the standalone *interbank.py* with the same results but plotted using Bokeh.
  - *interbank.lml*: LabPlot2 file to plot the results of the *interbank.py*.


## - RL with Stable Baselines3
  - *interbank_agent_ppo.py*: agent to test using PPO
  - *run_ppo.py*: run and simulate with PPO agent
  - *interbank_agent_XXX.py* and *run_XXX*: other agents and algorithms to simulate
  - *models/XXXX.zip*: instances of Gymnasium.env trained to use with *run_XXXX.py*
  - Usage:

          # train first and save the model env:
          run_ppo.py --train ppo_10000 --times 10000 --verbose

          # use the trained env and generate a simulation of T=1000 with Interbank model
          run_ppo.py --load ppo_10000 --save results_ppo.txt


