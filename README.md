# econo-ml: Reinforce Learning Policy Recommendation for Interbank Network Stability

- Auxiliary files:
  - *requeriments.txt*: list of the necessary python packages


- Interbank model usage:
  - *interbank.py*: use to execute standalone the Interbank simulation. It accepts command line options.
  When it is used as a package, the sequence should be:

        import interbank
        interbank.config( T=150,Φ=0.32 )
        interbank.do_step()
        Modelμ = interbank.get_current_fitness()
        interbank.set_policy_recommendation( 0.5 )

  - *interbank.ipynb*: Notebook version of the standalone interbank.py with the same results but plotted using Bokeh.
  - *interbank.lml*: LabPlot2 file to plot the results of the interbank.py.


- Reinforce learning with Pytorch and Stable Baselines3:
  - *agent_ppo.py*: agent to test using PPO
  - *run_ppo.py*: run and simulate with PPO agent
  - *agent_XXX.py* and *run_XXX*: other agents and algorithms to simulate

