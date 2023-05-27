# econo-ml: Reinforce Learning Policy Recommendation for Interbank Network Stability

1. Auxiliary files:
- *requeriments.txt*: list of the necessary python packages

1. Interbank model usage:
- *bank_net.py*: use to execute standalone the Interbank simulation. It accepts command line options.
When it is used as a package, the sequence should be:
    import bank_net
    
    bank_net.config( T=150,Φ=0.32 )
    bank_net.do_step()
    Modelμ = bank_net.get_current_fitness()
    bank_net.set_policy_recommendation( 0.5 )

# bank_net.set_policy_recommendation()
- *bank_net.ipynb*: Notebook version of the standalone bank_net.py with the same results but plotted using Bokeh.
- *bank_net.lml*: LabPlot2 file to plot the results of the bank_net.py.


2. Reinforce learning with Pytorch and Stable Baselines3:
- *agent_ppo.py*: agent to test using PPO
- *run_ppo.py*: run and simulate with PPO agent
- *agent_XXX.py* and *run_XXX*: other agents and algorithms to simulate

