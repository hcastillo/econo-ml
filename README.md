# econo-ml: Reinforce Learning Policy Recommendation for Interbank Network Stability

- Auxiliary files:
  - *requirements.txt*: list of the necessary python packages


## - Interbank model

  - *interbank.py*: use to execute standalone the Interbank simulation.
    - It accepts command line options. For instance, you can execute this:
    
          interbank.py --log DEBUG --n 150 --t 2000
          interbank.py --save results.gdt --p 0.5 eta=0.35 param=X     
    - When it is used as a package, the sequence should be:

          import interbank
          model = interbank.Model()
          model.config.configure( param=x )
          model.forward()
          μ = model.get_current_fitness()
          model.set_policy_recommendation( ŋ=0.5 )


  - *colab_interbank.ipynb*: Notebook version of the standalone *interbank.py* with the same results but plotted using Bokeh.
  - *labplot2_interbank.lml*: [LabPlot2](https://labplot.org/) file to plot the results of the *interbank.py*. By the way the best way is to use [Gretl](https://gretl.sourceforge.net/) as an export format.
  - *interbank_lenderchange.py*: It contains the different algorithms that control the change of lender in the model.
  - *exp_runner.py*: A prototype for executing experiments with different parameters and using MonteCarlo.
        It uses *concurrent.futures* to process with different threads the executions.
        It also can use [Dask](https://www.dask.org/) to distribute among different nodes the load.
  - *exp_runner_comparer.py*: A derivation of the former prototype though to compare the evolution with *p* (probability of attachment in an Erdos-Renyi graph) in the *x* axis and other parameters accross the *y* axis.
  - *experiments/**: directory with all the experiments conducted. The results of that executions are stored in a folder determined inside each experiment.
  - *algorithm.drawio* and *algorithm.drawio.pf*: the [draw.io](https://www.drawio.com/) and PDF schema of the algorithm used in the model to propagate shocks and to balance sheets.
  - 


## - RL with Stable Baselines3
  - *interbank_agent.py*: agent to test using PPO
  - *run_ppo.py*: run and simulate with PPO agent
  - *run_td3.py*: run and simulate with TD3 algorithm 
  - *models/XXXX.zip*: instances of Gymnasium.env trained to use with *run_XXXX.py*
  - *plot_ppo.py*: auxiliary creator of plots to interprete the results of PPO
  - Usage:

          # train first and save the model env:
          run_ppo.py --train ppo_10000 --t 10000 --verbose

          # use the trained env and generate a simulation of T=1000 with Interbank model
          run_ppo.py --load ppo_10000 --save results_ppo.txt


