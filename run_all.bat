python interbank.py --log DEBUG --logfile interbank.log --eta 1 
python interbank.py --eta 1 --save interbank_eta1
python interbank.py --eta 0 --save interbank_eta0

python run_ppo.py --train models\ppo --verbose
python run_ppo.py --load ppo --verbose --save ppo

python run_mc.py --save mc

python plot.py --column 2 --load ppo1 --save ppo1t
#  0 POLICY
#  1 FITNESS
#  2 LIQUIDITY
#  3 IR
#  4 BANKRUPTCY
#  5 BEST_LENDER + BEST_LENDER_CLIENTS
#  6 CREDIT_CHANNELS
#  7 RATIONING
#  8 LEVERAGE


python plot_cumulative_fitness.py --save cum_fitness_ppo --type sma --load ppo_fitness,mc_fitness
python plot_frequency_policy.py --save freq_ppo_mc --load ppo_policy,mc_policy


python plot_cumulative_fitness.py --save cum_fitness_td3 --load td3_fitness,mc_fitness
python plot_frequency_policy.py --save freq_td3_mc --load td3_policy,mc_policy


