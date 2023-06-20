python interbank.py --eta 1 --save interbank_eta1
python interbank.py --eta 0 --save interbank_eta0

python run_ppo1.py --train models\ppo1  --verbose
python run_ppo1.py --load ppo1 --verbose --save ppo1

python run_mc1.py --save mc1

python plot.py --column 2 --load ppo1 --save ppo1t
#  0 POLICY
#  1 FITNESS
#  2 LIQUIDITY
#  3 IR
#  4 BANKRUPTCY
#  5 BEST_LENDER
#  6 BEST_LENDER_CLIENTS
#  7 CREDIT_CHANNELS