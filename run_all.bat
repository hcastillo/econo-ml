python interbank.py --log DEBUG --logfile interbank.log --eta 1 
python interbank.py --eta 1 --save interbank_eta1
python interbank.py --eta 0 --save interbank_eta0

python run_ppo.py --train models\ppo  --verbose
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
