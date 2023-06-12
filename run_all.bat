python interbank.py --eta 1 --save interbank_eta1
python interbank.py --eta 0 --save interbank_eta0

python run_ppo.py --train models\ppo  --verbose
python run_ppo.py --load ppo --verbose --save ppo
