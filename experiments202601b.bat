
mkdir \experiments\0


python -m experiments.exp_0psi0 --do --errorbar --remove_nans
python -m experiments.exp_0psi05 --do --errorbar --remove_nans
python -m experiments.exp_0psi025 --do --errorbar --remove_nans
python -m experiments.exp_0psi075 --do --errorbar --remove_nans
python -m experiments.exp_0psi098 --do --errorbar --remove_nans
python -m experiments.exp_0psi1 --do --errorbar --remove_nans
python -m experiments.exp_0psivar --do --errorbar --remove_nans
python -m experiments.exp_0endog --do --errorbar --remove_nans
python -m experiments.exp_0surviving_4 --do 
python -m experiments.exp_0surviving_4b --do

python -m utils.plot_p --input "psi0:psi=0.00 psi025:psi=0.25 psi05:psi=0.50 psi075:psi=0.75 psi1:psi=1.00 psiendog:endogenous" --working_dir /experiments/0 --p_values "0.0001 0.001 0.01 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.4 0.6 0.8 1"

python -m utils.plot_pb --input "psi0:psi=0.00 psi025:psi=0.25 psi05:psi=0.50 psi075:psi=0.75 psi1:psi=1.00 psiendog:endogenous" --working_dir /experiments/0 --p_values "0.0001 0.001 0.01 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.4 0.6 0.8 1"



