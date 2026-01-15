python -m experiments.exp_1psi0 --do --errorbar
python -m experiments.exp_1psi05 --do --errorbar
python -m experiments.exp_1psi025 --do --errorbar
python -m experiments.exp_1psi075 --do --errorbar
python -m experiments.exp_1psi098 --do --errorbar
python -m experiments.exp_1psi1 --do --errorbar
python -m experiments.exp_1psivar --do --errorbar
python -m experiments.exp_1surviving_4 --do
python -m experiments.exp_1surviving_4b --do
python -m experiments.exp_1endog --do --errorbar

python -m utils.plot_p --input "psi0:psi=0.00 psi025:psi=0.25 psi05:psi=0.50 psi075:psi=0.75 psi1:psi=1.00 psiendog:endogenous" --working_dir /experiments/1 --p_values "0.0001 0.001 0.01 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.4 0.6 0.8 1"

python -m utils.plot_p --input "psi0:psi=0.00 psi025:psi=0.25 psi05:psi=0.50 psi075:psi=0.75 psi1:psi=1.00 psiendog:endogenous" --working_dir /experiments/1 --p_values "0.0001 0.001 0.01 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.4 0.6 0.8 1"


python -m experiments.exp_0psi0 --do --errorbar
python -m experiments.exp_0psi05 --do --errorbar
python -m experiments.exp_0psi025 --do --errorbar
python -m experiments.exp_0psi075 --do --errorbar
python -m experiments.exp_0psi098 --do --errorbar
python -m experiments.exp_0psi1 --do --errorbar
python -m experiments.exp_0psivar --do --errorbar
python -m experiments.exp_0surviving_4 --do
python -m experiments.exp_0surviving_4b --do
python -m experiments.exp_0endog --do --errorbar

python -m utils.plot_p --input "psi0:psi=0.00 psi025:psi=0.25 psi05:psi=0.50 psi075:psi=0.75 psi1:psi=1.00 psiendog:endogenous" --working_dir /experiments/0 --p_values "0.0001 0.001 0.01 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.4 0.6 0.8 1"

python -m utils.plot_p --input "psi0:psi=0.00 psi025:psi=0.25 psi05:psi=0.50 psi075:psi=0.75 psi1:psi=1.00 psiendog:endogenous" --working_dir /experiments/0 --p_values "0.0001 0.001 0.01 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.4 0.6 0.8 1"



python -m experiments.exp_2psi0 --do --errorbar
python -m experiments.exp_2psi05 --do --errorbar
python -m experiments.exp_2psi025 --do --errorbar
python -m experiments.exp_2psi075 --do --errorbar
python -m experiments.exp_2psi098 --do --errorbar
python -m experiments.exp_2psi1 --do --errorbar
python -m experiments.exp_2psivar --do --errorbar
python -m experiments.exp_2surviving_4 --do
python -m experiments.exp_2surviving_4b --do
python -m experiments.exp_2endog --do --errorbar

python -m utils.plot_p --input "psi0:psi=0.00 psi025:psi=0.25 psi05:psi=0.50 psi075:psi=0.75 psi1:psi=1.00 psiendog:endogenous" --working_dir /experiments/2 --p_values "0.0001 0.001 0.01 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.4 0.6 0.8 1"

python -m utils.plot_p --input "psi0:psi=0.00 psi025:psi=0.25 psi05:psi=0.50 psi075:psi=0.75 psi1:psi=1.00 psiendog:endogenous" --working_dir /experiments/2 --p_values "0.0001 0.001 0.01 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.4 0.6 0.8 1"