python -m experiments.exp_1psi0 --do --errorbar
python -m experiments.exp_1psi05 --do --errorbar
python -m experiments.exp_1psi025 --do --errorbar
python -m experiments.exp_1psi075 --do --errorbar
python -m experiments.exp_1psi098 --do --errorbar
python -m experiments.exp_1psi1 --do --errorbar
python -m experiments.exp_1psivar --do --errorbar
rem python -m exp_surviving.psi --do
rem python -m exp_surviving.4 --do
rem python -m exp_surviving.psi.05 --do
rem python -m exp_surviving.psi.099 --do


rem copy c:\experiments\surviving.psi.05\_failures_rationed_acum.csv c:\experiments\surviving.4\_failures_rationed_acum.psi05.csv 
rem copy c:\experiments\surviving.psi.05\_failures_rationed_acum.png c:\experiments\surviving.4\_failures_rationed_acum.psi05.png
rem copy c:\experiments\surviving.psi.1\_failures_rationed_acum.csv c:\experiments\surviving.4\_failures_rationed_acum.psi1.csv 
rem copy c:\experiments\surviving.psi.1\_failures_rationed_acum.png c:\experiments\surviving.4\_failures_rationed_acum.psi1.png


rem python -m utils.plot_psi_horiz --input "1_psi0:psi=0.00 1_psi025:psi=0.25 1_psi05:psi=0.50 1_psi075:psi=0.75 1_psi1:psi=1.00" --working_dir /experiments/20251015.psi_0a1_raiz_cuadrada
rem python -m utils.plot_psi_horiz --input "1_psi0:psi=0.00 1_psi025:psi=0.25 1_psi05:psi=0.50 1_psi075:psi=0.75 1_psi1:psi=1.00" --working_dir /experiments/20251015.psi_0a1_raiz_cuadrada_0.99
rem python -m utils.plot_psi_horiz --input "1_psi0:psi=0.00 1_psi025:psi=0.25 1_psi05:psi=0.50 1_psi075:psi=0.75 1_psi1:psi=1.00" --working_dir /experiments/20251015.sin_normaliz_y_limitando_psi_a099

rem python -m utils.unify_gdt --input "1_psi0:psi=0.00 1_psi025:psi=0.25 1_psi05:psi=0.50 1_psi075:psi=0.75 1_psi1:psi=1.00" --working_dir /experiments/20251015.psi_0a1_raiz_cuadrada
rem python -m utils.unify_gdt --input "1_psi0:psi=0.00 1_psi025:psi=0.25 1_psi05:psi=0.50 1_psi075:psi=0.75 1_psi1:psi=1.00" --working_dir /experiments/20251015.psi_0a1_raiz_cuadrada_0.99
rem python -m utils.unify_gdt --input "1_psi0:psi=0.00 1_psi025:psi=0.25 1_psi05:psi=0.50 1_psi075:psi=0.75 1_psi1:psi=1.00" --working_dir /experiments/20251015.sin_normaliz_y_limitando_psi_a099



python -m utils.plot_psi_vert --input "1_psi0:psi=0.00 1_psi025:psi=0.25 1_psi05:psi=0.50 1_ si075:psi=0.75 1_psi098:psi=0.98 1_psi1:psi=1.00" --working_dir /experiments/20251021.normaliz_raiz_cuadr_p_mas_valores  --p_values "0.0001 0.05 0.1 0.15 0.2"
python -m utils.plot_psi_horiz --input "1_psi0:psi=0.00 1_psi025:psi=0.25 1_psi05:psi=0.50 1_ si075:psi=0.75 1_psi098:psi=0.98 1_psi1:psi=1.00" --working_dir /experiments/20251021.normaliz_raiz_cuadr_p_mas_valores  --p_values "0.0001 0.05 0.1 0.15 0.2"
python -m utils.plot_p --input "1_psi0:psi=0.00 1_psi025:psi=0.25 1_psi05:psi=0.50 1_psi075:psi=0.75 1_psi098:psi=0.98 1_psi1:psi=1.00" --working_dir /experiments/20251021.normaliz_raiz_cuadr_p_mas_valores --p_values "0.0001 0.05 0.1 0.15 0.2"