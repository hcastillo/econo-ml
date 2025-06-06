python exp_boltzmann.py --do --clear_results 
python exp_marketpower.py --do --clear_results 
python exp_preferential.py --do --clear_results 
python exp_restrictedmarket.py --do --clear_results 
python exp_shockedmarket.py --do --clear_results 
rem python exp_smallworld.py --do --clear_results 


python exp_surviving.py --do
python exp_surviving.psi.py --do
python exp_surviving.4.py --do
python exp_surviving.psi.05.py --do
python exp_surviving.psi.099.py --do


copy c:\experiments\surviving.psi.05\_failures_rationed_accum.csv c:\experiments\surviving.4\_failures_rationed_accum.psi05.csv 
copy c:\experiments\surviving.psi.05\_failures_rationed_accum.png c:\experiments\surviving.4\_failures_rationed_accum.psi05.png
copy c:\experiments\surviving.psi.1\_failures_rationed_accum.csv c:\experiments\surviving.4\_failures_rationed_accum.psi1.csv 
copy c:\experiments\surviving.psi.1\_failures_rationed_accum.png c:\experiments\surviving.4\_failures_rationed_accum.psi1.png

