@echo off

rem python interbank.py --save boltzman --plot_format png 
rem python interbank.py --lc initialstability --save inistab --plot_format png 
rem python interbank.py --lc shockedmarket --lc_p 0.001  --save shocked --plot_format png # --t 10
rem python interbank.py --lc shockedmarket --lc_p 0.002  --save shocked2 --plot_format png  # --t 10

python -m unittest discover -s tests
if %errorlevel%==0 (
 powershell -command "(Get-Content interbank.py) -replace 'lc\.','' -replace 'import interbank_lenderchange as lc','' | Out-File -encoding Default inter.py"
 jupytext inter.py -o inter.ipynb
 jupytext interbank_lenderchange.py -o lc.ipynb
 nbmerge lc.ipynb inter.ipynb -o interbank.ipynb
 del inter.ipynb lc.ipynb inter.py
 git add .
 git commit -a
 git push
)