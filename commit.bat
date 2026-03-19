@echo off

rem python interbank.py --save boltzman --plot_format png 
rem python interbank.py --lc initialstability --save inistab --plot_format png 
rem python interbank.py --lc shockedmarket --lc_p 0.001  --save shocked --plot_format png # --t 10
rem python interbank.py --lc shockedmarket --lc_p 0.002  --save shocked2 --plot_format png  # --t 10

python -m unittest discover -s tests
if %errorlevel%==0 (
 powershell -command "(Get-Content interbank.py)   -replace 'import interbank_lenderchange','' -replace 'interbank_lenderchange\.',''  -replace 'lc\.','' -replace 'import interbank_lenderchange as lc','' | Out-File -encoding Default inter.py"
 C:\ProgramData\Anaconda3\Scripts\jupytext.exe inter.py -o inter.ipynb
 C:\ProgramData\Anaconda3\Scripts\jupytext.exe interbank_lenderchange.py -o lc.ipynb
 C:\ProgramData\Anaconda3\Scripts\nbmerge.exe lc.ipynb inter.ipynb --out colab_interbank.ipynb
 del inter.ipynb lc.ipynb inter.py
 rem powershell -command "(Get-Content readme.md) -replace 'algorithm.png','doc/algorithm.png' | Out-File -encoding Default README.md"
 cd doc
 pdflatex algorithm.tex
 pdflatex algorithm_boltzmann.tex
 del alg-000001.png
 del alg1-000001.png
 pdftopng -r 300 algorithm.pdf alg
 pdftopng -r 300 algorithm_boltzmann.pdf alg1
 cd ..
 pandoc doc\README.tex -t markdown+pipe_tables-simple_tables-multiline_tables -o README.md
 git add .
 git commit -a
 git push
)

