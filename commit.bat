@echo off

python interbank.py --save boltzman
python interbank.py --lc initialstability --save inistab
python interbank.py --lc shockedmarket --lc_p 0.001 --t 10 --save shocked

python -m unittest discover -s test
if %errorlevel%==0 (
 jupytext interbank.py --to notebook
 git add .
 git commit -a
 git push
)