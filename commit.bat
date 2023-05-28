@echo off

python -m unittest discover -s test
if %errorlevel%==0 (
 jupytext interbank.py --to notebook
 git add .
 git commit -a
 git push
)