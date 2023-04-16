@echo off

python -m unittest discover
if %errorlevel%==0 (
 jupytext bank_net.py --to notebook
 git add .
 git commit -a
 git push
)