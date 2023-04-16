@echo off

python -m unittest discover
if %errorlevel%==0 (
 jupytext model.py --to notebook
 git add .
 git commit -a
 git push
)