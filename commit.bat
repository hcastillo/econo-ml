@echo off

jupytext model.py --to notebook
git add .
git commit -a
git push
