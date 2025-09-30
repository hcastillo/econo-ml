#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plots:
 - top frequency of policy recommendation comparing mc.txt to ppo.txt (pag21 paper)
 - cumulative fitness

@author: hector@bith.net
@date:   06/2023, 09/2025
"""

import pandas as pd
import lxml.etree
import sys
import matplotlib.pyplot as plt
import numpy as np



eje_x = ['equity','bankruptcies','interest_rate' ]
titles_x = ['$E$','bankruptcy','$i_r$']
eje_y = ['1_psi0', '1_psi025', '1_psi05', '1_psi075', '1_psi1', '3_psiendog']
titles_y  = ['psi=0','psi=0.25','psi=0.5','psi=0.75','psi=0.99','psi endogenous' ]
resultado = 'psi_p.png'
titulo = 'comparing psi/p'

# valores para p
x  = list(np.linspace(0.0001, 0.2, num=5))

input_dir = 'c:\experiments'


# load data:
def __transform_line_from_string(line_with_values):
    items = []
    for i in line_with_values.replace('  ', ' ').strip().split(' '):
        try:
            items.append(int(i))
        except ValueError:
            items.append(float(i))
    return items
def read_gdt(filename):
    tree = lxml.etree.parse(filename)
    root = tree.getroot()
    children = root.getchildren()
    values = []
    columns = []
    if len(children) == 3:
        for variable in children[1].getchildren():
            column_name = variable.values()[0].strip()
            if column_name == 'leverage_':
                column_name = 'leverage'
            columns.append(column_name)
        for value in children[2].getchildren():
            values.append(__transform_line_from_string(value.text))
    if columns and values:
        return pd.DataFrame(columns=columns, data=values)
    else:
        return pd.DataFrame()



rows, cols = len(eje_y), len(eje_x)
plt.title(titulo)
fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
for i, itemi in enumerate(eje_y):
    # load data:
    data = read_gdt(input_dir + '\\' + itemi + '\\results.gdt')
    for j, itemj in enumerate(eje_x):

        y = data[itemj]
        yerr = data['std_'+itemj]/2

        # y = np.random.normal(10, 0.5, size=len(x)) + i + 0.3 * j
        # yerr = np.random.uniform(0.05, 0.15, size=len(x))

        # Plot line + error bars
        axes[i, j].errorbar(x, y, yerr=yerr, fmt='o-', color='black', ecolor='black',
                            capsize=3, elinewidth=1, markerfacecolor='none', markeredgecolor='black')

        # Remove grid and background
        axes[i, j].grid(False)
        axes[i, j].set_facecolor('white')
        # axes[i, j].set_title( 'a'+ ' '+ )

        # Optionally set axis labels
        if i == rows - 1:
            axes[i, j].set_xlabel(titles_x[j] )
        if j == 0:
            axes[i, j].set_ylabel(titles_y[i])


plt.tight_layout()
plt.savefig(resultado)
print(resultado)

sys.exit(0)
