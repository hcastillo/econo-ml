# -*- coding: utf-8 -*-
"""
Trying to determine  if the randomness of new_lender() is correct or not

@author: hector@bith.net
@date:   03/2024
"""
import interbank
import matplotlib.pyplot as plt

OUTPUT = "lenders\\"

def check_lenders(array_best_lenders,time):
    max_duration = 0
    final_best_lender = -1
    current_lender = -1
    current_duration = 0
    for i in range(time):
        if current_lender != array_best_lenders[i] or i == time-1:
            if max_duration < current_duration:
                max_duration = current_duration
                final_best_lender = current_lender
            current_lender = array_best_lenders[i]
            current_duration = 0
        else:
            current_duration += 1
    return final_best_lender, max_duration

def avg_lenders(array_best_lenders,time):
    num_changes = 0
    duration = 0
    current_duration = 0
    current_lender = -1
    for i in range(time):
        current_duration += 1
        if current_lender != array_best_lenders[i] or i == time-1:
            num_changes += 1
            current_lender = array_best_lenders[i]
            duration += current_duration
            current_duration = 0
    return duration / num_changes


prob_change_lender = {}
indicators = {}
indicators["Max guru life"] = {}
indicators["Avg guru life"] = {}
indicators["Avg liquidity"] = {}
indicators["Avg probability"] = {}
indicators["Avg Bankruptcy"] = {}
indicators["Inv avg probability"] = {}
for x in range(0, 10):
    gamma = x/10
    model = interbank.Model()
    model.initialize(export_datafile=f'{OUTPUT}gamma{gamma}')
    model.config.gamma = gamma
    model.simulate_full()
    result_iteration = model.finish()
    prob_change_lender[gamma] = result_iteration[interbank.DataColumns.PROB_CHANGE_LENDER]
    indicators["Avg guru life"][gamma] = avg_lenders(result_iteration[interbank.DataColumns.BEST_LENDER],
                                                     model.config.T)
    _, indicators["Max guru life"][gamma] = check_lenders(result_iteration[interbank.DataColumns.BEST_LENDER],
                                                          model.config.T)
    indicators["Avg liquidity"][gamma] = result_iteration[interbank.DataColumns.LIQUIDITY].mean()
    indicators["Avg probability"][gamma] = result_iteration[interbank.DataColumns.PROB_CHANGE_LENDER].mean()
    indicators["Inv avg probability"][gamma] = 1-result_iteration[interbank.DataColumns.PROB_CHANGE_LENDER].mean()
    indicators["Avg Bankruptcy"][gamma] = result_iteration[interbank.DataColumns.BANKRUPTCY].mean()
    gamma += 0.1

plt.clf()
for x in prob_change_lender:
    plt.plot( prob_change_lender[x], color=plt.colormaps['viridis'](x), label=f"$\\gamma={x}$")
plt.legend(loc="lower left")
plt.savefig("lenders\\test_gamma.svg")


for plot in indicators:
    plt.clf()
    x = []
    y = []
    for gamma in indicators[plot]:
        x.append(gamma)
        y.append(indicators[plot][gamma])
    plt.plot(x, y)
    plt.legend(plot)
    plt.savefig('lenders\\test_'+plot.replace(" ","")+'.svg')



