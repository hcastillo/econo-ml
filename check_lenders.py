# -*- coding: utf-8 -*-
"""
Trying to determine  if the randomness of new_lender() is correct or not

@author: hector@bith.net
@date:   03/2024
"""
import interbank
import matplotlib.pyplot as plt
import random

model = interbank.Model()
model.config.N = 200
model.initialize()

NUMBER_OF_EXECUTIONS=len(model.banks)*1000

elements = [0 for i in range(len(model.banks))]
randoms  = [0 for i in range(len(model.banks))]
for i in model.banks:
    for j in range(NUMBER_OF_EXECUTIONS):
        new_lender = i.new_lender()
        elements[new_lender]+=1
        randoms[random.randint(0,len(model.banks)-1)]+=1
        i.lender = new_lender


plt.clf()
plt.plot(elements, 'r-', label="new_lender")
plt.plot(randoms, 'b-', label="random")
plt.xlabel("Firm id")
plt.title(f"Random apparitions over #{NUMBER_OF_EXECUTIONS}")
plt.ylabel("#")
plt.legend(loc="upper left")
plt.savefig("lenders\\test_lenders.svg")
