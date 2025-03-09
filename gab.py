import random

from pymysql import connect

N = 5
RANDOM_CONNECTIVITY = 1

marketBank = []
vicino = []
connected = []
for i in range(N):
    marketBank.append([])
    vicino.append(None)
    for j in range(N):
        marketBank[i].append(None)
        connected.append(0)


#for i in range(N):
#    print(marketBank[i])


for i in range(N):
    if connected[i] != 1:
        if random.random() < RANDOM_CONNECTIVITY:
            j = random.randint(0, N-1)
            if i != j:
                marketBank[i][j] = 1
                vicino[i] = j
                connected[i] = 1
            else:
                while i==j:
                    j = random.randint(0, N-1)
                marketBank[i][j] = 1
                vicino[i] = j
                connected[i] = 1
print("----------------")
for i in range(N):
    print(marketBank[i])
