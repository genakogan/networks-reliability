import networkx as nx
import matplotlib.pyplot as plt
import random
from disjoint_set import DisjointSet
import itertools
import numpy as np
import math
import pickle

global G, r, Fr, T1, T2, T3, M1, M2, M3, table, table1, table2, table3, table4, ds, p_values, Sp, randomParmutation, edgeNumbers, edgeDrops

T1 = 4
T2 = 10
T3 = 17

M1 = 1000
M2 = 10000
M3 = 50000

p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

G = nx.Graph()

ds = DisjointSet()
edgeDrops = {}
table = {}
table1 = {}
table2 = {}
table3 = {}
table4 = {}
r = 0
edgeNumbers = []
for i in range(1, 31):
    edgeNumbers.append(i)
    edgeDrops[i] = 0


# ---- Functions --------- !

def relativeError(values):
    avrg = 0
    S = 0
    sum = 0

    for value in values:
        avrg += value
    avrg = avrg / len(values)

    for value in values:
        power = math.pow((value - avrg), 2)
        sum += power

    S = math.sqrt(sum / (len(values) - 1))

    return S / avrg


def calcFsMultiplication(f_table, tableX, index):
    sum = 0
    calcFi = 0
    for k, v in tableX.items():  # K is row number
        for value in range(1, k):
            calcFi += f_table[value]
        sum += calcFi * tableX[k][index]
        calcFi = 0
    return sum


def makeDSS():
    for i in range(1, 21):
        ds.find(i)


def booleanValue(p, u, v):
    value = random.uniform(0, 1)
    if value <= p:
        ds.union(u, v)
        return "g"
    return "r"


def calculateDSS():
    if ds.connected(T1, T2) and ds.connected(T2, T3):
        return True


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


def initTable1():
    p = 1
    for i in range(1, 31):
        table1[i] = [i, -1, -1]


def initTable():
    p = 1
    for i in range(1, 31):
        table[i] = [i, -1, -1, -1]


def initTable2():
    p = 0.01
    for i in range(99):
        table2[p] = [p, -1, -1, -1]
        p += 0.01
        p = "{:.2f}".format(p)
        p = float(p)


def initTable3():
    p = 0.01
    for i in range(99):
        table3[p] = [p, -1, -1, -1]
        p += 0.01
        p = "{:.2f}".format(p)
        p = float(p)


def initTable4():
    p = 0.01
    for i in range(1, 11):
        table4[i] = [i, -1, -1]
    table4["r.e"] = ["r.e=", -1, -1]

from tabulate import tabulate
def printTable1():
    res={'Index':[],'M1':[],'M2':[]}
    for i in table1:
        res['Index'].append(table1[i][0])
        res['M1'].append(table1[i][1])
        res['M2'].append(table1[i][2])
    print(tabulate(res, headers="keys"))



def printTable2():
    res = { '9M1000': [], '9M10000': [],'16M1000': [], '16M10000': [],'15M1000': [], '15M10000': [],'21M1000': [], '21M10000': []}
    for edge in table2.items():
        for M in edge[1].items():
            for i in M[1]:
                res[str(edge[0])+'M'+str(M[0])].append(M[1][i])
    print(tabulate(res, headers="keys"))


def printTable3(p_values):
    res={'p_values':[]}
    for p in table3.items():
        res['p_values'].append(p[0])
        for value in p[1]:
            if value in res:
                res[value].append(p[1][value])
            else:
                res[value]=[p[1][value]]
    print(tabulate(res, headers="keys"))




def printTable4():
    print("Gain in Reliability by means of CMC")
    p_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    edges = [9, 16, 15, 21]
    print("",end='\t')
    for edge in edges:
        print(edge, end='\t')
    print()
    for p in p_values:
        print(p,end='\t')
        for edge in edges:
         print(round(table4[edge][p],3),end='\t')
        print()
    print("---------------------END TABLE 4---------------------")

def printGraph(G):

    def getNode(u):
        if u==4 or u == 10 or u==17:
           return "green"
        return '#7EC0EE'

    plt.subplots(1, 1, figsize=(14, 9))
    pos = nx.get_node_attributes(G, name='pos')
    labels = nx.get_edge_attributes(G, 'weight')
    edges = G.edges()
    colorsE = [G[u][v]['color'] for u, v in edges]
    colorsN = [getNode(u) for u in G.nodes()]
    edges = list(edges)
    nx.draw(G, pos=pos, arrows=False, edge_color=colorsE, node_color=colorsN, with_labels=True, width=1.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    #plt.get_current_fig_manager().window.wm_geometry("+300+25")
    plt.show()
def printAll():
    print("T1=" + str(T1) + " T2=" + str(T2) + " T3=" + str(T3))
    print("-----------------------------------")
    print("-----------------------------------")
    print("Table 1:Cumulative Destruction Spectra:")
    print("M1=1000   M2=10,000")
    printTable1()
    print("-----------------------------------")
    print("-----------------------------------")
    print("-----------------------------------")

    print("Table 2: Cumulative BIM Spectra")
    printTable2()
    print("-----------------------------------")
    print("-----------------------------------")
    print("-----------------------------------")

    print("Table 3:")
    print("  i   M2=10,000 ")
    printTable3(p_values)
    print("-----------------------------------")
    print("-----------------------------------")
    print("-----------------------------------")
    print("Table 4:")
    printTable4()
    printGraph(G)


def calcFs(p_values):
    f_table = {}
    p_table = {}
    for q in p_values:
        p = 1 - q
        q = float(q)
        p = float(p)

        p = round(p, 2)
        for n in range(1, 31):
            for k in range(1, 30 - n + 2):
                ncrr = int(nCr(30, k))

                q_pow_k = math.pow(q, k)

                p_n_minus_k = pow(p, (30 - k))

                resultTemp = ncrr * q_pow_k * p_n_minus_k

            f_table[n] = resultTemp

        p_table[q] = f_table
        f_table = {}
    return p_table


def destructionSpectra():
    global G, randomParmutation, ds, r, edgeDrops, b, a

    tempDict = {}
    makeGraph(1)
    for k, v in G.edges():
        vertex = G[k][v]
        tempDict[vertex['weight']] = [k, v]
    # G[][]['color'] = 'r'
    for edge in randomParmutation:
        vertex = tempDict[edge]
        edge = G[vertex[0]][vertex[1]]
        edge['color'] = 'r'

        ds = DisjointSet()
        makeDSS()

        for u, v in G.edges():
            if G[u][v]['color'] == 'g':
                ds.union(u, v)
        result = calculateDSS()
        if result == True:
            r += 1
        else:
            edgeDrops[r] += 1
            break


def destructionBimSpectra():
    global G, randomParmutation, ds, r, edgeDrops, b, a

    tempDict = {}
    makeGraph(1)
    for k, v in G.edges():
        vertex = G[k][v]
        tempDict[vertex['weight']] = [k, v]
    # G[][]['color'] = 'r'
    for edge in randomParmutation:
        vertex = tempDict[edge]
        edge = G[vertex[0]][vertex[1]]
        edge['color'] = 'r'

        ds = DisjointSet()
        makeDSS()

        for u, v in G.edges():
            if G[u][v]['color'] == 'g':
                ds.union(u, v)
        result = calculateDSS()
        if result == True:
            r += 1
        else:
            a[r+1] += 1
            while r < 30:
                for i in range(0, r+1):
                    b[r + 1][randomParmutation[i]] += 1
                r += 1

            break


def facto(n):
    return math.factorial(n)


def calcBim(j, p,Z, Y, n=30):
    res = 0
    for i in range(1, n+1):
        z = Z[i][j]
        y = Y[i]
        pq = math.pow(1-p, (i - 1)) * math.pow(p, (n - i))
        pq2 = math.pow(1-p, i) * math.pow(p, (n - i - 1))
        divENbas = facto(i) * facto(n - i)
        natz=facto(n)
        droit=(y - z) * pq2
        gauche=(z * pq)
        bim = (gauche-droit)*natz
        bim/=divENbas
        res += bim
    return abs(res)




def makeGraph(p):
    makeDSS()
    G.add_node(1, pos=(6.123233995736766e-17, -1.0))
    G.add_node(2, pos=(-0.9510565162951535, -0.3090169943749475))
    G.add_node(3, pos=(-0.5877852522924732, 0.8090169943749473))
    G.add_node(4, pos=(0.5877852522924731, 0.8090169943749475))
    G.add_node(5, pos=(0.9510565162951535, -0.3090169943749474))
    G.add_node(6, pos=(2.367, -0.75))
    G.add_node(7, pos=(1.7633557568774194, -2.4270509831248424))
    G.add_node(8, pos=(1.2246467991473532e-16, -2.5))
    G.add_node(9, pos=(-1.7633557568774192, -2.4270509831248424))
    G.add_node(10, pos=(-2.3, -0.78))
    G.add_node(11, pos=(-2.853169548885461, 0.9270509831248419))
    G.add_node(12, pos=(-1.461, 2.04))
    G.add_node(13, pos=(-5.51091059616309e-16, 3.0))
    G.add_node(14, pos=(1.4, 1.99))
    G.add_node(15, pos=(2.8531695488854605, 0.9270509831248421))
    G.add_node(16, pos=(4.755282581475767, 1.545084971874737))
    G.add_node(17, pos=(2.938926261462366, -4.045084971874737))
    G.add_node(18, pos=(-2.938926261462365, -4.045084971874737))
    G.add_node(19, pos=(-4.755282581475768, 1.5450849718747364))
    G.add_node(20, pos=(-9.184850993605148e-16, 5.0))

    G.add_edge(1, 2, color=booleanValue(p, 1, 2), weight=1)
    G.add_edge(2, 3, color=booleanValue(p, 2, 3), weight=2)
    G.add_edge(3, 4, color=booleanValue(p, 3, 4), weight=3)
    G.add_edge(4, 5, color=booleanValue(p, 4, 5), weight=4)
    G.add_edge(5, 1, color=booleanValue(p, 5, 1), weight=5)
    G.add_edge(6, 7, color=booleanValue(p, 6, 7), weight=6)
    G.add_edge(7, 8, color=booleanValue(p, 7, 8), weight=7)
    G.add_edge(8, 9, color=booleanValue(p, 8, 9), weight=8)
    G.add_edge(9, 10, color=booleanValue(p, 9, 10), weight=9)
    G.add_edge(10, 11, color=booleanValue(p, 10, 11), weight=10)
    G.add_edge(11, 12, color=booleanValue(p, 11, 12), weight=11)
    G.add_edge(12, 13, color=booleanValue(p, 12, 13), weight=12)
    G.add_edge(13, 14, color=booleanValue(p, 13, 14), weight=13)
    G.add_edge(14, 15, color=booleanValue(p, 14, 15), weight=14)
    G.add_edge(15, 6, color=booleanValue(p, 15, 6), weight=15)
    G.add_edge(16, 17, color=booleanValue(p, 16, 17), weight=16)
    G.add_edge(17, 18, color=booleanValue(p, 17, 18), weight=17)
    G.add_edge(18, 19, color=booleanValue(p, 18, 19), weight=18)
    G.add_edge(19, 20, color=booleanValue(p, 19, 20), weight=19)
    G.add_edge(20, 16, color=booleanValue(p, 20, 16), weight=20)
    G.add_edge(15, 16, color=booleanValue(p, 15, 16), weight=21)
    G.add_edge(7, 17, color=booleanValue(p, 7, 17), weight=22)
    G.add_edge(9, 18, color=booleanValue(p, 9, 18), weight=23)
    G.add_edge(11, 19, color=booleanValue(p, 11, 19), weight=24)
    G.add_edge(20, 13, color=booleanValue(p, 20, 13), weight=25)
    G.add_edge(1, 8, color=booleanValue(p, 1, 8), weight=26)
    G.add_edge(2, 10, color=booleanValue(p, 2, 10), weight=27)
    G.add_edge(3, 12, color=booleanValue(p, 3, 12), weight=28)
    G.add_edge(4, 14, color=booleanValue(p, 4, 14), weight=29)
    G.add_edge(5, 6, color=booleanValue(p, 5, 6), weight=30)

def returnBoolToInt(on):
    if on:
        return 1
    return 0
def makeGraphON(p,egde,on):
    makeDSS()
    G.add_node(1, pos=(6.123233995736766e-17, -1.0))
    G.add_node(2, pos=(-0.9510565162951535, -0.3090169943749475))
    G.add_node(3, pos=(-0.5877852522924732, 0.8090169943749473))
    G.add_node(4, pos=(0.5877852522924731, 0.8090169943749475))
    G.add_node(5, pos=(0.9510565162951535, -0.3090169943749474))
    G.add_node(6, pos=(2.367, -0.75))
    G.add_node(7, pos=(1.7633557568774194, -2.4270509831248424))
    G.add_node(8, pos=(1.2246467991473532e-16, -2.5))
    G.add_node(9, pos=(-1.7633557568774192, -2.4270509831248424))
    G.add_node(10, pos=(-2.3, -0.78))
    G.add_node(11, pos=(-2.853169548885461, 0.9270509831248419))
    G.add_node(12, pos=(-1.461, 2.04))
    G.add_node(13, pos=(-5.51091059616309e-16, 3.0))
    G.add_node(14, pos=(1.4, 1.99))
    G.add_node(15, pos=(2.8531695488854605, 0.9270509831248421))
    G.add_node(16, pos=(4.755282581475767, 1.545084971874737))
    G.add_node(17, pos=(2.938926261462366, -4.045084971874737))
    G.add_node(18, pos=(-2.938926261462365, -4.045084971874737))
    G.add_node(19, pos=(-4.755282581475768, 1.5450849718747364))
    G.add_node(20, pos=(-9.184850993605148e-16, 5.0))

    G.add_edge(1, 2, color=booleanValue(p, 1, 2), weight=1)
    G.add_edge(2, 3, color=booleanValue(p, 2, 3), weight=2)
    G.add_edge(3, 4, color=booleanValue(p, 3, 4), weight=3)
    G.add_edge(4, 5, color=booleanValue(p, 4, 5), weight=4)
    G.add_edge(5, 1, color=booleanValue(p, 5, 1), weight=5)
    G.add_edge(6, 7, color=booleanValue(p, 6, 7), weight=6)
    G.add_edge(7, 8, color=booleanValue(p, 7, 8), weight=7)
    G.add_edge(8, 9, color=booleanValue(p, 8, 9), weight=8)
    if egde==9:
        G.add_edge(9, 10, color=booleanValue(returnBoolToInt(on),9, 10), weight=9)
    else:
        G.add_edge(9, 10, color=booleanValue(p, 9, 10), weight=9)
    G.add_edge(10, 11, color=booleanValue(p, 10, 11), weight=10)
    G.add_edge(11, 12, color=booleanValue(p, 11, 12), weight=11)
    G.add_edge(12, 13, color=booleanValue(p, 12, 13), weight=12)
    G.add_edge(13, 14, color=booleanValue(p, 13, 14), weight=13)
    G.add_edge(14, 15, color=booleanValue(p, 14, 15), weight=14)
    if egde == 15:
        G.add_edge(15, 6, color=booleanValue(returnBoolToInt(on), 15, 6), weight=15)
    else:
        G.add_edge(15, 6, color=booleanValue(p, 15, 6), weight=15)

    if egde==16:
        G.add_edge(16, 17, color=booleanValue(returnBoolToInt(on), 16, 17), weight=16)
    else:
        G.add_edge(16, 17, color=booleanValue(p, 16, 17), weight=16)

    G.add_edge(17, 18, color=booleanValue(p, 17, 18), weight=17)
    G.add_edge(18, 19, color=booleanValue(p, 18, 19), weight=18)
    G.add_edge(19, 20, color=booleanValue(p, 19, 20), weight=19)
    G.add_edge(20, 16, color=booleanValue(p, 20, 16), weight=20)
    if egde == 21:
        G.add_edge(15, 16, color=booleanValue(returnBoolToInt(on), 15, 16), weight=21)
    else:
        G.add_edge(15, 16, color=booleanValue(p, 15, 16), weight=21)


    G.add_edge(7, 17, color=booleanValue(p, 7, 17), weight=22)
    G.add_edge(9, 18, color=booleanValue(p, 9, 18), weight=23)
    G.add_edge(11, 19, color=booleanValue(p, 11, 19), weight=24)
    G.add_edge(20, 13, color=booleanValue(p, 20, 13), weight=25)
    G.add_edge(1, 8, color=booleanValue(p, 1, 8), weight=26)
    G.add_edge(2, 10, color=booleanValue(p, 2, 10), weight=27)
    G.add_edge(3, 12, color=booleanValue(p, 3, 12), weight=28)
    G.add_edge(4, 14, color=booleanValue(p, 4, 14), weight=29)
    G.add_edge(5, 6, color=booleanValue(p, 5, 6), weight=30)

'''
# Example running ----------------- #
randomParmutation = list(np.random.permutation(edgeNumbers))
initTable1()
initTable2()
initTable3()
initTable4()
destructionSpectra()
makeGraph(1)
plt.subplots(1, 1, figsize=(14, 9))
pos = nx.get_node_attributes(G, name='pos')
labels = nx.get_edge_attributes(G, 'weight')
edges = G.edges()
colors = [G[u][v]['color'] for u, v in edges]
edges = list(edges)
nx.draw(G, pos=pos, arrows=False, edge_color=colors, node_color='#7EC0EE', with_labels=True, width=1.5)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#plt.get_current_fig_manager().window.wm_geometry("+300+25")
plt.show()

'''
'''
# part 1 ----------------------------------- !
initTable1()
edgeDrops = {}
for i in range(1, 31):
    edgeDrops[i] = 0

for i in range(M1):
    randomParmutation = list(np.random.permutation(edgeNumbers))
    r = 0
    destructionSpectra()
tempEdgeDrops={}
for k,v in edgeDrops.items():
    if k==1:
        tempEdgeDrops[k]=v
    else:
        tempEdgeDrops[k] = tempEdgeDrops[k-1]+v
for k,v in edgeDrops.items():
     table1[k][1] = tempEdgeDrops[k] / M1

edgeDrops={}
for i in range(1,31):
    edgeDrops[i] = 0

for i in range(M2):
    randomParmutation = list(np.random.permutation(edgeNumbers))
    r = 0
    destructionSpectra()
tempEdgeDrops={}
for k,v in edgeDrops.items():
    if k==1:
        tempEdgeDrops[k]=v
    else:
        tempEdgeDrops[k] = tempEdgeDrops[k-1]+v
for k,v in edgeDrops.items():
     table1[k][2] = tempEdgeDrops[k] / M2




with open('table1.pickle', 'wb') as handle:
    pickle.dump(table1, handle, protocol=pickle.HIGHEST_PROTOCOL)

#part 1 output --- done ---- !
#printTable1()
'''
# part 2-3-----------------------------------#
'''
table2 = {}

# Iteration M1

edgeDrops = {}
a = {}
b = {}
for i in range(1, 31):
    edgeDrops[i] = 0
for i in range(1, 31):
    a[i] = 0
    temp = {}
    for j in range(1, 31):
        temp[j] = 0
    b[i] = temp

for i in range(M1):
    randomParmutation = list(np.random.permutation(edgeNumbers))
    r = 0
    destructionBimSpectra()


z = {}
y = {}
for i in range(1, 31):
    z[i] = {}
    y[i] = a[i]/ M1
    for j in range(1, 31):
        z[i][j] = b[i][j]/ M1


mostImportant = [9, 16]
lessImportant = [15, 21]

for edge in mostImportant:
    table2[edge] = {M1: {}, M2: {}}
    for i in range(1, 31):
        table2[edge][M1][i] = z[i][edge]
for edge in lessImportant:
    table2[edge] = {M1: {}, M2: {}}
    for i in range(1, 31):
        table2[edge][M1][i] = z[i][edge]

a = []
for i in range(1, 31):
    l = []
    for j in range(1, 31):
        l.append(z[i][j])
    a.append(l)

npm = np.array([i for i in a])

# Iteration M2
edgeDrops = {}
a = {}
b = {}
for i in range(1, 31):
    edgeDrops[i] = 0
for i in range(1, 31):
    a[i] = 0
    temp = {}
    for j in range(1, 31):
        temp[j] = 0
    b[i] = temp
for i in range(M2):
    randomParmutation = list(np.random.permutation(edgeNumbers))
    r = 0
    destructionBimSpectra()

z = {}
y = {}
for i in range(1, 31):
    z[i] = {}
    y[i] = a[i] / M2
    for j in range(1, 31):
        z[i][j] = b[i][j] / M2

for edge in mostImportant:
    for i in range(1, 31):
        table2[edge][M2][i] = z[i][edge]
for edge in lessImportant:
    for i in range(1, 31):
        table2[edge][M2][i] = z[i][edge]

aa = []
for i in range(1, 31):
    l = []
    for j in range(1, 31):
        l.append(z[i][j])
    aa.append(l)

npma = np.array([i for i in aa])

with open('table2.pickle', 'wb') as handle:
    pickle.dump(table2, handle, protocol=pickle.HIGHEST_PROTOCOL)

pass
'''

# part 4-----------------------------------#

edgeDrops = {}
a = {}
b = {}
for i in range(1, 31):
    edgeDrops[i] = 0
for i in range(1, 31):
    a[i] = 0
    temp = {}
    for j in range(1, 31):
        temp[j] = 0
    b[i] = temp
for i in range(M2):
    randomParmutation = list(np.random.permutation(edgeNumbers))
    r = 0
    destructionBimSpectra()
Z = {}
Y = {}
tempEdgeDrops={}
for k in range(1,31):
    if k==1:
        tempEdgeDrops[k]=a[k]
    else:
        tempEdgeDrops[k] = tempEdgeDrops[k-1]+a[k]

a=tempEdgeDrops
for i in range(1, 31):
    Z[i] = {}
    Y[i] = a[i] / M2
    for j in range(1, 31):
        
        Z[i][j] = b[i][j] / M2

with open('Y.pickle', 'wb') as handle:
    pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Z.pickle', 'wb') as handle:
    pickle.dump(Z, handle, protocol=pickle.HIGHEST_PROTOCOL)

table3 = {}
P = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for p in P:
    table3[p]={
        'BIM9':calcBim(9,p,Z,Y),
        'BIM9Sp':calcBim(9,p,Z,Y)*(1-p),
        'BIM16':calcBim(16,p,Z,Y),
        'BIM16Sp':calcBim(16,p,Z,Y)*(1-p),
        'BIM15':calcBim(15,p,Z,Y),
        'BIM15Sp':calcBim(15,p,Z,Y)*(1-p),
        'BIM21':calcBim(21,p,Z,Y),
        'BIM21Sp':calcBim(21,p,Z,Y)*(1-p)
    }

with open('table3.pickle', 'wb') as handle:
    pickle.dump(table3, handle, protocol=pickle.HIGHEST_PROTOCOL)

a=5

'''
# part 5 ------------------------------- #
p_values = [0.4,0.5,0.6,0.7,0.8,0.9]
edges = [9,16,15,21]
for edge in edges:
    table4[edge]={}
    for p in p_values:
        for i in range(M2):
            ds = DisjointSet()
            makeGraphON(p,edge,True)
            result = calculateDSS()
            if (result == True):
                r += 1
        R1 = r / M2
        r=0
        for i in range(M2):
            ds = DisjointSet()
            makeGraphON(p,edge,False)
            result = calculateDSS()
            if (result == True):
                r += 1
        R2 = r / M2
        r=0
        table4[edge][p]=abs(R2-R1)


with open('table4.pickle', 'wb') as handle:
    pickle.dump(table4, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''








with open('table1.pickle', 'rb') as handle:
    table1 = pickle.load(handle)

with open('table2.pickle', 'rb') as handle:
    table2 = pickle.load(handle)

with open('table3.pickle', 'rb') as handle:
    table3 = pickle.load(handle)

with open('table4.pickle', 'rb') as handle:
    table4 = pickle.load(handle)

with open('Graph.pickle', 'rb') as handle:
    G = pickle.load(handle)


printAll()
