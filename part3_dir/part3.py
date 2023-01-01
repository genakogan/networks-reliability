import networkx as nx
import matplotlib.pyplot as plt
import random
from disjoint_set import DisjointSet
import numpy as np
import math
import pickle

from tabulate import tabulate



from part2_dir.part2 import Part2

from part1_dir.part1 import Part1

class Part3(Part2):
    def __init__(self,T1=None,T2=None,T3=None):

        Part2.__init__(self,T1=None,T2=None,T3=None)
        self.T1=T1
        self.T2=T2
        self.T3=T3
        self.edgeDrops = {}
        self.edgeNumbers  = []
        self.randomParmutation =[]
        self.a = {}
        self.b = {}
        self.r=0
        self.table2Part3_2 = {}

        self.table3 = {}

    def part3_1(self):
       
        self.initTable1Part3()
        self.edgeDrops = {}
        self.edgeNumbers = []
       
        for i in range(1, 31):
            self.edgeNumbers.append(i)
            self.edgeDrops[i] = 0

        for i in range(self.M1):
            self.randomParmutation = list(np.random.permutation(self.edgeNumbers))
            self.r = 0
            self.destructionSpectra()
        tempEdgeDrops={}
        for k,v in self.edgeDrops.items():
            if k==1:
                tempEdgeDrops[k]=v
            else:
                tempEdgeDrops[k] = tempEdgeDrops[k-1]+v
        for k,v in self.edgeDrops.items():
            self.table1[k][1] = tempEdgeDrops[k] / self.M1

        self.randomParmutation =[]
        self.edgeDrops = {}
        self.edgeNumbers = []
        for i in range(1,31):
            self.edgeNumbers.append(i)
            self.edgeDrops[i] = 0

        for i in range(self.M2):
            self.randomParmutation = list(np.random.permutation(self.edgeNumbers))
            self.r = 0
            self.destructionSpectra()
        tempEdgeDrops={}

        for k,v in self.edgeDrops.items():
            if k==1:
                tempEdgeDrops[k]=v
            else:
                tempEdgeDrops[k] = tempEdgeDrops[k-1]+v
       
        for k,v in self.edgeDrops.items():
            self.table1[k][2] = tempEdgeDrops[k] / self.M2
        

        with open('table1.pickle', 'wb') as handle:
            pickle.dump(self.table1, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('table1.pickle', 'rb') as handle:
            self.table1 = pickle.load(handle)


    def destructionBimSpectra(self):
        
        tempDict = {}
        self.makeGraph(1)
        for k, v in self.G.edges():
            vertex = self.G[k][v]
            tempDict[vertex['weight']] = [k, v]
        # G[][]['color'] = 'r'
        for edge in self.randomParmutation:
            vertex = tempDict[edge]
            edge = self.G[vertex[0]][vertex[1]]
            edge['color'] = 'r'

            self.ds = DisjointSet()
            for i in range(1, 21):
                self.ds.find(i)

            for u, v in self.G.edges():
                if self.G[u][v]['color'] == 'g':
                    self.ds.union(u, v)
            result = self.calculateDSS()
            if result == True:
                self.r += 1
            else:
                self.a[self.r+1] += 1
                
                while self.r < 30:
                    for i in range(0, self.r+1):
                        self.b[self.r + 1][self.randomParmutation[i]] += 1
                    self.r += 1

                break
# ------------------------------------------------------------------------------------
    def part3_2(self):
        
        self.table2Part3_2 = {}

        # Iteration M1

        edgeDrops = {}
        self.a = {}
        self.b = {}
        for i in range(1, 31):
            edgeDrops[i] = 0
        for i in range(1, 31):
            self.a[i] = 0
            temp = {}
            for j in range(1, 31):
                temp[j] = 0
            self.b[i] = temp

        for i in range(self.M1):
            self.randomParmutation = list(np.random.permutation(self.edgeNumbers))
            self.r = 0
            self.destructionBimSpectra()


        z = {}
        y = {}
        for i in range(1, 31):
            z[i] = {}
            y[i] = self.a[i]/ self.M1
            for j in range(1, 31):
                z[i][j] = self.b[i][j]/ self.M1


        mostImportant = [9, 16]
        lessImportant = [15, 21]

        for edge in mostImportant:
            self.table2Part3_2[edge] = {self.M1: {}, self.M2: {}}
            for i in range(1, 31):
                self.table2Part3_2[edge][self.M1][i] = z[i][edge]
        for edge in lessImportant:
            self.table2Part3_2[edge] = {self.M1: {}, self.M2: {}}
            for i in range(1, 31):
                self.table2Part3_2[edge][self.M1][i] = z[i][edge]

        a = []
        for i in range(1, 31):
            l = []
            for j in range(1, 31):
                l.append(z[i][j])
            a.append(l)

        npm = np.array([i for i in a])

        # Iteration M2
        edgeDrops = {}
        self.a = {}
        self.b = {}
        for i in range(1, 31):
            edgeDrops[i] = 0
        for i in range(1, 31):
            self.a[i] = 0
            temp = {}
            for j in range(1, 31):
                temp[j] = 0
            self.b[i] = temp
        for i in range(self.M2):
            self.randomParmutation = list(np.random.permutation(self.edgeNumbers))
            self.r = 0
            self.destructionBimSpectra()

        z = {}
        y = {}
        for i in range(1, 31):
            z[i] = {}
            y[i] = self.a[i] / self.M2
            for j in range(1, 31):
                z[i][j] = self.b[i][j] / self.M2

        for edge in mostImportant:
            for i in range(1, 31):
                self.table2Part3_2[edge][self.M2][i] = z[i][edge]
        for edge in lessImportant:
            for i in range(1, 31):
                self.table2Part3_2[edge][self.M2][i] = z[i][edge]

        aa = []
        for i in range(1, 31):
            l = []
            for j in range(1, 31):
                l.append(z[i][j])
            aa.append(l)

        npma = np.array([i for i in aa])

        with open('table2.pickle', 'wb') as handle:
            pickle.dump(self.table2Part3_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('table2.pickle', 'rb') as handle:
            self.table2Part3_2 = pickle.load(handle)

# ------------------------------------------------------------------------------------
    
    def part3_4(self):
       
        def calcBim(j, p,Z, Y, n=30):
            res = 0
            for i in range(1, n+1):
                z = Z[i][j]
                y = Y[i]
                pq = math.pow(1-p, (i - 1)) * math.pow(p, (n - i))
                pq2 = math.pow(1-p, i) * math.pow(p, (n - i - 1))
                divENbas = math.factorial(i) * math.factorial(n - i)
                natz=math.factorial(n)
                droit=(y - z) * pq2
                gauche=(z * pq)
                bim = (gauche-droit)*natz
                bim/=divENbas
                res += bim
            return abs(res)

        edgeDrops = {}
        self.a = {}
        b = {}
        for i in range(1, 31):
            edgeDrops[i] = 0
        for i in range(1, 31):
            self.a[i] = 0
            temp = {}
            for j in range(1, 31):
                temp[j] = 0
            b[i] = temp
        for i in range(self.M2):
            self.randomParmutation = list(np.random.permutation(self.edgeNumbers))
            self.r = 0
            self.destructionBimSpectra()
        Z = {}
        Y = {}
        tempEdgeDrops={}
        for k in range(1,31):
            if k==1:
                tempEdgeDrops[k]=self.a[k]
            else:
                tempEdgeDrops[k] = tempEdgeDrops[k-1]+self.a[k]

        a=tempEdgeDrops
        for i in range(1, 31):
            Z[i] = {}
            Y[i] = self.a[i] / self.M2
            for j in range(1, 31):
                
                Z[i][j] = b[i][j] / self.M2

        with open('Y.pickle', 'wb') as handle:
            pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Z.pickle', 'wb') as handle:
            pickle.dump(Z, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.table3={}
        P = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for p in P:
           
            self.table3[p]={
                 
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
            pickle.dump(self.table3, handle, protocol=pickle.HIGHEST_PROTOCOL)       
        with open('table3.pickle', 'rb') as handle:
            self.table3 = pickle.load(handle)
    def initTable3(self):
        p = 0.01
        for i in range(99):
            self.table3[p] = [p, -1, -1, -1]
            p += 0.01
            p = "{:.2f}".format(p)
            p = float(p)      
      
    def printTable3(self):
        res={'p_values':[]}
        for p in self.table3.items():
            res['p_values'].append(p[0])
            for value in p[1]:
                if value in res:
                    res[value].append(p[1][value])
                else:
                    res[value]=[p[1][value]]
        print(tabulate(res, headers="keys"))
    def printAllPart3(self):
        print("T1=" + str(self.T1) + " T2=" + str(self.T2) + " T3=" + str(self.T3))
        print("-----------------------------------")
        print("Random Permutation:")
        print(self.randomParmutation)

        print("-----------------------------------")
        print("Table 1: Destruction Spectra:")
        print("  i    M1=1000   M2=10,000")
        self.printTable1Part3()

        print("Table 2: Cumulative BIM Spectra")
        self.printTable2Part3()
        print("-----------------------------------")
        print("-----------------------------------")
        print("-----------------------------------")

        print("Table 3:")
        print("  i   M2=10,000 ")
        
        self.printTable3()
        print("-----------------------------------")
        print("-----------------------------------")
        print("-----------------------------------")