import networkx as nx
import matplotlib.pyplot as plt
import random
from disjoint_set import DisjointSet
import numpy as np
import math
import pickle



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
            print("asddad")
            self.table2Part3_2 = pickle.load(handle)


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