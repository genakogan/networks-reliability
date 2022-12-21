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

    def printAllPart3(self):
        print("T1=" + str(self.T1) + " T2=" + str(self.T2) + " T3=" + str(self.T3))
        print("-----------------------------------")
        print("Random Permutation:")
        print(self.randomParmutation)

        print("-----------------------------------")
        print("Table 1: Destruction Spectra:")
        print("  i    M1=1000   M2=10,000")
        self.printTable1Part3()