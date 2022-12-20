import networkx as nx
import matplotlib.pyplot as plt
import random
from disjoint_set import DisjointSet
import numpy as np
import math
import pickle



from part2_dir.part2 import Part2
from part2_dir.table import Tables
from part1_dir.part1 import Part1

class Part3(Part2,Part1):
    def __init__(self,T1=None,T2=None,T3=None):

        
        Part1.__init__(self,T1=None,T2=None,T3=None)
        Part2.__init__(self,T1=None,T2=None,T3=None)
        self.T1=T1
        self.T2=T2
        self.T3=T3
        self.edgeDrops = {}
        self.edgeNumbers  = []
        self.randomParmutation =[]

    def part3_1(self):
        
        self.initTable1()
        for i in range (1,30):
            self.edgeNumbers.append(i)
            self.edgeDrops[i] = 0
        
        for i in range(self.M1):
            self.randomParmutation,self.r= list(np.random.permutation(self.edgeNumbers)),0
           
            self.destructionSpectra()
        tempEdgeDrops={}
        print(self.edgeDrops.items())
        for k,v in self.edgeDrops.items():
           
            if 1 == k: tempEdgeDrops[k]=v
            else: tempEdgeDrops[k] = tempEdgeDrops[k-1]+v
        for k, _ in self.edgeDrops.items():
            self.table1[k][1] = tempEdgeDrops[k] / self.M1
        
