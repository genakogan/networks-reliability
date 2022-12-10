import networkx as nx
import matplotlib.pyplot as plt
import random
from disjoint_set import DisjointSet
import itertools
import numpy as np
import math
import pickle

class Part2():
    def __init__(self,T1=None,T2=None,T3=None):
        self.r=0
        self.R=-1

        self.T1=T1
        self.T2=T2
        self.T3=T3

        self.M1 = 1000
        self.M2 = 10000
        self.M3 = 50000
        
        self.p_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]

        self.G = nx.Graph()
        self.ds = DisjointSet()
        
        self.edgeDrops = {}
        self.table1 = {}
        self.table2={}
        self.table3 = {}
        self.table4= {}
        self.r = 0
        self.edgeNumbers  = []
        
        self.randomParmutation =[]
       

    # Help function for edge settings. accorsing to step 3
    def booleanValue(self,p,nodeU, nodeV):

        # step 3.1
        # random.uniform(0, 1)
        # Return a random number between, and included, 0 and 1:
        value = random.uniform(0, 1)
        
        # step 3.2
        # if p >= value graph edge in UP else Down
        # g --> up
        # r --> down
        if value <= p:

            # Attach the roots of x and y trees together if they are not the same already.
            # :param x: first element
            # :param y: second element
            self.ds.union(nodeU, nodeV)
            return "g"
        return "r"
    
    def makeGraph(self,p):

        # Disjoint Set Structures
        # Using DSS allow us define if edge in UP or in DOWN
        self.ds=DisjointSet()
        for i in range(1, 21):

            # find() return the canonical element of a given item.
            # In case the element was not present in the data structure,
            # the canonical element is the item itself.
            self.ds.find(i)

        # Creating state vector 
        # Add a single node `node_for_adding` and update node attributes 
        # :first param: node for adding
        # :second param pos: posinton in type: tuple[float, float]     
        self.G.add_node(1,pos=(0.002, -0.80))
        self.G.add_node(2, pos=(-0.96, -0.30))
        self.G.add_node(3, pos=(-0.59, 0.81))
        self.G.add_node(4, pos=(0.59, 0.81))
        self.G.add_node(5, pos=(0.95, -0.31))
        self.G.add_node(6, pos=(2.367, -0.75))
        self.G.add_node(7, pos=(1.76, -2.43))
        self.G.add_node(8, pos=(0.00, -2.5))
        self.G.add_node(9, pos=(-1.76, -2.43))
        self.G.add_node(10, pos=(-2.3, -0.78))
        self.G.add_node(11, pos=(-2.85, 0.93))
        self.G.add_node(12, pos=(-1.461, 2.04))
        self.G.add_node(13, pos=(0, 3.0))
        self.G.add_node(14, pos=(1.4, 1.99))
        self.G.add_node(15, pos=(2.85, 0.93))
        self.G.add_node(16, pos=(4.76, 1.55))
        self.G.add_node(17, pos=(2.94, -4.04))
        self.G.add_node(18, pos=(-2.91, -4.04))
        self.G.add_node(19, pos=(-4.76, 1.55))
        self.G.add_node(20, pos=(0, 5.0))

        # Add an edge between u and v. The nodes u and v will be
        # automatically added if they are not already in the graph.
        # Edge attributes can be specified with keywords or by 
        # directly accessing the edge's attribute dictionary. 
        # :first and second params: nodes
            # Nodes can be, for example, strings or numbers.
            # Nodes must be hashable (and not None) Python objects.
        # attr : keyword arguments, optional
            # Edge data (or labels or objects) can be assigned using
            # keyword arguments.
        self.G.add_edge(1, 2, color=self.booleanValue(p, 1, 2), weight=1)
        self.G.add_edge(2, 3, color=self.booleanValue(p, 2, 3), weight=2)
        self.G.add_edge(3, 4, color=self.booleanValue(p, 3, 4), weight=3)
        self.G.add_edge(4, 5, color=self.booleanValue(p, 4, 5), weight=4)
        self.G.add_edge(5, 1, color=self.booleanValue(p, 5, 1), weight=5)
        self.G.add_edge(6, 7, color=self.booleanValue(p, 6, 7), weight=6)
        self.G.add_edge(7, 8, color=self.booleanValue(p, 7, 8), weight=7)
        self.G.add_edge(8, 9, color=self.booleanValue(p, 8, 9), weight=8)
        self.G.add_edge(9, 10, color=self.booleanValue(p, 9, 10), weight=9)
        self.G.add_edge(10, 11, color=self.booleanValue(p, 10, 11), weight=10)
        self.G.add_edge(11, 12, color=self.booleanValue(p, 11, 12), weight=11)
        self.G.add_edge(12, 13, color=self.booleanValue(p, 12, 13), weight=12)
        self.G.add_edge(13, 14, color=self.booleanValue(p, 13, 14), weight=13)
        self.G.add_edge(14, 15, color=self.booleanValue(p, 14, 15), weight=14)
        self.G.add_edge(15, 6, color=self.booleanValue(p, 15, 6), weight=15)
        self.G.add_edge(16, 17, color=self.booleanValue(p, 16, 17), weight=16)
        self.G.add_edge(17, 18, color=self.booleanValue(p, 17, 18), weight=17)
        self.G.add_edge(18, 19, color=self.booleanValue(p, 18, 19), weight=18)
        self.G.add_edge(19, 20, color=self.booleanValue(p, 19, 20), weight=19)
        self.G.add_edge(20, 16, color=self.booleanValue(p, 20, 16), weight=20)
        self.G.add_edge(15, 16, color=self.booleanValue(p, 15, 16), weight=21)
        self.G.add_edge(7, 17, color=self.booleanValue(p, 7, 17), weight=22)
        self.G.add_edge(9, 18, color=self.booleanValue(p, 9, 18), weight=23)
        self.G.add_edge(11, 19, color=self.booleanValue(p, 11, 19), weight=24)
        self.G.add_edge(20, 13, color=self.booleanValue(p, 20, 13), weight=25)
        self.G.add_edge(1, 8, color=self.booleanValue(p, 1, 8), weight=26)
        self.G.add_edge(2, 10, color=self.booleanValue(p, 2, 10), weight=27)
        self.G.add_edge(3, 12, color=self.booleanValue(p, 3, 12), weight=28)
        self.G.add_edge(4, 14, color=self.booleanValue(p, 4, 14), weight=29)
        self.G.add_edge(5, 6, color=self.booleanValue(p, 5, 6), weight=30)

    def calculateDSS(self):
    
        # Return True if T1 and T2 and T3 belong to the same set.
        return self.ds.connected(self.T1,self.T2) and self.ds.connected(self.T2,self.T3)

    # --------------------------------------------
    def initTable1(self):
        p = 1
        for i in range(1,31):
            self.table1[i] = [i, -1, -1, -1]

    def printTable1(self):
        for p in range(1, 31):
            print(self.table1[p])

    # --------------------------------------------
    def initTable2(self):
        p = 0.01
        for i in range(99):
            self.table2[p] = [p, -1, -1,-1]
            p += 0.01
            p = "{:.2f}".format(p)
            p = float(p)

    def printTable2(self):
        for p in self.p_values:
            print(self.table2[p])

    # --------------------------------------------
    def initTable3(self):
        p = 0.01
        for i in range(99):
            self.table3[p] = [p, -1, -1, -1]
            p += 0.01
            p = "{:.2f}".format(p)
            p = float(p)

    def printTable3(self):
        for p in self.p_values:
            print(self.table3[p])


    def printAll(self):
        print("T1=" + str(self.T1) + " T2=" + str(self.T2) + " T3=" + str(self.T3))
        print("-----------------------------------")
        print("Random Permutation:")
        print(self.randomParmutation)
        print("-----------------------------------")
        
        print("Table 1: Destruction Spectra:")
        print("  i    M1=1000   M2=10,000  M3=50,000")
        self.printTable1()
        print("-----------------------------------")

        print("Table 2: R(N;P)(Destruction Spectrum)")
        print("  p    M1=1000   M2=10,000  M3=50,000")
        self.printTable2()
        print("-----------------------------------")
        print("Table 3: R(N;P)(Destruction Spectrum)")
        print("  i    M1=1000   M2=10,000  M3=50,000")
        self.printTable3()
    def makeDSS(self):
        for i in range(1, 21):
            self.ds.find(i)

 # --------------------------------------------------------------------------------           
    def destructionSpectra(self):
        tempDict = {}
        self.makeGraph(1)
        for k,v in self.G.edges():
            vertex = self.G[k][v]
            tempDict[vertex['weight']] = [k,v]
        #G[][]['color'] = 'r'
        for edge in self.randomParmutation:
            vertex = tempDict[edge]
            edge = self.G[vertex[0]][vertex[1]]
            edge['color'] = 'r'

            self.ds = DisjointSet()
            self.makeDSS()

            for u,v in self.G.edges():
                if self.G[u][v]['color'] == 'g':
                    self.ds.union(u, v)
            result = self.calculateDSS()
            if result == True:
                self.r += 1
            else:
                self.edgeDrops[self.r] += 1
                break

    def part2_1(self):
        self.initTable1()
        for i in range(1,31):
            self.edgeNumbers.append(i)
            self.edgeDrops[i] = 0
        for i in range(1, 31):
            self.edgeDrops[i] = 0

        for i in range(self.M1):
            self.randomParmutation = list(np.random.permutation(self.edgeNumbers))
            self.r = 0
            self.destructionSpectra()
        for k,v in self.edgeDrops.items():
            self.table1[k][1] = self.edgeDrops[k] / self.M1

        self.edgeDrops={}
        for i in range(1,31):
             self.edgeDrops[i] = 0

        for i in range(self.M2):
            self.randomParmutation = list(np.random.permutation(self.edgeNumbers))
            self.r = 0
            self.destructionSpectra()
        for k,v in  self.edgeDrops.items():
            self.table1[k][2] =  self.edgeDrops[k] / self.M2

        self.edgeDrops={}
        for i in range(1,31):
             self.edgeDrops[i] = 0

        for i in range(self.M3):
            self.randomParmutation = list(np.random.permutation(self.edgeNumbers))
            self.r = 0
            self.destructionSpectra()
        for k,v in  self.edgeDrops.items():
            self.table1[k][3] =  self.edgeDrops[k] /self.M3

        with open('table1.pickle', 'wb') as handle:
            pickle.dump(self.table1, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('table1.pickle', 'rb') as handle:
            self.table1 = pickle.load(handle)

# -------------------------------------------------------------------------------- 
    def nCr(self,n,r):
        f = math.factorial
        return f(n) / f(r) / f(n-r)
    def calcFs(self,p_values):
        f_table={}
        p_table={}
        for q in p_values:
            p = 1 - q
            q = float(q)
            p = float(p)

            p = round(p,2)
            for n in range(1,31):
                for k in range(1,30 - n + 2):
                    ncrr=int(self.nCr(30,k))

                    q_pow_k=math.pow(q,k)

                    p_n_minus_k=pow(p,(30-k))

                    resultTemp= ncrr* q_pow_k * p_n_minus_k

                f_table[n] = resultTemp

            p_table[q]=f_table
            f_table = {}
        return p_table
    def calcFsMultiplication(self,f_table,tableX,index):
        sum=0
        calcFi = 0
        for k,v in tableX.items(): # K is row number
            for value in range(1,k):
                calcFi += f_table[value]
            sum += calcFi * tableX[k][index]
            calcFi = 0
        return sum
    def part2_2(self):
        self.initTable2()
        p_table=self.calcFs(self.p_values)

        for p in self.p_values:
            f_table=p_table[p]
            for i in range (1,4):
                result=self.calcFsMultiplication(f_table,self.table1,i)
                self.table2[p][i]=result

        with open('table2.pickle', 'wb') as handle:
            pickle.dump( self.table2, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.printTable2()


        with open('table2.pickle', 'rb') as handle:
            self.table2 = pickle.load(handle)

# --------------------------------------------------------------------------------     
    def part2_3(self):
        self.initTable3()
        for p in self.p_values:
            for i in range(self.M1):

                self.ds = DisjointSet()
                self.makeGraph(p)
                result = self.calculateDSS()
                if (result == True):
                    self.r += 1
            self.R = self.r / self.M1
            self.table3[p][1] = self.R

            for i in range(self.M2):
                self.ds = DisjointSet()
                self.makeGraph(p)
                result = self.calculateDSS()
                if (result == True):
                    self.r += 1

            self.R = self.r / self.M2
            self.table3[p][2] = self.R


            for i in range(self.M3):
                self.ds = DisjointSet()
                self.makeGraph(p)
                result = self.calculateDSS()
                if (result == True):
                    self.r += 1

            self.R = self.r / self.M3
            self.table3[p][3] = self.R

            self.r = 0

        #printTable3(p_values)
        with open('table3.pickle', 'wb') as handle:
            pickle.dump(self.table3, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #part 3 output --- done ---- !
        with open('table3.pickle', 'rb') as handle:
            self.table3 = pickle.load(handle)
#printTable3(p_values)
#printTable3(p_values)
if __name__ == "__main__":
    p=Part2(2,5,19)
    p.part2_1()
    p.part2_2()
    p.part2_3()
    p.printAll()