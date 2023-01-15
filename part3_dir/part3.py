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

        # Part3_3
        self.mostImportant = [2,4]
        self.lessImportant = [16, 20]
        self.P = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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

        # Cumulative Destruction Spectra 
        cumulativeEdgeDrops={}
        for k,v in self.edgeDrops.items():
            if k==1:
                cumulativeEdgeDrops[k]=v
            else:
                cumulativeEdgeDrops[k] = cumulativeEdgeDrops[k-1]+v
        for k,v in self.edgeDrops.items():
            self.table1[k][1] = cumulativeEdgeDrops[k] / self.M1

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

        # Cumulative Destruction Spectra 
        cumulativeEdgeDrops={}

        for k,v in self.edgeDrops.items():
            if k==1:
                cumulativeEdgeDrops[k]=v
            else:
                cumulativeEdgeDrops[k] = cumulativeEdgeDrops[k-1]+v
       
        for k,v in self.edgeDrops.items():
            self.table1[k][2] = cumulativeEdgeDrops[k] / self.M2
        

# ------------------------------------------------------------------------------------

    def destructionBimSpectra(self):
        
        tempDict = {}
        self.makeGraph(1)
        for k, v in self.G.edges():
            vertex = self.G[k][v]
            tempDict[vertex['weight']] = [k, v]
       
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
            
            if self.calculateDSS():
                self.r += 1
            else:
                # from up to down
                self.a[self.r+1] += 1
                while self.r < 30:
                    # for all j in down put b(r,j)+=1
                    for i in range(0, self.r+1):
                        self.b[self.r + 1][self.randomParmutation[i]] += 1
                    self.r += 1
                break

    def part3_2_3(self):
        
        self.table2Part3_2 = {}

        # Iteration M1
        edgeDrops = {}

        # step 1
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
            # step 2
            self.randomParmutation = list(np.random.permutation(self.edgeNumbers))
            self.r = 0
            self.destructionBimSpectra()

        # step 8
        z = {}
        y = {}
        for i in range(1, 31):
            z[i] = {}
            y[i] = self.a[i]/ self.M1
            for j in range(1, 31):
                z[i][j] = self.b[i][j]/ self.M1

        for edge in self.mostImportant:
            self.table2Part3_2[edge] = {self.M1: {}, self.M2: {}}
            for i in range(1, 31):
                self.table2Part3_2[edge][self.M1][i] = z[i][edge]
        for edge in self.lessImportant:
            self.table2Part3_2[edge] = {self.M1: {}, self.M2: {}}
            for i in range(1, 31):
                self.table2Part3_2[edge][self.M1][i] = z[i][edge]

       
        # part 3 with
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

        for edge in self.mostImportant:
            for i in range(1, 31):
                self.table2Part3_2[edge][self.M2][i] = z[i][edge]
        for edge in self.lessImportant:
            for i in range(1, 31):
                self.table2Part3_2[edge][self.M2][i] = z[i][edge]

        

        

# ------------------------------------------------------------------------------------
    
    def part3_4(self):
       
        def calcBim(j, p,Z, Y):
            res = 0
            for i in range(1, 31):
                pq = math.pow(1-p, (i - 1)) * math.pow(p, (30 - i))
                pq2 = math.pow(1-p, i) * math.pow(p, (30 - i - 1))
                natz=math.factorial(30)          
                res += (((Z[i][j] * pq)-(Y[i] - Z[i][j]) * pq2)*natz)/(math.factorial(i) * math.factorial(30 - i))
            return abs(res)

        edgeDrops,self.a,self.b = {},{},{}
       
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

        Z,Y= {},{}
        
        for i in range(1, 31):
            Z[i] = {}
            Y[i] = self.a[i] / self.M2
            for j in range(1, 31): Z[i][j] = self.b[i][j] / self.M2

        # BIMs and Gain in Reliability
        self.table3Part3={}
       
        for p in self.P: self.table3Part3[p]={
                 
                'BIM10':calcBim(2,p,Z,Y),
                'BIM10Sp':calcBim(2,p,Z,Y)*(1-p),
                'BIM20':calcBim(4,p,Z,Y),
                'BIM20Sp':calcBim(4,p,Z,Y)*(1-p),
                'BIM8':calcBim(16,p,Z,Y),
                'BIM8Sp':calcBim(16,p,Z,Y)*(1-p),
                'BIM23':calcBim(20,p,Z,Y),
                'BIM23Sp':calcBim(20,p,Z,Y)*(1-p)
            }


# ------------------------------------------------------------------------------------

    
    def newMakeGraph(self,p):
        self.ds=DisjointSet()
        for i in range(1, 21):
            self.ds.find(i)
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

        self.G.add_edge(1, 2, color=self.booleanValue(p, 1, 2), weight=1)
        self.G.add_edge(2, 3, color=self.booleanValue(1, 2, 3), weight=2)
        self.G.add_edge(3, 4, color=self.booleanValue(p, 3, 4), weight=3)
        self.G.add_edge(4, 5, color=self.booleanValue(1, 4, 5), weight=4)
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
        self.G.add_edge(16, 17, color=self.booleanValue(0, 16, 17), weight=16)
        self.G.add_edge(17, 18, color=self.booleanValue(p, 17, 18), weight=17)
        self.G.add_edge(18, 19, color=self.booleanValue(p, 18, 19), weight=18)
        self.G.add_edge(19, 20, color=self.booleanValue(p, 19, 20), weight=19)
        self.G.add_edge(20, 16, color=self.booleanValue(0, 20, 16), weight=20)
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
    
    def part3_5(self):
        
        edges = [2,4,16,20]
        for edge in edges:
            self.table4Part3[edge]={}
            for p in self.pValues:
                for _ in range(self.M2):
                    self.ds = DisjointSet()
                    self.newMakeGraph(p)
                    result = self.calculateDSS()
                    if (result == True):
                        self.r += 1
                self.R1,self.r= self.r / self.M2,0
               
                for _ in range(self.M2):
                    self.ds = DisjointSet()
                    self.newMakeGraph(p)
                    result = self.calculateDSS()
                    if (result == True):
                        self.r += 1
                self.R2,self.r = self.r / self.M2,0
                self.table4Part3[edge][p]=abs(self.R2-self.R1)

    # Part 6
    # ======================================================================

    def graphShowPart3(self):
        def terminals(T):
            if 2==T or 5==T or 19==T:
                return "#1916EA"
            return '#16EA16'

        self.makeGraphPart3_6(1)

        # Create a figure and a set of subplots.
        # This utility wrapper makes it convenient to create
        # common layouts of subplots, including the enclosing
        # figure object, in a single call.
        plt.subplots(1, 1, figsize=(8, 8))

        # creating state vector from step 4
        self.creatingStateVector()
   
        # Draw the graph as a simple representation with no node
        # labels or edge labels and using the full Matplotlib figure area
        # and no axis labels by default.
        # :pos: Get node attributes from graph
        nodeColors=[terminals(T) for T in self.G.nodes()]

        nx.draw(self.G, pos=nx.get_node_attributes(self.G, name='pos'), arrows=False,
                edge_color=[self.G[u][v]['color'] for u, v in self.G.edges()],
                node_color=nodeColors, with_labels=True, width=1.5)
        
        # Draw edge labels
        # :pos: Get node attributes from graph
        # :edge_labels: Get edge attributes from graph
        nx.draw_networkx_edge_labels(self.G, pos=nx.get_node_attributes(self.G, name='pos'),
                                     edge_labels=nx.get_edge_attributes(self.G, 'weight'))

        # Return the figure manager of the current figure.
        # The figure manager is a container for the actual
        # backend-depended window that displays the figure
        # on screen. If no current figure exists, a new one
        # is created, and its figure manager is returned.
        plt.get_current_fig_manager().window.wm_geometry("+300+25")

        #  Display all open figures.
        plt.show()

    def makeGraphPart3_6(self,p):

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
        self.G.add_edge(1, 2, color='red', weight=1)
        self.G.add_edge(2, 3, color='red', weight=2)
        self.G.add_edge(3, 4, color='#9A16EA', weight=3)
        self.G.add_edge(4, 5, color='red', weight=4)
        self.G.add_edge(5, 1, color='red', weight=5)
        self.G.add_edge(6, 7, color='#9A16EA', weight=6)
        self.G.add_edge(7, 8, color='#9A16EA', weight=7)
        self.G.add_edge(8, 9, color='#9A16EA', weight=8)
        self.G.add_edge(9, 10, color='#9A16EA', weight=9)
        self.G.add_edge(10, 11, color='#9A16EA', weight=10)
        self.G.add_edge(11, 12, color='#9A16EA', weight=11)
        self.G.add_edge(12, 13, color='#9A16EA', weight=12)
        self.G.add_edge(13, 14, color='#9A16EA', weight=13)
        self.G.add_edge(14, 15, color='black', weight=14)
        self.G.add_edge(15, 6, color='#9A16EA', weight=15)
        self.G.add_edge(16, 17, color='black', weight=16)
        self.G.add_edge(17, 18, color='black', weight=17)
        self.G.add_edge(18, 19, color='red', weight=18)
        self.G.add_edge(19, 20, color='red', weight=19)
        self.G.add_edge(20, 16, color='black', weight=20)
        self.G.add_edge(15, 16, color='black', weight=21)
        self.G.add_edge(7, 17, color='black', weight=22)
        self.G.add_edge(9, 18, color='#9A16EA', weight=23)
        self.G.add_edge(11, 19, color='red', weight=24)
        self.G.add_edge(20, 13, color='#9A16EA', weight=25)
        self.G.add_edge(1, 8, color='#9A16EA', weight=26)
        self.G.add_edge(2, 10, color='red', weight=27)
        self.G.add_edge(3, 12, color='#9A16EA', weight=28)
        self.G.add_edge(4, 14, color='#9A16EA', weight=29)
        self.G.add_edge(5, 6, color='red', weight=30)