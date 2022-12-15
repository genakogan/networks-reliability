import networkx as nx
import matplotlib.pyplot as plt
import random
from disjoint_set import DisjointSet
import itertools
import numpy as np
import math
import pickle
from part1_dir.part1 import Part1
class Part2(Part1):
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
        self.table = {}
        self.table1 = {}
        self.table2 = {}
        self.table3 = {}
        self.table4 = {}
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
   
    def calculateDSS(self):
    
        # Return True if T1 and T2 and T3 belong to the same set.
        return self.ds.connected(self.T1,self.T2) and self.ds.connected(self.T2,self.T3)

# Tables

    def initTable(self):
        p = 1
        for i in range(1,31):
            self.table[i] = [i, -1, -1, -1]


    # Table 1
    # --------------------------------------------
    def initTable1(self):
        p = 1
        for i in range(1,31):
            self.table1[i] = [i, -1, -1, -1]

    def printTable1(self):
        for p in range(1, 31):
            print(self.table1[p])

    # Table 2
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

    # Table 3
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

    # Table 4
    # --------------------------------------------
    
    def initTable4(self):
        p = 0.01
        for i in range(1,11):
            self.table4[i] = [i,-1,-1]
        self.table4["r.e"]=["r.e=",-1,-1]

    def printTable4(self):
        print("Destraction Spectrum R(N;p=0.95)")
        for i in range(1, 11):
            print(self.table4[i])
        print(self.table4["r.e"])
       
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

        print("-----------------------------------")
        print("Table 4: R(N;p = 0.95)")
        self.printTable4()

    def makeDSS(self):
        for i in range(1, 21):
            self.ds.find(i)
# Part 2_1
# --------------------------------------------------------------------------------           
    def destructionSpectra(self):
        tempDict = {}
        Part1.makeGraph(self,1)
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
            result = Part1.calculateDSS(self)
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
# Part 2_2
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
# Part 2_3
# --------------------------------------------------------------------------------     

    def part2_3(self):
        self.initTable3()
        for p in self.p_values:
            for _ in range(self.M1):

                self.ds = DisjointSet()
                Part1.makeGraph(self,p)
                result = Part1.calculateDSS(self)
                if (result == True):
                    self.r += 1
                       
            self.table3[p][1],self.r = Part1.calculateReliabilityNetwork(self,self.M1),0
          
            for _ in range(self.M2):
                self.ds = DisjointSet()
                Part1.makeGraph(self,p)
                result = Part1.calculateDSS(self)
                if (result == True):
                    self.r += 1
           
            self.table3[p][2],self.r = Part1.calculateReliabilityNetwork(self,self.M2),0

            for _ in range(self.M3):
                self.ds = DisjointSet()
                Part1.makeGraph(self,p)
                result = Part1.calculateDSS(self)
                if (result == True):
                    self.r += 1

            self.table3[p][3], self.r = Part1.calculateReliabilityNetwork(self,self.M3),0
     
        with open('table3.pickle', 'wb') as handle:
            pickle.dump(self.table3, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #part 3 output --- done ---- !
        with open('table3.pickle', 'rb') as handle:
            self.table3 = pickle.load(handle)

# Part 2_4
# --------------------------------------------------------------------------------
    
    def part2_4(self):
        table = {}
        self.initTable4()
        for i in range(1,31):
            table[i] = [i, -1, -1, -1]

        for i in range (1,11):
            self.edgeDrops = {}
            for h in range(1, 31):
                self.edgeDrops[h] = 0

            for j in range (self.M1):
                self.randomParmutation = list(np.random.permutation(self.edgeNumbers))
                self.r = 0
                self.destructionSpectra()
            for k,v in self.edgeDrops.items():
                table[k][1] = self.edgeDrops[k] / self.M1

            p_table=self.calcFs([0.95])
            f_table=p_table[0.95]


            result=self.calcFsMultiplication(f_table, table,1)

            self.table4[i][1]=result

        with open('table4.pickle', 'wb') as handle:
            pickle.dump(self.table4, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('table4.pickle', 'rb') as handle:
            self.table4 = pickle.load(handle)

# Part 2_5
# --------------------------------------------------------------------------------
    def relativeError(self, values):
        avrg, S, sum = 0,0,0
        

        for value in values:
            avrg+= value
        avrg  = avrg / len(values)

        for value in values:
            power = math.pow((value - avrg),2)
            sum += power

        S = math.sqrt(sum/(len(values) - 1 ))
        return S/avrg
    
    def part2_5(self):
       for j in range(1,11):
            self.r = 0
            for i in range(self.M1):

                self.ds = DisjointSet()
                Part1.makeGraph(self,0.95)
                result = Part1.calculateDSS(self)
                if (result == True):
                    self.r += 1
            R = self.r / self.M1
            self.table4[j][2] = R
            self.r = 0

    def part2_4_5(self):
        r_e4 = []
        r_e5 = []
        for k,v in self.table4.items():
            r_e4.append(v[1])
            r_e5.append(v[2])
        del r_e4[-1]
        del r_e5[-1]
        r_e4 = self.relativeError(r_e4)
        r_e5 = self.relativeError(r_e5)
        self.table4["r.e"][1] = r_e4
        self.table4["r.e"][2] = r_e5
    
        with open('table4.pickle', 'wb') as handle:
            pickle.dump(self.table4, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('table4.pickle', 'rb') as handle:
            self.table4 = pickle.load(handle)