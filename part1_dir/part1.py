# Genady Kogan

# dependency
import random

#Data structures for graphs, digraphs, and multigraphs
#Many standard graph algorithms
#Network structure and analysis measures
#Generators for classic graphs, random graphs, and synthetic networks#
import networkx as nx

# using Disjoint Set Structures according 4
from disjoint_set import DisjointSet
import matplotlib.pyplot as plt


class Part1():
    def __init__(self,T1=None,T2=None,T3=None):
        self.r=0

        # reliabilityNetwork = R
        self.R=-1
        self.G = nx.Graph()

        # init Disjoint Set Structures according to step 4
        self.ds = DisjointSet()

        # M1 and M2 according D
        self.M1=1000
        self.M2=10000

        # p values from accordng C
        self.pValues = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
        self.table = {}

        self.stateVector=[]

        # get from teacher 
        self.T1=T1
        self.T2=T2
        self.T3=T3
    def getM1(self):
        return self.M1
    def getM2(self):
        return self.M2
    def getPValues(self):
        return self.pValues
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

    # Step A
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
        
    def creatingStateVector(self):
        print("DSS")
        # Printing connected node
        print(list(self.ds.itersets()))

        # get edges from G --> nx.Graph()
        edges = list(self.G.edges())
        state = {}
        for i in range(len(edges)):

            # Define state vector  for step 4
            state[edges[i]] = [self.G[u][v]['color'] for u, v in self.G.edges()][i]
        for k,v in state.items():
                self.stateVector.append((k,v))
        
        # Printing stata vector depending on p value
        print("State Vector")
        print(self.stateVector)

    # dodecahedron network representation
    def graphShow(self):
        self.makeGraph(0.5)

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
        nx.draw(self.G, pos=nx.get_node_attributes(self.G, name='pos'), arrows=False,
                edge_color=[self.G[u][v]['color'] for u, v in self.G.edges()],
                node_color='#5e488a', with_labels=True, width=1.5)
        
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

    # Step 6 using M1
    # Calculating reliability network
    def calculateReliabilityNetworkM1(self):
        return self.r/self.M1

    # Step 6 using M2
    # Calculating reliability network using M1 
    def calculateReliabilityNetwork(self,Mx):
        return self.r/Mx

    # Step D
    # Craeting table 
    def createTable(self):
        p = 0.01
        for _ in range(99):
            self.table[p] = [p, -1, -1]
            p += 0.01
            p = "{:.2f}".format(p)
            p = float(p)
    
    # Printing table 
    def printTable(self):
        print("Terminals:"+" T1=" + str(self.T1) + " T2=" + str(self.T2) + " T3=" + str(self.T3))
        print("\n  P    M1   M2\n")
        for p in self.pValues: print(self.table[p])
    
    # step B
    def calculateDSS(self):
        
        # Return True if T1 and T2 and T3 belong to the same set.
        return self.ds.connected(self.T1,self.T2) and self.ds.connected(self.T2,self.T3)

    # Step C
    # Calculating network reliability for p value
    def assessNetworkReliability(self):
        self.createTable()
        for p in self.pValues:
           
            for _ in range(self.M1):
                
                # Function from according to A
                self.makeGraph(p)

                # Step 4
                if True == self.calculateDSS(): self.r+=1
            self.table[p][1], self.r = self.calculateReliabilityNetwork(self.M1),0
            
            for _ in range(self.M2):
                
                # Function from according to A
                self.makeGraph(p)

                # Step 4
                if True == self.calculateDSS(): self.r+=1
            self.table[p][2], self.r = self.calculateReliabilityNetwork(self.M2),0
        self.printTable()

'''
if __name__ == "__main__":
    p=part1(2,5,19)
    p.graphShow() 
    p.assessNetworkReliability()
'''   
    