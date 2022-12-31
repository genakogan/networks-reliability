from tabulate import tabulate

class Tables():
    def __init__(self):
        self.table = {}
        self.table1 = {}
        self.table2 = {}
        self.table3 = {}
        self.table4 = {}
        self.table2Part3_2 = {}
    
    # Part 2
    # --------------------------------------------

    # Table 
    # --------------------------------------------    
    def initTable(self):
        for i in range(1,31):
            self.table[i] = [i, -1, -1, -1]

    # Table 1
    # --------------------------------------------
    def initTable1(self):
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
        for p in self.pValues:
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
        for p in self.pValues:
            print(self.table3[p])

    # Table 4
    # --------------------------------------------
    
    def initTable4(self):
        for i in range(1,11):
            self.table4[i] = [i,-1,-1]
        self.table4["r.e"]=["r.e=",-1,-1]

    def printTable4(self):
        print("Destraction Spectrum R(N;p=0.95)")
        for i in range(1, 11):
            print(self.table4[i])
        print(self.table4["r.e"])

    # Part 3
    # --------------------------------------------

    def initTable1Part3(self):
        p = 1
        for i in range(1, 31):
            self.table1[i] = [i, -1, -1]

    def printTable1Part3(self):
        res={'Index':[],'M1':[],'M2':[]}
        for i in self.table1:
            res['Index'].append(self.table1[i][0])
            res['M1'].append(self.table1[i][1])
            res['M2'].append(self.table1[i][2])
        print(tabulate(res, headers="keys"))

    def inittable2Part3_2(self):
        print("asdasdsd")
        p = 0.01
        for i in range(99):
            self.table2Part3_2[p] = [p, -1, -1, -1]
            p += 0.01
            p = "{:.2f}".format(p)
            p = float(p)

    def printTable2Part3(self):
        res = { '9M1000': [], '9M10000': [],'16M1000': [], '16M10000': [],'15M1000': [], '15M10000': [],'21M1000': [], '21M10000': []}
        for edge in  self.table2Part3_2.items():
            for M in edge[1].items():
                for i in M[1]:
                    res[str(edge[0])+'M'+str(M[0])].append(M[1][i])
        print(tabulate(res, headers="keys"))