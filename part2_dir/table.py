

class Tables():
    def __init__(self):
        self.table = {}
        self.table1 = {}
        self.table2 = {}
        self.table3 = {}
        self.table4 = {}
    
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


