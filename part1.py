# Genady Kogan
# dependency
import os
import sys


class part1():
    def __init__(self):
        self.r=0
        self.R=-1
        self.M1=1000
        self.M2=10000
        self.pValues = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
        self.table = {}
        # get from teacher 
        self.T1=0
        self.T2=0
        self.T3=0

    # Craeting table according D
    def createTable(self):
        p = 0.01
        for i in range(99):
            self.table[p] = [p, -1, -1]
            p += 0.01
            p = "{:.2f}".format(p)
            p = float(p)
            
    # Printing table according D
    def printTable(self):
        print("Terminals:"+" T1=" + str(self.T1) + " T2=" + str(self.T2) + " T3=" + str(self.T3))
        print("  P    M1   M2")
        print(" -------------")
        for p in self.pValues:
            print(self.table[p])

if __name__ == "__main__":
    test=part1()
    test.createTable()
    test.printTable()
    