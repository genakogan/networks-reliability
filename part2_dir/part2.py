from part1_dir.part1 import Part1
import math
class Part2(Part1):
    def __init__(self,x,v,c):
        super().__init__(x,v,c)

    def relativeError(self,values):
        sampleMean = 0
        standardDeviation  = 0
        sum = 0
        for value in values:
            sum += value
        sampleMean = sum / len(values)
        sum=0
        for value in values:
            power = math.pow((value - sampleMean),2)
            sum += power        
        standardDeviation = math.sqrt(sum/(len(values) - 1))
        return standardDeviation/sampleMean

    
       