from Chemical import *


class Cell() :
    def __init__(self,start_energy,nuclear=False):
        


class Cluster() :
    def __init__(self,nuclear) :
        self.__nuclear:Cell = nuclear
        self.__cells:list[Cell] = []

class Block() :
    def __init__(self,x,y):
        self.__x = x
        self.__y = y
        self.__cluster:Cluster = None
        self.__temperature:float = None
        self.__substances = {}
        
    def rewrite_cluster(self,cluster:Cluster):
        self.__cluster = cluster
        
    def add_substance(self,substances:list[Chemical]):
        for substance in substances:
            if substance in self.__substances.keys():
                self.__substances[substance] += 1
            elif substance not in self.__substances.keys():
                self.__substances[substance] = 1
            else :
                raise ValueError()
            
    def rewrite_temperature(self,temperature:float) :
        self.__temperature = temperature
        

class Field() :
    def __init__(self,h:int,w:int) :
        self.__field:list = [[None for k in range(h)] for l in range(w)]
