from Chemical import *
import numpy as np
from itertools import islice


class Cell() :
    def __init__(self,start_energy,gene):
        self.__energy = start_energy
        self.__gene = gene
        self.__storage = {}
        self.__energy_max = None #geneからとってくる
        self.__chem_max = None #geneからとってくる
        self.__temperature_max = None #geneからとってくる
        
    def unzip(self,gene):
        """
        A:0 B:1 C:2 D:3 E:4 F:5 G:6 H:7 I:8 J:9 K:,
        """
        gene_dictionary = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":","}
        list_gene = list(gene)

    def detect(self):
        pass


class Cluster() :
    pass


class Block() :
    def __init__(self,x,y):
        self.__x = x
        self.__y = y
        self.__cluster:Cluster = None
        self.__temperature:float = 0
        self.__lightness:float = 0
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

    def get_info(self):
        return self.__temperature, self.__lightness, self.__substances
        

class Field() :
    def __init__(self,h:int,w:int) :
        self.__field:list = [[None for k in range(h)] for l in range(w)]
        self.__h = h
        self.__w = w

    def add_block(self, block:Block, x:int, y:int):
        if x < 0 or x >= len(self.__field) or y < 0 or y >= len(self.__field[0]):
            raise IndexError()
        self.__field[x][y] = block

    def init_block(self) :
        for i in range(self.__h):
            for j in range(self.__w):
                block = Block(i, j)
                self.add_block(block, i, j)

class CMS() :
    """Center Management System."""
    def __init__(self, field:Field, cluster_num:int) :
        self.__field = field
        self.__clusters:list = [Cluster() for _ in range(cluster_num)]
        self.__current_cluster_num = 0 #この上のCluster引数あとで変える必要あり


field = Field(10,10)
field.init_block()
