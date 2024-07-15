from Chemical import *
import numpy as np
from itertools import islice


class Cell() :
    def __init__(self,start_energy,gene,nuclear=False):
        self.__energy = start_energy
        self.__gene = gene
        self.__nuclear:bool = nuclear
        self.__storage = {}
        self.__energy_max = None #geneからとってくる
        self.__chem_max = None #geneからとってくる
        self.__temperature_max = None #geneからとってくる
        
    def unzip(self,gene):
        """
        AAA:0 AAB:1 AAC:2 AAD:3 ABA:4 ABB:5 ABC:6 ABD:7 ACA:8 ACB:9 ACC:, ACD:a ADA:b ADB:c ADC:d ADD:e
        """
        gene_dictionary = {"AAA":0,"AAB":1,"AAC":2,"AAD":3,"ABA":4,"ABB":5,"ABC":6,"ABD":7,"ACA":8,"ACB":9,"ACC":",","ACD":"a","ADA":"b","ADB":"c","ADC":"d","ADD":"e"}
        list_gene = [gene[i:i+3] for i in range(0, len(gene), 3)]


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
