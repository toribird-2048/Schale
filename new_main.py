import numpy as np
from dataclasses import dataclass


class Cell :
    def __init__(self,energy,red,green,blue,white,detect_max_list,detect_min_list,neural,stage,ClusterID):
        """
        Detect_list : In Stage -> Info1~Info16,energy,red,green,blue | In Cell -> energy,red,green,blue
        """
        self.storage:list[float,int,int,int,int] = [energy,red,green,blue,white]
        self.detect_max_list:list = detect_max_list # Length must be 21 
        self.detect_min_list:list = detect_min_list # Length must be 21 
        self.neural:np.array = neural
        self.outputs:list[float,int,int,int,int] = [0,0,0,0,0]
        self.maxes:list[float,int,int,int,int] = [10,2,2,2,0]
        self.stage:int = stage
        self.ClusterID:int = ClusterID
        self.input_queue:list[float,int,int,int,int] = [0,0,0,0,0]
    
    def __post_init__(self):
        for k in range(len(self.storage)) :
            if self.maxes[k] <= self.storage[k] :
                self.storage[k] = self.maxes[k]
    
    def detector(self):
        detecteds = []
        for k in range(21):
            if ClusterIDs[self.ClusterID].info_stage(self.stage)[k] <= self.detect_min_list[k] :
                detecteds.append(-1)
            elif self.detect_max_list[k] <= ClusterIDs[self.ClusterID].info_stage(self.stage)[k] :
                detecteds.append(1)
            else :
                detecteds.append((ClusterIDs[self.ClusterID].info_stage(self.stage)[k] - self.detect_min_list[k]) / (self.detect_max_list[k] - self.detect_min_list[k]) - 0.5)
        detecteds.append(2*self.storage[0]/self.maxes[0]-0.5)
        detecteds.append(2*self.storage[1]/self.maxes[1]-0.5)
        detecteds.append(2*self.storage[2]/self.maxes[2]-0.5)
        detecteds.append(2*self.storage[3]/self.maxes[3]-0.5)
        return detecteds

    def neural_net(self):
        output = np.dot(self.neural,np.array(self.detector()).T)
        output_list = output.T
        adds = output_list[:4]
        removes = output_list[4:]
        for k in range(len(adds)):
            self.input_queue[k] += adds[k]
        pass
        #TODO: ClusterIDsの中のClusterのstageから必要数を引いてadd_storageするところ

    def info(self):
        return f"energy : {self.storage[0]}, red : {self.storage[1]}, green : {self.storage[2]}, blue : {self.storage[3]}, white : {self.storage[4]}, ClusterID : {self.ClusterID}, stage : {self.stage}"
    
    def input(self,inputs:list) :
        ClusterIDs[self.ClusterID].remove_stage(self.stage,17,1)
        ClusterIDs[self.ClusterID].remove_stage(self.stage,18,1)
        ClusterIDs[self.ClusterID].remove_stage(self.stage,19,1)
        ClusterIDs[self.ClusterID].remove_stage(self.stage,20,1)
        self.add_storage([])

    def add_storage(self,inputs):
        """
        energy,red,green,blue
        """
        output = [0,0,0,0,0]
        if self.maxes[0] <= self.storage[0] + inputs[0]:
            self.storage[0] = self.maxes[0]
            output[0] = self.storage[0] + inputs[0] - self.maxes[0]
        else :
            self.storage[0] += inputs[0]
        if self.maxes[1] <= self.storage[1] + inputs[1]:
            self.storage[1] = self.maxes[1]
            output[1] = self.storage[1] + inputs[1] - self.maxes[1]
        else :
            self.storage[1] += inputs[1]
        if self.maxes[2] <= self.storage[2] + inputs[2]:
            self.storage[2] = self.maxes[2]
            output[2] = self.storage[2] + inputs[2] - self.maxes[2]
        else :
            self.storage[2] += inputs[2]
        if self.maxes[3] <= self.storage[3] + inputs[3]:
            self.storage[3] = self.maxes[3]
            output[3] = self.storage[3] + inputs[3] - self.maxes[3]
        else :
            self.storage[3] += inputs[3]
        if self.maxes[4] <= self.storage[4] + inputs[4]:
            self.storage[4] = self.maxes[4]
            output[4] = self.storage[4] + inputs[4] - self.maxes[4]
        else :
            self.storage[4] += inputs[4]
        self.outputs = [self.outputs[0]+output[0], self.outputs[1]+output[1], self.outputs[2]+output[2], self.outputs[3]+output[3], self.outputs[4]+output[4]]
    
    def remove_storage(self,inputs):
        """
        energy,red,green,blue
        """
        output = [0,0,0,0,0]
        if self.storage[0] >= inputs[0]:
            self.storage[0] -= inputs[0]
            output[0] = inputs[0]
        else :
            raise ValueError
        if self.storage[1] >= inputs[1]:
            self.storage[1] -= inputs[1]
            output[1] = inputs[1]
        else :
            raise ValueError
        if self.storage[2] >= inputs[2]:
            self.storage[2] -= inputs[2]
            output[2] = inputs[2]
        else :
            raise ValueError
        if self.storage[3] >= inputs[3]:
            self.storage[3] -= inputs[3]
            output[3] = inputs[3]
        else :
            raise ValueError
        if self.storage[4] >= inputs[4]:
            self.storage[4] -= inputs[4]
            output[4] = inputs[4]
        else :
            raise ValueError
        self.outputs = [self.outputs[0]+output[0], self.outputs[1]+output[1], self.outputs[2]+output[2], self.outputs[3]+output[3], self.outputs[4]+output[4] ]


class InOut(Cell) :
    pass


class Factory(Cell) :
    def __init__(self,*args) :
        super().__init__(*args)
        self.maxes[4] = 2
    
    def create_white(self):
        if self.storage[0] >= 1 and self.storage[1] >= 1 and self.storage[2] >= 1 and self.storage[3] >= 1:
            self.add_storage([0,0,0,0,1])
            self.remove_storage([1,1,1,1,0])
            return True
        else :
            return False
    
    def disassemble_white(self):
        if self.storage[4] >= 1:
            self.storage[4] -= 1
            self.add_storage([1,1,1,1,0])

    
class Storage(Cell) :
    def __init__(self,*args):
        super().__init__(*args)
        self.maxes[0] = 100
        self.maxes[1] = 20
        self.maxes[2] = 20
        self.maxes[3] = 20
        self.maxes[4] = 20


class Leaf(Cell) :
    def __init__(self, color, *args):
        super().__init__(*args)
        self.color:str = color

    def create_energy(self):
        self.add_storage([1,0,0,0,0])


class Detector(Cell) :
    pass


class Eater(Cell) :
    pass


class Cluster :
    def __init__(self,stage_count,gene,ClusterID):
        """
        info1~info16,energy,red,green,blue,white
        """
        self.ClusterID:int = ClusterID
        self.stage_count:int = stage_count
        self.gene:str = gene
        self.stages:list = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for k in range(stage_count)]
        #stage内の物量が多いほどenergy多く

    def info_stage(self, stage_num):
        return self.stages[stage_num]
    
    def next_stage(self):
        for k in range(self.stage_count-1):
            self.stages[k+1] = self.stages[k]
        self.stages[0] = self.stages[self.stage_count]

    def remove_stage(self,stage,type,amount):
        if self.stages[stage][type] < amount :
            pass
        else :
            self.stages[stage][type] -= amount

    def add_stage(self,stage,type,amount) :
        self.stages[stage][type] += amount


ClusterIDs: list[Cluster] = []


cell = Storage(10,2,2,2,0,[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],np.array([0,0,0]),0,0)
cluster = Cluster(1,"aaa",0)
ClusterIDs = [cluster]


print(cell.info())