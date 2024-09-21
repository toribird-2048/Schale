import numpy as np


class Cell :
    def __init__(self,energy,red,green,blue,detect_max_list,detect_min_list,neural,stage,ClusterID):
        """
        Detect_list : In Stage -> Info1~Info16,energy,red,green,blue and In Cell -> energy,red,green,blue
        """
        self.energy:float = energy
        self.red:int = red
        self.green:int = green
        self.blue:int = blue
        self.detect_max_list:list = detect_max_list # Length must be 21 
        self.detect_min_list:list = detect_min_list # Length must be 21 
        self.neural:np.array = neural
        self.outputs:list[float,int,int, int] = [0.0,0,0,0]
        self.energy_max:float = 10
        self.red_max:int = 10
        self.green_max:int = 10
        self.blue_max:int = 10
        self.stage:int = stage
        self.ClusterID:int = ClusterID
        self.input_queue:list[float,int,int,int] = [0.0,0,0]
    
    def detector(self):
        detecteds = []
        for k in range(21):
            if ClusterIDs[self.ClusterID].info_stage(self.stage)[k] <= self.detect_min_list[k] :
                detecteds.append(-1)
            elif self.detect_max_list[k] <= ClusterIDs[self.ClusterID].info_stage(self.stage)[k] :
                detecteds.append(1)
            else :
                detecteds.append((ClusterIDs[self.ClusterID].info_stage(self.stage)[k] - self.detect_min_list[k]) / (self.detect_max_list[k] - self.detect_min_list[k]) - 0.5)
        detecteds.append(2*self.energy/self.energy_max-0.5)
        detecteds.append(2*self.red/self.red_max-0.5)
        detecteds.append(2*self.green/self.green_max-0.5)
        detecteds.append(2*self.blue/self.blue_max-0.5)
        return detecteds

    def neural_net(self):
        output = np.dot(self.neural,np.array(self.detector()))
        output_list = output.T
        adds = output_list[:4]
        removes = output_list[4:]
        for k in range(len(adds)):
            self.input_queue[k] += adds[k]
        pass
        #TODO: ClusterIDsの中のClusterのstageから必要数を引いてadd_storageするところ

    def info(self):
        return f"energy : {self.energy}, red : {self.red}, green : {self.green}, blue : {self.blue}"
    
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
        output = [0,0,0,0]
        if self.energy_max <= self.energy + inputs[0]:
            self.energy = self.energy_max
            output[0] = self.energy + inputs[0] - self.energy_max
        else :
            self.energy += inputs[0]
        if self.red_max <= self.red + inputs[1]:
            self.red = self.red_max
            output[1] = self.red + inputs[1] - self.red_max
        else :
            self.red += inputs[1]
        if self.green_max <= self.green + inputs[2]:
            self.green = self.green_max
            output[2] = self.green + inputs[2] - self.green_max
        else :
            self.green += inputs[2]
        if self.blue_max <= self.blue + inputs[3]:
            self.blue = self.blue_max
            output[3] = self.blue + inputs[3] - self.blue_max
        else :
            self.blue += inputs[3]
        self.outputs = [self.outputs[0]+output[0], self.outputs[1]+output[1], self.outputs[2]+output[2], self.outputs[3]+output[3] ]
    
    def remove_storage(self,inputs):
        """
        energy,red,green,blue
        """
        output = [0,0,0,0]
        if self.energy >= inputs[0]:
            self.energy -= inputs[0]
            output[0] = inputs[0]
        else :
            return False
        if self.red >= inputs[1]:
            self.red -= inputs[1]
            output[1] = inputs[1]
        else :
            return False
        if self.green >= inputs[2]:
            self.green -= inputs[2]
            output[2] = inputs[2]
        else :
            return False
        if self.blue >= inputs[3]:
            self.blue -= inputs[3]
            output[3] = inputs[3]
        else :
            return False
        self.outputs = [self.outputs[0]+output[0], self.outputs[1]+output[1], self.outputs[2]+output[2], self.outputs[3]+output[3] ]


class InOut(Cell) :
    pass


class Factory(Cell) :
    def __init__(self,white) :
        self.outputs:list[float,int,int, int, int] = [0,0,0,0,0]
        self.white:int = white
        super().__init__()
        self.white_max = 2
    
    def create_chem(self):
        if self.energy >= 1 and self.red >= 1 and self.green >= 1 and self.blue >= 1:
            self.energy -= 1
            self.red -= 1
            self.green -= 1
            self.blue -= 1
            self.add_storage([0,0,0,0,1])
            self.remove_storage([1,1,1,1,0])
            return True
        else :
            return False
        
    def add_storage(self,inputs):
        """
        energy,red,green,blue,white
        """
        output = [0,0,0,0,0]
        if self.energy_max <= self.energy + inputs[0]:
            self.energy = self.energy_max
            output[0] = self.energy + inputs[0] - self.energy_max
        else :
            self.energy += inputs[0]
        if self.red_max <= self.red + inputs[1]:
            self.red = self.red_max
            output[1] = self.red + inputs[1] - self.red_max
        else :
            self.red += inputs[1]
        if self.green_max <= self.green + inputs[2]:
            self.green = self.green_max
            output[2] = self.green + inputs[2] - self.green_max
        else :
            self.green += inputs[2]
        if self.blue_max <= self.blue + inputs[3]:
            self.blue = self.blue_max
            output[3] = self.blue + inputs[3] - self.blue_max
        else :
            self.blue += inputs[3]
        if self.white_max <= self.white + inputs[4]:
            self.white = self.white_max
            output[4] = self.white + inputs[4] - self.white_max
        else :
            self.white += inputs[4]
        self.outputs = [self.outputs[0]+output[0], self.outputs[1]+output[1], self.outputs[2]+output[2], self.outputs[3]+output[3], self.outputs[4]+output[4] ]

    def remove_storage(self,inputs):
        """
        energy,red,green,blue,white
        """
        output = [0,0,0,0,0]
        if self.energy >= inputs[0]:
            self.energy -= inputs[0]
            output[0] = inputs[0]
        else :
            raise ValueError
        if self.red >= inputs[1]:
            self.red -= inputs[1]
            output[1] = inputs[1]
        else :
            raise ValueError
        if self.green >= inputs[2]:
            self.green -= inputs[2]
            output[2] = inputs[2]
        else :
            raise ValueError
        if self.blue >= inputs[3]:
            self.blue -= inputs[3]
            output[3] = inputs[3]
        else :
            raise ValueError
        if self.white >= inputs[4]:
            self.white -= inputs[4]
            output[4] = inputs[4]
        else :
            raise ValueError
        self.outputs = [self.outputs[0]+output[0], self.outputs[1]+output[1], self.outputs[2]+output[2], self.outputs[3]+output[3], self.outputs[4]+output[4] ]
    
    def create_white(self):
        if self.energy >= 1 and self.red >= 1 and self.green >= 1 and self.blue >= 1:
            self.white += 1
            self.energy -= 1
            self.red -= 1
            self.green -= 1
            self.blue -= 1
            self.add_storage([0,0,0,0,1])

    
    def disassemble_white(self):
        if self.white >= 1:
            self.white -= 1
            self.add_storage([1,1,1,1,0])

    
class Storage(Cell) :
    def __init__(self):
        super().__init__()
        self.energy_max = 100
        self.red_max = 20
        self.green_max = 20
        self.blue_max = 20
        self.white_max = 20

    def add_storage(self,inputs):
        """
        energy,red,green,blue,white
        """
        output = [0,0,0,0,0]
        if self.energy_max <= self.energy + inputs[0]:
            self.energy = self.energy_max
            output[0] = self.energy + inputs[0] - self.energy_max
        else :
            self.energy += inputs[0]
        if self.red_max <= self.red + inputs[1]:
            self.red = self.red_max
            output[1] = self.red + inputs[1] - self.red_max
        else :
            self.red += inputs[1]
        if self.green_max <= self.green + inputs[2]:
            self.green = self.green_max
            output[2] = self.green + inputs[2] - self.green_max
        else :
            self.green += inputs[2]
        if self.blue_max <= self.blue + inputs[3]:
            self.blue = self.blue_max
            output[3] = self.blue + inputs[3] - self.blue_max
        else :
            self.blue += inputs[3]
        if self.white_max <= self.white + inputs[4]:
            self.white = self.white_max
            output[4] = self.white + inputs[4] - self.white_max
        else :
            self.white += inputs[4]
        self.outputs = [self.outputs[0]+output[0], self.outputs[1]+output[1], self.outputs[2]+output[2], self.outputs[3]+output[3], self.outputs[4]+output[4] ]

    def remove_storage(self,inputs):
        """
        energy,red,green,blue,white
        """
        output = [0,0,0,0,0]
        if self.energy >= inputs[0]:
            self.energy -= inputs[0]
            output[0] = inputs[0]
        else :
            raise ValueError
        if self.red >= inputs[1]:
            self.red -= inputs[1]
            output[1] = inputs[1]
        else :
            raise ValueError
        if self.green >= inputs[2]:
            self.green -= inputs[2]
            output[2] = inputs[2]
        else :
            raise ValueError
        if self.blue >= inputs[3]:
            self.blue -= inputs[3]
            output[3] = inputs[3]
        else :
            raise ValueError
        if self.white >= inputs[4]:
            self.white -= inputs[4]
            output[4] = inputs[4]
        else :
            raise ValueError
        self.outputs = [self.outputs[0]+output[0], self.outputs[1]+output[1], self.outputs[2]+output[2], self.outputs[3]+output[3], self.outputs[4]+output[4] ]


class Leaf(Cell) :
    def __init__(self, color):
        super().__init__()

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


cell = Cell(10,0,0,0,[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],np.array(10),0,0)
cluster = Cluster(1,"aaa",0)
ClusterIDs = [cluster]

print(cell.detector())
print(cell.info())