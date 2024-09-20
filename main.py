import numpy as np



self.white_max = 2


class Cell :
    def __init__(self,energy,red,green,blue,detect_list,neural):
        self.energy:float = energy
        self.red:int = red
        self.green:int = green
        self.blue:int = blue
        self.detect_list:list[str] = detect_list
        self.neural:np.array = neural
        self.outputs:list[float,int,int, int] = [0.0,0,0,0]
        self.energy_max = 10
        self.red_max = 10
        self.green_max = 10
        self.blue_max = 10
    
    def detector(self,object_list,detect_list):
        return [detect_list[k] in object_list for k in range(detect_list)]

    def neural_net(self,inputs:np.array):
        return np.dot(inputs,self.neural)
    
    def info(self):
        return f"energy : {self.energy}, red : {self.red}, green : {self.green}, blue : {self.blue}"

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
        self.outputs = [self.outputs[0]+output[0], self.outputs[1]+output[1], self.outputs[2]+output[2], self.outputs[3]+output[3] ]


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


class Cluster :
    def __init__(self,stage_count,gene):
        self.stage_count:int = stage_count
        self.gene:str = gene


