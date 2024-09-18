import numpy as np


base_energy_max = 10
base_red_max = 2
base_green_max = 2
base_blue_max = 2
base_white_max = 2


class Cell :
    def __init__(self,energy,red,green,blue,detect_list,neural):
        self.energy:float = energy
        self.red:int = red
        self.green:int = green
        self.blue:int = blue
        self.detect_list:list[str] = detect_list
        self.neural:np.array = neural
    
    def detector(self,object_list,detect_list):
        return [detect_list[k] in object_list for k in range(detect_list)]

    def neural_net(self,inputs:np.array):
        return np.dot(inputs,self.neural)

    def add_storage(self,inputs):
        """
        energy,red,green,blue
        """
        output = [0,0,0,0]
        if base_energy_max <= self.energy + inputs[0]:
            self.energy = base_energy_max
            output[0] = self.energy + inputs[0] - base_energy_max
        else :
            self.energy += inputs[0]
        if base_red_max <= self.red + inputs[1]:
            self.red = base_red_max
            output[1] = self.red + inputs[1] - base_red_max
        else :
            self.red += inputs[1]
        if base_green_max <= self.green + inputs[2]:
            self.green = base_green_max
            output[2] = self.green + inputs[2] - base_green_max
        else :
            self.green += inputs[2]
        if base_blue_max <= self.blue + inputs[3]:
            self.blue = base_blue_max
            output[3] = self.blue + inputs[3] - base_blue_max
        else :
            self.blue += inputs[3]
        return output
    
    def remove_storage(self,inputs):
        """
        energy,red,green,blue
        """
        output = [0,0,0,0]
        if self.energy >= inputs[0]:
            self.energy -= inputs[0]
            output[0] = inputs[0]
        if self.red >= inputs[1]:
            self.red -= inputs[1]
            output[1] = inputs[1]
        if self.green >= inputs[2]:
            self.green -= inputs[2]
            output[2] = inputs[2]
        if self.blue >= inputs[3]:
            self.blue -= inputs[3]
            output[3] = inputs[3]
        return output


class Factory(Cell) :
    def __init__(self,white) :
        self.white:int = white
        super().__init__()
    
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
        if base_energy_max <= self.energy + inputs[0]:
            self.energy = base_energy_max
            output[0] = self.energy + inputs[0] - base_energy_max
        else :
            self.energy += inputs[0]
        if base_red_max <= self.red + inputs[1]:
            self.red = base_red_max
            output[1] = self.red + inputs[1] - base_red_max
        else :
            self.red += inputs[1]
        if base_green_max <= self.green + inputs[2]:
            self.green = base_green_max
            output[2] = self.green + inputs[2] - base_green_max
        else :
            self.green += inputs[2]
        if base_blue_max <= self.blue + inputs[3]:
            self.blue = base_blue_max
            output[3] = self.blue + inputs[3] - base_blue_max
        else :
            self.blue += inputs[3]
        if base_white_max <= self.white + inputs[4]:
            self.white = base_white_max
            output[4] = self.white + inputs[4] - base_white_max
        else :
            self.white += inputs[4]
        return output

    def remove_storage(self,inputs):
        """
        energy,red,green,blue,white
        """
        output = [0,0,0,0,0]
        if self.energy >= inputs[0]:
            self.energy -= inputs[0]
            output[0] = inputs[0]
        if self.red >= inputs[1]:
            self.red -= inputs[1]
            output[1] = inputs[1]
        if self.green >= inputs[2]:
            self.green -= inputs[2]
            output[2] = inputs[2]
        if self.blue >= inputs[3]:
            self.blue -= inputs[3]
            output[3] = inputs[3]
        if self.white >= inputs[4]:
            self.white -= inputs[4]
            output[4] = inputs[4]
        return output


class Cluster :
    def __init__(self,stage_count,gene):
        self.stage_count:int = stage_count
        self.gene:str = gene