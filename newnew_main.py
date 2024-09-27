import numpy as np
import random
import copy
from dataclasses import dataclass

mutation_rate = 0.1

class Field:
    def __init__(self,height,width):
        self.height = height
        self.width = width
        self.cluster_blocks:list[list[Field.Cluster]] = [[None for _ in range(width)] for _ in range(height)]
        self.material_blocks = [[[0 for _ in range(5)] for _ in range(width)] for _ in range(height)]

    class Cluster :
        def __init__(self,field,x,y,stage_count,gene,detect_max_list,detect_min_list,neural_output_maxes,neural,ClusterID):
            """
            info1~info16,energy,red,green,blue,white
            """
            self.field:Field = field
            self.x = x
            self.y = y
            self.ClusterID:int = ClusterID
            self.stage_count:int = stage_count
            self.gene:list = gene
            self.stages:list = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for k in range(stage_count)]
            self.cell_stages:list[list[Field.Cluster.Cell]] = [[] for k in range(stage_count)]
            self.detect_max_list:list = detect_max_list # length must be 16
            self.detect_min_list:list = detect_min_list # same as detect_max_list
            self.neural:np.array = neural #横21縦16
            self.neural_output_maxes = neural_output_maxes
            #self.create_cell([7,0,0,0,0,0,0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            #self.add_cell_stage(0,self.ClusterDupricator(self,0,[0,0,0,0,0],[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],]),0))

            #stage内の物量が多いほどenergy多く


        class Cell : #stage,energy,red,green,blue,white(material_list),energy,red,green,blue,white(detect_max_list),energy,red,green,blue,white(detect_min_list),energy,red,green,blue,white(neural_add_maxes),energy,red,green,blue,white(n),b00,b01...b025,b10...b125...b225......b2525(neural)
            def __init__(self,cluster,stage,materials_list,detect_max_list,detect_min_list,neural_add_maxes,neural_remove_maxes,neural,cell_ID):
                """
                Detect_list : In Stage -> Info1~Info16,energy,red,green,blue | In Cell -> energy,red,green,blue
                """
                self.cluster:Field.Cluster = cluster # 所属するクラスター
                self.storage:list[int] = materials_list
                self.detect_max_list:list = detect_max_list # Length must be 21 (ステージ内物質の検知情報)
                self.detect_min_list:list = detect_min_list # Length must be 21 ()
                self.neural:np.array = neural #横26縦26
                self.outputs:list[int] = [0,0,0,0,0]
                self.maxes:list[int] = [10,2,2,2,0]
                self.stage:int = stage
                self.input_queue:list[int] = [0,0,0,0,0]
                self.neural_add_maxes:list[int] = neural_add_maxes #意味的にはinputだけどinputだとニューラルネットワークの入力と意味が競合しちゃうからaddにしてある
                self.neural_remove_maxes:list[int] = neural_remove_maxes #こっちも意味はoutput
                self.cell_ID:int = cell_ID

            def __post_init__(self):
                for k in range(len(self.storage)) :
                    if self.maxes[k] <= self.storage[k] :
                        self.storage[k] = self.maxes[k]

            def detector(self):
                detecteds = []
                #print(len(self.detect_min_list))
                for k in range(21):
                    if self.cluster.info_stage(self.stage)[k] <= self.detect_min_list[k] :
                        detecteds.append(-1)
                    elif self.detect_max_list[k] <= self.cluster.info_stage(self.stage)[k] :
                        detecteds.append(1)
                    else :
                        detecteds.append((self.cluster.info_stage(self.stage)[k] - self.detect_min_list[k]) / (self.detect_max_list[k] - self.detect_min_list[k]) - 0.5)
                for k in range(5):
                    if self.maxes[k] == 0 :
                        detecteds.append(-1)
                    else:
                        detecteds.append(2*self.storage[k]/self.maxes[k]-0.5)
                return detecteds

            def neural_net(self):
                output = np.dot(self.neural,np.array(self.detector()).T)
                output_list = output.T
                output_list = [(output_list[k]+1)*0.5 for k in range(len(output_list))]
                #print(f"output{output_list}")
                adds = output_list[:5]
                removes = output_list[5:10]
                infos = output_list[10:]
                for k in range(len(adds)):
                    self.input_queue[k] += adds[k]
                    #print(adds[k] * self.neural_add_maxes[k])
                self.input_queue = [round(adds[k] * self.neural_add_maxes[k]) for k in range(5)]
                self.input()
                self.outputs = [round(removes[k] * self.neural_remove_maxes[k]) for k in range(5)]
                self.output()
                self.infos_output([round(infos[k]) for k in range(16)])

            def info(self):
                return f"energy : {self.storage[0]}, red : {self.storage[1]}, green : {self.storage[2]}, blue : {self.storage[3]}, white : {self.storage[4]}, ClusterID : {self.cluster.ClusterID}, stage : {self.stage}"

            def infos_output(self,infos):
                for k in range(16) :
                    self.cluster.add_stage(self.stage,k,infos[k])

            def input(self):
                """
                type-> 0:energy, 1:red, 2:green, 3:blue, 4:white
                """
                for k in range(5):
                    self.input_queue[k] = self.cluster.remove_stage(self.stage,16+k,self.input_queue[k])
                #print(self.input_queue)
                self.add_storage(self.input_queue)
                self.input_queue = [0,0,0,0,0]

            def output(self):
                """
                type-> 0:energy, 1:red, 2:green, 3:blue, 4:white
                """
                output = self.remove_storage(self.outputs)
                for k in range(5):
                    self.cluster.add_stage(self.stage,16+k,output[k])

            def output_infos(self,type) :
                self.cluster.add_stage(self.stage,type,1)

            def consume_energy(self,amount):
                if self.storage[0] >= amount:
                    self.storage[0] -= amount
                else:
                    self.storage[0] = 0

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
                return output

            def remove_storage(self,inputs):
                """
                energy,red,green,blue
                """
                output = [0,0,0,0,0]
                if self.storage[0] >= inputs[0]:
                    self.storage[0] -= inputs[0]
                    output[0] = inputs[0]
                if self.storage[1] >= inputs[1]:
                    self.storage[1] -= inputs[1]
                    output[1] = inputs[1]
                if self.storage[2] >= inputs[2]:
                    self.storage[2] -= inputs[2]
                    output[2] = inputs[2]
                if self.storage[3] >= inputs[3]:
                    self.storage[3] -= inputs[3]
                    output[3] = inputs[3]
                if self.storage[4] >= inputs[4]:
                    self.storage[4] -= inputs[4]
                    output[4] = inputs[4]
                self.outputs = [self.outputs[0]+output[0], self.outputs[1]+output[1], self.outputs[2]+output[2], self.outputs[3]+output[3], self.outputs[4]+output[4] ]
                return output

            def print_info(self):
                print(f"{self.cell_ID} :: {self.info()}")

            def cycle(self):
                self.print_info()
                self.neural_net()


        class InOut(Cell) :
            def __init__(self,*args) :
                super().__init__(*args)
                self.maxes[4] = 2
                self.neural_add_maxes = [2,2,2,2,2]

            def output_field(self,outputs):
                outputs = self.remove_storage(outputs)
                self.cluster.field.add_material(self.cluster.x,self.cluster.y,outputs)

            def input_field(self,inputs):
                inputs = self.cluster.field.remove_material(self.cluster.x,self.cluster.y,inputs)
                self.add_storage(inputs)

            def neural_net(self):#26*36
                output = np.dot(self.neural,np.array(self.detector()).T)
                output_list = output.T
                #print(f"output{output_list}")
                adds = output_list[:5]
                removes = output_list[5:10]
                infos = output_list[10:26]
                inputs = output_list[26:31]
                outputs = output_list[31:36]
                for k in range(len(adds)):
                    self.input_queue[k] += adds[k]
                    #print(adds[k] * self.neural_add_maxes[k])
                self.input_queue = [round(adds[k] * self.neural_add_maxes[k]) for k in range(5)]
                self.input()
                self.outputs =[round(removes[k] * self.neural_remove_maxes[k]) for k in range(5)]
                self.output()
                self.infos_output([round(infos[k]) for k in range(16)])
                self.input_field([round(inputs[k] * self.neural_add_maxes[k]) for k in range(5)])
                self.output_field([round(outputs[k] * self.neural_remove_maxes[k]) for k in range(5)])


        class Factory(Cell) :
            def __init__(self,*args) :
                super().__init__(*args)
                self.maxes[4] = 2
                self.neural_add_maxes = [2,2,2,2,2]

            def create_white(self,count):
                if self.storage[0] >= count and self.storage[1] >= count and self.storage[2] >= count and self.storage[3] >= count:
                    self.add_storage([0,0,0,0,1])
                    self.remove_storage([1,1,1,1,0])
                    return True
                else :
                    return False

            def disassemble_white(self,count):
                if self.storage[4] >= count:
                    self.storage[4] -= count
                    self.add_storage([count,count,count,count,0])

            def neural_net(self):
                output = np.dot(self.neural,np.array(self.detector()).T)
                output_list = output.T
                #print(f"output{output_list}")
                adds = output_list[:5]
                removes = output_list[5:10]
                infos = output_list[10:26]
                whites = output_list[26:]
                for k in range(len(adds)):
                    self.input_queue[k] += adds[k]
                    #print(adds[k] * self.neural_add_maxes[k])
                self.input_queue = [round(adds[k] * self.neural_add_maxes[k]) for k in range(5)]
                self.input()
                self.outputs = [round(removes[k] * self.neural_remove_maxes[k]) for k in range(5)]
                self.output()
                self.infos_output([round(infos[k]) for k in range(16)])
                self.create_white(round(whites[0]))
                self.disassemble_white(round(whites[1]))


        class Storage(Cell) :
            def __init__(self,*args):
                super().__init__(*args)
                self.maxes[0] = 100
                self.maxes[1] = 20
                self.maxes[2] = 20
                self.maxes[3] = 20
                self.maxes[4] = 20
                self.neural_add_maxes = [20,20,20,20,20]


        class Leaf(Cell) :
            def __init__(self, *args):
                super().__init__(*args)
                self.maxes[0] = 100

            def create_energy(self):
                self.add_storage([10,0,0,0,0])

            def cycle(self):
                self.print_info()
                self.create_energy()
                self.neural_net()


        class Detector(Cell) :
            def __init__(self,*args) :
                super().__init__(*args) #neural : 31*26

            def detector(self):
                detecteds = [] #31行
                for k in range(21):
                    if self.cluster.info_stage(self.stage)[k] <= self.detect_min_list[k] :
                        detecteds.append(-1)
                    elif self.detect_max_list[k] <= self.cluster.info_stage(self.stage)[k] :
                        detecteds.append(1)
                    else :
                        detecteds.append((self.cluster.info_stage(self.stage)[k] - self.detect_min_list[k]) / (self.detect_max_list[k] - self.detect_min_list[k]) - 0.5)
                for k in range(5):
                    if self.maxes[k] == 0 :
                        detecteds.append(-1)
                    else:
                        detecteds.append(2*self.storage[k]/self.maxes[k]-0.5)
                for k in range(5):
                    if self.cluster.field.material_blocks[self.cluster.x][self.cluster.y][k] <= self.detect_min_list[16+k] :
                        detecteds.append(-1)
                    elif self.detect_min_list[16+k] <= self.cluster.field.material_blocks[self.cluster.x][self.cluster.y][k] :
                        detecteds.append(1)
                    else :
                        detecteds.append((self.cluster.field.material_blocks[self.cluster.x][self.cluster.y][k] - self.detect_min_list[k]) / (self.detect_max_list[k] - self.detect_min_list[k]) - 0.5)

                return detecteds


        class Eater(Cell) :
            def __init__(self,*args) :
                super().__init__(*args)
                self.maxes[0] = 200

            def eat(self,dest):
                if dest == 1 :
                    if self.cluster.field.cluster_blocks[self.cluster.x + 1][self.cluster.y] is not None:
                        self.consume_energy(100)
                        self.cluster.field.remove_cluster(self.cluster.x + 1, self.cluster.y)
                if dest == 2 :
                    if self.cluster.field.cluster_blocks[self.cluster.x - 1][self.cluster.y] is not None:
                        self.consume_energy(100)
                        self.cluster.field.remove_cluster(self.cluster.x - 1, self.cluster.y)
                if dest == 3 :
                    if self.cluster.field.cluster_blocks[self.cluster.x][self.cluster.y + 1] is not None:
                        self.consume_energy(100)
                        self.cluster.field.remove_cluster(self.cluster.x, self.cluster.y + 1)
                if dest == 4 :
                    if self.cluster.field.cluster_blocks[self.cluster.x][self.cluster.y - 1] is not None:
                        self.consume_energy(100)
                        self.cluster.field.remove_cluster(self.cluster.x, self.cluster.y - 1)

            def neural_net(self): #26*28
                output = np.dot(self.neural,np.array(self.detector()).T)
                output_list = output.T
                output_list = [(output_list[k]+1)*0.5 for k in range(len(output_list))]
                #print(f"output{output_list}")
                adds = output_list[:5]
                removes = output_list[5:10]
                infos = output_list[10:26]
                eat = output_list[26:28]
                for k in range(len(adds)):
                    self.input_queue[k] += adds[k]
                    #print(adds[k] * self.neural_add_maxes[k])
                self.input_queue = [round(adds[k] * self.neural_add_maxes[k]) for k in range(5)]
                self.input()
                self.outputs = [round(removes[k] * self.neural_remove_maxes[k]) for k in range(5)]
                self.output()
                self.infos_output([round(infos[k]) for k in range(16)])
                if 0.5 < eat[0] :
                    self.eat(np.ceil(eat[1]*4))



        class ClusterDupricator(Cell) :
            def __init__(self, *args):
                super().__init__(*args)
                self.maxes[0] = 1000

            def create_cluster(self,dest):
                pass

            def neural_net(self):
                output = np.dot(self.neural,np.array(self.detector()).T)
                output_list = output.T
                output_list = [(output_list[k]+1)*0.5 for k in range(len(output_list))]
                #print(f"output{output_list}")
                adds = output_list[:5]
                removes = output_list[5:10]
                infos = output_list[10:26]
                duprication = output_list[26]
                for k in range(len(adds)):
                    self.input_queue[k] += adds[k]
                    #print(adds[k] * self.neural_add_maxes[k])
                self.input_queue = [round(adds[k] * self.neural_add_maxes[k]) for k in range(5)]
                self.input()
                self.outputs = [round(removes[k] * self.neural_remove_maxes[k]) for k in range(5)]
                self.output()
                self.infos_output([round(infos[k]) for k in range(16)])
                self.create_cluster(np.ceil(duprication * 4))


            def decode_genes(self,genes:list):
                #print(f"genes{genes}")
                stage_count = genes[0]
                materials_list = genes[0:5]
                detect_max_list = genes[5:26]
                detect_min_list = genes[26:47]
                neural_output_maxes = genes[47:64]
                #print(f"{stage_count}")
                #print(f"{materials_list}")
                #print(f"{detect_max_list}")
                #print(f"{detect_min_list}")
                #print(f"{neural_output_maxes}")
                neural = []
                for k in range(16):
                    neural.append([])
                    for l in range(21):
                        neural[k].append(genes[64+k*21+l])
                #print(f"{neural}")
                cells = genes[400:]
                return stage_count, materials_list, detect_max_list, detect_min_list, neural_output_maxes, neural, cells

            def create_cluster(self,dest):
                genes = self.cluster.gene
                if dest != 0 :
                    if dest == 1 :
                        x = self.cluster.x + 1
                        y = self.cluster.y
                    elif dest == 2 :
                        x = self.cluster.x - 1
                        y = self.cluster.y
                    elif dest == 3 :
                        x = self.cluster.x
                        y = self.cluster.y + 1
                    elif dest == 4 :
                        x = self.cluster.x
                        y = self.cluster.y - 1
                    stage_count, materials_list, detect_max_list, detect_min_list, neural_output_maxes, neural, cells = self.decode_genes(genes)
                    self.cluster.field.add_cluster(x, y, Field.Cluster(self,x,y,stage_count,genes,detect_max_list, detect_min_list, neural_output_maxes,neural,len(cells)))
                    for k in range(len(materials_list)) :
                        self.cluster.field.cluster_blocks[x][y].add_stage(0,k,materials_list[k])
                    #print(f"create_cluster{cells}")
                    for cell in cells:
                        self.cluster.field.cluster_blocks[x][y].create_cell(cell)
                    print("cluster breeded!")


        def decode_gene(self,gene_list:list):
            #print(gene_list)
            if len(gene_list) != 994 :
                print("Invalid gene length!")
            cell_type = gene_list[0]
            stage = gene_list[1]
            material_list = gene_list[1:6]
            detect_max_list = gene_list[6:27]
            detect_min_list = gene_list[27:48]
            neural_add_maxes = gene_list[48:53]
            neural_remove_maxes = gene_list[53:58]
            neural = []
            if cell_type == 0 or cell_type == 3 or cell_type == 4:
                for k in range(26):
                    neural.append([])
                    for l in range(26):
                        neural[k].append(gene_list[58+k*26+l])
            elif cell_type == 1:
                for k in range(36):
                    neural.append([])
                    for l in range(26):
                        print(f"                    58+k*26+l:{58+k*26+l}")
                        neural[k].append(gene_list[58+k*26+l])
            elif cell_type == 2 or cell_type == 6:
                for k in range(28):
                    neural.append([])
                    for l in range(26):
                        print(f"                    58+k*26+l:{58+k*26+l}")
                        neural[k].append(gene_list[58+k*26+l])
            elif cell_type == 5:
                for k in range(26):
                    neural.append([])
                    for l in range(31):
                        neural[k].append(gene_list[58+k*31+l])
            elif cell_type == 7:
                for k in range(27):
                    neural.append([])
                    for l in range(26):
                        neural[k].append(gene_list[58+k*26+l])
            return cell_type, stage, material_list, detect_max_list, detect_min_list, neural_add_maxes, neural_remove_maxes, neural

        def create_cell(self, gene_list:list):
            print(f"                create_cell:{gene_list}")
            print(f"                len:{len(gene_list)}")
            gene = self.decode_gene(gene_list)
            if gene[0] == 0:
                self.add_cell_stage(gene[1],self.Cell(self,gene[1],gene[2],gene[3],gene[4],gene[5],gene[6],gene[7],0))
            if gene[0] == 1:
                self.add_cell_stage(gene[1],self.InOut(self,gene[1],gene[2],gene[3],gene[4],gene[5],gene[6],gene[7],1))
            if gene[0] == 2:
                self.add_cell_stage(gene[1],self.Factory(self,gene[1],gene[2],gene[3],gene[4],gene[5],gene[6],gene[7],2))
            if gene[0] == 3:
                self.add_cell_stage(gene[1],self.Storage(self,gene[1],gene[2],gene[3],gene[4],gene[5],gene[6],gene[7],3))
            if gene[0] == 4:
                self.add_cell_stage(gene[1],self.Leaf(self,gene[1],gene[2],gene[3],gene[4],gene[5],gene[6],gene[7],4))
            if gene[0] == 5:
                self.add_cell_stage(gene[1],self.Detector(self,gene[1],gene[2],gene[3],gene[4],gene[5],gene[6],gene[7],5))
            if gene[0] == 6:
                self.add_cell_stage(gene[1],self.Eater(self,gene[1],gene[2],gene[3],gene[4],gene[5],gene[6],gene[7],6))
            if gene[0] == 7:
                self.add_cell_stage(gene[1],self.ClusterDupricator(self,gene[1],gene[2],gene[3],gene[4],gene[5],gene[6],gene[7],7))

        def info_stage(self, stage_num):
            return self.stages[stage_num-1]

        def print_stages_info(self):
            for k in range(self.stage_count):
                print(f"stage{k}:{self.stages[k]}")
                print(f"{k}Cell:{len(self.cell_stages[k])}")

        def stage_zero_clear_infos(self):
            for k in range(16):
                self.stages[0][k] = 0

        def old_next_stage(self):
            temp_stages = []
            temp_stages.append(self.stages[self.stage_count-1])
            for k in range(self.stage_count-1):
                temp_stages.append(self.stages[k])
            self.stages = copy.deepcopy(temp_stages)
        def next_stage(self):
            temp_stages = copy.deepcopy(self.stages)
            for k in range(self.stage_count-1):
                self.stages[k+1] = copy.deepcopy(temp_stages[k])
            self.stages[0] = temp_stages[-1]

        def remove_stage(self,stage,type,amount):
            if self.stages[stage][type] < amount :
                output = 0 #error処理
            else :
                self.stages[stage][type] -= amount
                output = amount
            return output

        def add_stage(self,stage,type,amount) :
            self.stages[stage][type] += amount
            return amount


        def add_cell_stage(self, stage, cell:Cell):
            print(f"correct_stage:{self.stage_count}, stage:{stage}")
            self.cell_stages[stage].append(cell)

        def cycle(self):
            print("================================================================")
            for stage in range(self.stage_count):
                for cell in self.cell_stages[stage]:
                    cell.cycle()
            self.print_stages_info()
            self.neural_net()
            self.next_stage()

        def detector(self):
            detecteds = []
            for k in range(21):
                if self.info_stage(0)[k] <= self.detect_min_list[k] :
                    detecteds.append(-1)
                elif self.detect_max_list[k] <= self.info_stage(0)[k] :
                    detecteds.append(1)
                else :
                    detecteds.append((self.info_stage(0)[k] - self.detect_min_list[k]) / (self.detect_max_list[k] - self.detect_min_list[k]) - 0.5)
            return detecteds

        def neural_net(self):
            output = np.dot(self.neural,np.array(self.detector()).T) #21*16
            output_list = output.T
            output_list = [(output_list[k]+1)*0.5 for k in range(len(output_list))]
            self.stage_zero_clear_infos()
            #print(f"CLoutput{output_list}")
            for k in range(16) :
                self.add_stage(0,k,round(output_list[k]*self.neural_output_maxes[k]))

    def add_cluster(self,x,y,cluster: Cluster):
        self.cluster_blocks[x][y] = cluster

    def remove_cluster(self,x, y):
        self.cluster_blocks[x][y] = None

    def add_material(self,x,y,inputs):
        for k in range(5):
            self.material_blocks[x][y][k] = inputs[k]

    def remove_material(self,x, y, outputs):
        return_outputs = [0,0,0,0,0]
        for k in range(5):
            if self.material_blocks[x][y][k] < outputs[k]:
                self.material_blocks[x][y][k] = 0
                return_outputs[k] = self.material_blocks[x][y][k]
            else:
                self.material_blocks[x][y][k] -= outputs[k]
                return_outputs[k] = outputs[k]
        return return_outputs


    def cycle(self):
        for x in range(self.width):
            for y in range(self.height):
                if self.cluster_blocks[x][y] is not None:
                    print("hai! I'm cluster!")
                    self.cluster_blocks[x][y].cycle()

    def decode_genes(self,genes:list):
        #print(f"genes{genes}")
        stage_count = genes[0]
        materials_list = genes[0:5]
        detect_max_list = genes[5:26]
        detect_min_list = genes[26:47]
        neural_output_maxes = genes[47:64]
        #print(f"{stage_count}")
        #print(f"{materials_list}")
        #print(f"{detect_max_list}")
        #print(f"{detect_min_list}")
        #print(f"{neural_output_maxes}")
        neural = []
        for k in range(16):
            neural.append([])
            for l in range(21):
                neural[k].append(genes[64+k*21+l])
        #print(f"{neural}")
        cells = genes[400:]
        return stage_count, materials_list, detect_max_list, detect_min_list, neural_output_maxes, neural, cells

    def create_cluster(self,x,y,genes):
        stage_count, materials_list, detect_max_list, detect_min_list, neural_output_maxes, neural, cells = self.decode_genes(genes)
        self.add_cluster(x, y, Field.Cluster(self,x,y,stage_count,genes,detect_max_list, detect_min_list, neural_output_maxes,neural,len(cells)))
        for k in range(len(materials_list)) :
            self.cluster_blocks[x][y].add_stage(0,k,materials_list[k])
        #print(f"create_cluster{cells}")
        for cell in cells:
            self.cluster_blocks[x][y].create_cell(cell)

def mutate_genes(genes:list):
    print(f"first_genes{genes}")
    if np.random.random() <= mutation_rate:
        genes.append(genes[-1])
        return genes
    if mutation_rate < np.random.random() <= mutation_rate * 2:
        if type(genes[-2]) != int:
            del genes[-1]
    else:
        picked = random.sample(range(len(genes)),round(len(genes)*mutation_rate))
        print(f"picked_calc{len(genes)*mutation_rate}")
        print(f"picked{picked}")
        for p in picked:
            if type(genes[400]) != list:
                print("       I am not a list!")
            print(f"    p{p}")
            if p == 0:
                pass #stage_countを変えると動かなくなる危険性がある
            elif p in range(1,6):
                if np.random.random() <= 0.5:
                    genes[p] += 1
                else :
                    if genes[p] == 0:
                        genes[p] = 0
                    else:
                        genes[p] -= 1
            elif p in range(6,27):
                if np.random.random() <= 0.5:
                    genes[p] += 1
                else :
                    if genes[p] == 0:
                        genes[p] = 0
                    else:
                        genes[p] -= 1
                    if genes[p] < genes[p+21] :
                        genes[p] = genes[p+21]
            elif p in range(27,48):
                if np.random.random() <= 0.5:
                    genes[p] += 1
                else :
                    if genes[p] == 0:
                        genes[p] = 0
                    else:
                        genes[p] -= 1
                    if genes[p] > genes[p-21] :
                        genes[p] = genes[p-21]
            elif p in range(48,64):
                if np.random.random() <= 0.5:
                    genes[p] += 1
                else :
                    if genes[p] == 0:
                        genes[p] = 0
                    else:
                        genes[p] -= 1
            elif p in range(64,400):
                genes[p] = np.random.random() * 2 - 1

            else:
                cell = genes[p]
                print(f"        cell{cell}")
                print(f"        genes{genes}")
                picked_cell = random.sample(range(len(cell)),round(len(cell)*mutation_rate))
                print(f"        picked_cell_calc{len(cell)*mutation_rate}")
                print(f"        picked_cell{picked_cell}")
                for p_c in picked_cell:
                    if type(genes[400]) != list:
                        print("I am not a list!")
                    print(f"            p_c{p_c}")
                    print(f"            len(cell){len(cell)}")
                    if p_c == 0:
                        cell[p_c] = random.randint(0,7)
                    elif p_c == 1:
                        cell[p_c] = random.randrange(0,genes[0])
                    elif p_c in range(2,7):
                        if np.random.random() <= 0.5:
                            cell[p_c] += 1
                        else :
                            if cell[p_c] == 0:
                                cell[p_c] = 0
                            else:
                                cell[p_c] -= 1
                    elif p_c in range(7,28):
                        if np.random.random() <= 0.5:
                            cell[p_c] += 1
                        else :
                            if cell[p_c] == 0:
                                cell[p_c] = 0
                            else:
                                cell[p_c] -= 1
                            if cell[p_c] < cell[p_c+5] :
                                cell[p_c] = cell[p_c+5]
                    elif p_c in range(28,49):
                        if np.random.random() <= 0.5:
                            cell[p_c] += 1
                        else :
                            if cell[p_c] == 0:
                                cell[p_c] = 0
                            else:
                                cell[p_c] -= 1
                            if cell[p_c] > cell[p_c-21] :
                                cell[p_c] = cell[p_c-21]
                    elif p_c in range(49,54):
                        if np.random.random() <= 0.5:
                            cell[p_c] += 1
                        else :
                            if cell[p_c] == 0:
                                cell[p_c] = 0
                            else:
                                cell[p_c] -= 1
                    elif p_c in range(54,59):
                        if np.random.random() <= 0.5:
                            cell[p_c] += 1
                        else :
                            if cell[p_c] == 0:
                                cell[p_c] = 0
                            else:
                                cell[p_c] -= 1
                    elif p_c in range(59,962):
                        cell[p_c] = np.random.random() * 2 - 1
    return genes


field = Field(10,10)
field.create_cluster(0,0,[3,0,0,0,0,0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[0,0,0,0,0,0,0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,1,1,0,3,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[4,0,0,0,0,0,0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,1,1,1,0,3,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[7,0,0,0,0,0,0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
cluster = Field.Cluster(field,0,0,3,[],[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]),0)
#field.add_cluster(0,0,cluster)
#field.cluster_blocks[0][0].create_cell([0,0,0,0,0,0,0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,1,1,0,3,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#cell = Field.Cluster.Cell(cluster,0,[0,0,0,0,0],[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[3,1,1,1,0],[3,1,1,1,0],np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]),1)
#field.cluster_blocks[0][0].add_cell_stage(0,cell)
#field.cluster_blocks[0][0].create_cell([4,0,0,0,0,0,0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,1,1,1,0,3,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#leaf = Field.Cluster.Leaf(cluster,0,[0,0,0,0,0],[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[30,1,1,1,0],[3,1,1,1,0],np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]),2)
#field.cluster_blocks[0][0].add_cell_stage(0,leaf)
gene = field.cluster_blocks[0][0].gene
for k in range(len([3, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, -0.7415102564729918, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8697140108411625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9404010580334337, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.10539544735363338, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.2609445855298007, 0.4296071107294721, 0, 0, 0, 0, 0, -0.5639035801785157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.3036535845476347, 0.951252981947122, 0, 0, 0, 0.41495907225046724, 0, 0, 0, 0.006526934879678592, 0, 0, 0, 0, 0, 0.09744288457900852, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05572104098062414, 0, 0.23921596641423593, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.16202929969704183, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.07880910110232286, 0.7381859828260513, 0, 0, 0, 0, 0, 0, -0.39626971046683357, 0, -0.4280582796363266, 0, 0, 0, -0.7955873374490632, -0.21088953031046564, 0, 0, 0, 0, 0, 0, 0, 0, -0.22480503331852786, 0.5947347970766099, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5459087896293102, 0.15086113982354776, 0, 0, 0.34840442724827714, 0, 0.605118429733694, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.7435913218220169, 0, 0, 0, 0, 0.7862375136751716, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27783895509153544, 0, 0, 0, 0, 0, 0, 0, 0.9970397447895325, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.33492686299749375, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.7764957185006769, [4, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 1, 1, 1, 0, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [7, 0, 0, 0, 0, 0, 0, 11, 10, 10, 10, 10, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8498411967858961, 0, 0, 0.3158193020175286, 0, 0, 0, 0, 0, 0, 0, -0.8743488921694009, 0, 0, 0, 0, 0, 0, 0, 0, -0.8630542701818344, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9812282489073323, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.46214691098614513, 0, 0, 0, 0, 0, 0, -0.2288207307795478, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.762114905476537, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.13130767636161234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.708116131589617, 0, 0, 0, 0, 0, 0, 0.611853780398069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5563330084395761, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.172957973386189, 0, 0, 0, 0, 0, 0.33928831426664674, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.19930266132547336, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6939593547662717, 0.5, 0, 0, 0, 0, 0.979326271621169, 0, 0, 0, -0.003623215324062734, 0.5152278842722937, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7109101050679383, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4592497891469791, 0.8931787103616899, 0, 0, 0, 0, 0, -0.8339823677567371, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6759339477907305, 0, 0, 0, 0, 0, 0, 0, 0, 0.32508999304714226, 0, 0, 0, -0.7491919920764178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9904613891847209, -0.32148411989598746, 0, 0.25252561081296787, 0, 0, 0, 0.42474999721660156, 0, 0, 0, -0.6500959508512023, 0, 0, 0, -0.2540423506999643, 0, 0, 0, 0, 0, 0, 0, 0.5648889228128691, 0, 0, 0, 0, 0, 0.07395522503757745, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2606106807097277, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.13726034889892214, 0, 0, 0, 0, 0, 0, -0.6179558394185269, -0.34849327371763916, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0815795681364635, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6235274109452742, 0.05762735414222386, 0, 0, 0, 0, 0, -0.3312476762417105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5078966816346941, 0, 0, 0, 0, 0, 0.8710086596043547, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.556437482386027, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.13153503734155847, 0, 0, 0, 0, 0, 0.0833048318732279, 0, 0, -0.8343018546474261, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9148905591678886, 0, 0, 0, 0, 0, 0.6666878751388801, 0, 0, -0.05193803448395351, 0.45899137530274325, 0, 0, 0, 0, 0.5598108171926368, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.561402865843309, 0, 0, 0, 0, 0.7428128111625523, 0, 0, 0, 0, 0, 0, 0, -0.32487303194139616, -0.9430558490244467, 0, 0, 0, 0, 0.05487781385249213, 0, 0, 0, 0.31807325676181164, 0, 0, -0.6907758034866658, 0, -0.24659688606918606, 0.7567710634791465, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9122300194570492, 0, 0, -0.13966806740079463, 0, 0, 0, 0, 0, -0.27315791669244915, 0, 0, 0, 0.4994457695431762, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6936633862248949, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4881837990682525, 0, 0.14747437931637064, 0.1653075081004427, 0.11331506695795057, 0, 0, 0, -0.21284886149752236, 0, 0, 0, 0, 0, 0, 0, -0.6554656518847366, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3756668234826641, 0, 0, 0, 0, 0, 0, 0, 0, -0.8641046031300204, 0, 0, 0, 0, 0.8096430089450195, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.21157091806928063, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1968728499949013, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.4752496923276255, 0.9714488481610506, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8752207047446667, 0, 0, 0, 0, 0, 0, 0.9268792132098398, 0, 0.46262166957267103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5764706624373315, 0, 0, 0, 0, 0, 0.6851804575058922, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6773169879473142, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.8316888729636125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0665087383751497, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])):
    if type([3, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, -0.7415102564729918, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8697140108411625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9404010580334337, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.10539544735363338, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.2609445855298007, 0.4296071107294721, 0, 0, 0, 0, 0, -0.5639035801785157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.3036535845476347, 0.951252981947122, 0, 0, 0, 0.41495907225046724, 0, 0, 0, 0.006526934879678592, 0, 0, 0, 0, 0, 0.09744288457900852, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05572104098062414, 0, 0.23921596641423593, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.16202929969704183, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.07880910110232286, 0.7381859828260513, 0, 0, 0, 0, 0, 0, -0.39626971046683357, 0, -0.4280582796363266, 0, 0, 0, -0.7955873374490632, -0.21088953031046564, 0, 0, 0, 0, 0, 0, 0, 0, -0.22480503331852786, 0.5947347970766099, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5459087896293102, 0.15086113982354776, 0, 0, 0.34840442724827714, 0, 0.605118429733694, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.7435913218220169, 0, 0, 0, 0, 0.7862375136751716, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.27783895509153544, 0, 0, 0, 0, 0, 0, 0, 0.9970397447895325, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.33492686299749375, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.7764957185006769, [4, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 1, 1, 1, 0, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [7, 0, 0, 0, 0, 0, 0, 11, 10, 10, 10, 10, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8498411967858961, 0, 0, 0.3158193020175286, 0, 0, 0, 0, 0, 0, 0, -0.8743488921694009, 0, 0, 0, 0, 0, 0, 0, 0, -0.8630542701818344, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9812282489073323, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.46214691098614513, 0, 0, 0, 0, 0, 0, -0.2288207307795478, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.762114905476537, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.13130767636161234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.708116131589617, 0, 0, 0, 0, 0, 0, 0.611853780398069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5563330084395761, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.172957973386189, 0, 0, 0, 0, 0, 0.33928831426664674, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.19930266132547336, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6939593547662717, 0.5, 0, 0, 0, 0, 0.979326271621169, 0, 0, 0, -0.003623215324062734, 0.5152278842722937, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7109101050679383, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4592497891469791, 0.8931787103616899, 0, 0, 0, 0, 0, -0.8339823677567371, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6759339477907305, 0, 0, 0, 0, 0, 0, 0, 0, 0.32508999304714226, 0, 0, 0, -0.7491919920764178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9904613891847209, -0.32148411989598746, 0, 0.25252561081296787, 0, 0, 0, 0.42474999721660156, 0, 0, 0, -0.6500959508512023, 0, 0, 0, -0.2540423506999643, 0, 0, 0, 0, 0, 0, 0, 0.5648889228128691, 0, 0, 0, 0, 0, 0.07395522503757745, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2606106807097277, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.13726034889892214, 0, 0, 0, 0, 0, 0, -0.6179558394185269, -0.34849327371763916, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0815795681364635, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6235274109452742, 0.05762735414222386, 0, 0, 0, 0, 0, -0.3312476762417105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5078966816346941, 0, 0, 0, 0, 0, 0.8710086596043547, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.556437482386027, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.13153503734155847, 0, 0, 0, 0, 0, 0.0833048318732279, 0, 0, -0.8343018546474261, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9148905591678886, 0, 0, 0, 0, 0, 0.6666878751388801, 0, 0, -0.05193803448395351, 0.45899137530274325, 0, 0, 0, 0, 0.5598108171926368, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.561402865843309, 0, 0, 0, 0, 0.7428128111625523, 0, 0, 0, 0, 0, 0, 0, -0.32487303194139616, -0.9430558490244467, 0, 0, 0, 0, 0.05487781385249213, 0, 0, 0, 0.31807325676181164, 0, 0, -0.6907758034866658, 0, -0.24659688606918606, 0.7567710634791465, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9122300194570492, 0, 0, -0.13966806740079463, 0, 0, 0, 0, 0, -0.27315791669244915, 0, 0, 0, 0.4994457695431762, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6936633862248949, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4881837990682525, 0, 0.14747437931637064, 0.1653075081004427, 0.11331506695795057, 0, 0, 0, -0.21284886149752236, 0, 0, 0, 0, 0, 0, 0, -0.6554656518847366, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3756668234826641, 0, 0, 0, 0, 0, 0, 0, 0, -0.8641046031300204, 0, 0, 0, 0, 0.8096430089450195, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.21157091806928063, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1968728499949013, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.4752496923276255, 0.9714488481610506, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8752207047446667, 0, 0, 0, 0, 0, 0, 0.9268792132098398, 0, 0.46262166957267103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5764706624373315, 0, 0, 0, 0, 0, 0.6851804575058922, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6773169879473142, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.8316888729636125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0665087383751497, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]][k]) == list :
        print("Hi! I'm a list")
        print(k)

print(f"len(gene){len(gene)}")
for k in range(10000):
    print(k)
    mutated_genes = mutate_genes(copy.deepcopy(gene))
    print(f"mutated_genes{mutated_genes}")
    print(len(f"len(mutated_genes){len(mutated_genes)}"))
    field.cluster_blocks[0][0] = None
    field.create_cluster(0,0,mutated_genes)

#for k in range(5):
    #print("----------------------------------------------------------------")
    #field.cycle()