import numpy as np
import random
from dataclasses import dataclass

mutation_rate = 10 # 10/961

class Field:
    def __init__(self,height,width):
        self.height = height
        self.width = width
        self.cluster_blocks = [[None for _ in range(width)] for _ in range(height)]
        self.material_blocks = [[[0 for _ in range(5)] for _ in range(width)] for _ in range(height)]

    class Cluster :
        def __init__(self,field,x,y,stage_count,gene,ClusterID,detect_max_list,detect_min_list,neural,neural_output_maxes):
            """
            info1~info16,energy,red,green,blue,white
            """
            self.field:Field = field
            self.x = x
            self.y = y
            self.ClusterID:int = ClusterID
            self.stage_count:int = stage_count
            self.gene:str = gene
            self.stages:list = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for k in range(stage_count)]
            self.cell_stages:list[Field.Cluster.Cell] = [[] for k in range(stage_count)]
            self.detect_max_list:list = detect_max_list # length must be 16
            self.detect_min_list:list = detect_min_list # same as detect_max_list
            self.neural:np.array = neural #横21縦16
            self.neural_output_maxes = neural_output_maxes
            self.add_cell_stage(0,self.ClusterDupricator(self,0,[0,0,0,0,0],[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],]),0))

            #stage内の物量が多いほどenergy多く


        class Cell : #stage,energy,red,green,blue,white(material_list),energy,red,green,blue,white(detect_max_list),energy,red,green,blue,white(detect_min_list),energy,red,green,blue,white(neural_add_maxes),energy,red,green,blue,white(n),b00,b01...b025,b10...b125...b225......b2525(neural)
            def __init__(self,cluster,stage,materials_list,detect_max_list,detect_min_list,neural_add_maxes,neural_remove_maxes,neural,cell_ID):
                """
                Detect_list : In Stage -> Info1~Info16,energy,red,green,blue | In Cell -> energy,red,green,blue
                """
                self.cluster:Field.Cluster = cluster # 所属するクラスター
                self.storage:list[int,int,int,int,int] = materials_list
                self.detect_max_list:list = detect_max_list # Length must be 21 (ステージ内物質の検知情報)
                self.detect_min_list:list = detect_min_list # Length must be 21 ()
                self.neural:np.array = neural #横26縦26
                self.outputs:list[int,int,int,int,int] = [0,0,0,0,0]
                self.maxes:list[int,int,int,int,int] = [10,2,2,2,0]
                self.stage:int = stage
                self.input_queue:list[int,int,int,int,int] = [0,0,0,0,0]
                self.neural_add_maxes:list[int,int,int,int,int] = neural_add_maxes #意味的にはinputだけどinputだとニューラルネットワークの入力と意味が競合しちゃうからaddにしてある
                self.neural_remove_maxes:list[int,int,int,int,int] = neural_remove_maxes #こっちも意味はoutput
                self.cell_ID:int = cell_ID

            def __post_init__(self):
                for k in range(len(self.storage)) :
                    if self.maxes[k] <= self.storage[k] :
                        self.storage[k] = self.maxes[k]

            def detector(self):
                detecteds = []
                for k in range(21):
                    if cluster.info_stage(self.stage)[k] <= self.detect_min_list[k] :
                        detecteds.append(-1)
                    elif self.detect_max_list[k] <= cluster.info_stage(self.stage)[k] :
                        detecteds.append(1)
                    else :
                        detecteds.append((cluster.info_stage(self.stage)[k] - self.detect_min_list[k]) / (self.detect_max_list[k] - self.detect_min_list[k]) - 0.5)
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
                    cluster.add_stage(self.stage,k,infos[k])

            def input(self):
                """
                type-> 0:energy, 1:red, 2:green, 3:blue, 4:white
                """
                for k in range(5):
                    self.input_queue[k] = cluster.add_stage(self.stage,16+k,self.input_queue[k])
                self.add_storage(self.input_queue)
                self.input_queue = [0,0,0,0,0]

            def output(self):
                """
                type-> 0:energy, 1:red, 2:green, 3:blue, 4:white
                """
                output = self.remove_storage(self.outputs)
                for k in range(5):
                    cluster.add_stage(self.stage,16+k,output[k])

            def output_infos(self,type) :
                cluster.add_stage(self.stage,type,1)

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
                    if cluster.info_stage(self.stage)[k] <= self.detect_min_list[k] :
                        detecteds.append(-1)
                    elif self.detect_max_list[k] <= cluster.info_stage(self.stage)[k] :
                        detecteds.append(1)
                    else :
                        detecteds.append((cluster.info_stage(self.stage)[k] - self.detect_min_list[k]) / (self.detect_max_list[k] - self.detect_min_list[k]) - 0.5)
                for k in range(5):
                    if self.maxes[k] == 0 :
                        detecteds.append(-1)
                    else:
                        detecteds.append(2*self.storage[k]/self.maxes[k]-0.5)
                for k in range(5):
                    if cluster.field.material_blocks[cluster.x][cluster.y][k] <= self.detect_min_list[16+k] :
                        detecteds.append(-1)
                    elif self.detect_min_list[16+k] <= cluster.field.material_blocks[cluster.x][cluster.y][k] :
                        detecteds.append(1)
                    else :
                        detecteds.append((cluster.field.material_blocks[cluster.x][cluster.y][k] - self.detect_min_list[k]) / (self.detect_max_list[k] - self.detect_min_list[k]) - 0.5)

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

            def mutate_gene(self, gene_list:list):
                picked = random.sample(range(961),mutation_rate)
                for k in picked:
                    pass #TODO変異の制限（material_listがマイナス、MinがMaxより大きいとか）
                #TODO Cluster遺伝記法


        def decode_gene(self,gene_list:list):
            cell_type = gene_list[0]
            stage = gene_list[1]
            material_list = gene_list[2:7]
            detect_max_list = gene_list[7:12]
            detect_min_list = gene_list[12:17]
            neural_add_maxes = gene_list[17:22]
            neural_remove_maxes = gene_list[22:27]
            neural = []
            if cell_type == 0 or cell_type == 3 or cell_type == 4:
                for k in range(26):
                    neural.append([])
                    for l in range(26):
                        neural[k].append(gene_list[27+k*26+l])
            elif cell_type == 1:
                for k in range(36):
                    neural.append([])
                    for l in range(26):
                        neural[k].append(gene_list[27+k*26+l])
            elif cell_type == 2 or cell_type == 6:
                for k in range(28):
                    neural.append([])
                    for l in range(26):
                        neural[k].append(gene_list[27+k*26+l])
            elif cell_type == 5:
                for k in range(26):
                    neural.append([])
                    for l in range(31):
                        neural[k].append(gene_list[27+k*31+l])
            elif cell_type == 7:
                for k in range(27):
                    neural.append([])
                    for l in range(26):
                        neural[k].append(gene_list[27+k*26+l])
            return cell_type, stage, material_list, detect_max_list, detect_min_list, neural_add_maxes, neural_remove_maxes, neural

        def create_cell(self, gene_list:list):
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

        def next_stage(self):
            temp_stages = []
            temp_stages.append(self.stages[self.stage_count-1])
            for k in range(self.stage_count-1):
                temp_stages.append(self.stages[k])
            self.stages = temp_stages

        def remove_stage(self,stage,type,amount):
            if self.stages[stage][type] < amount :
                output = 0
                pass #error処理
            else :
                self.stages[stage][type] -= amount
                output = amount
                return output

        def add_stage(self,stage,type,amount) :
            self.stages[stage][type] += amount
            return amount


        def add_cell_stage(self, stage, cell:Cell):
            self.cell_stages[stage].append(cell)

        def cycle(self):
            for stage in range(self.stage_count):
                for cell in self.cell_stages[stage]:
                    cell.cycle()
            self.neural_net()
            self.next_stage()

        def detector(self):
            detecteds = []
            for k in range(21):
                if cluster.info_stage(0)[k] <= self.detect_min_list[k] :
                    detecteds.append(-1)
                elif self.detect_max_list[k] <= cluster.info_stage(0)[k] :
                    detecteds.append(1)
                else :
                    detecteds.append((cluster.info_stage(0)[k] - self.detect_min_list[k]) / (self.detect_max_list[k] - self.detect_min_list[k]) - 0.5)
            return detecteds

        def neural_net(self):
            output = np.dot(self.neural,np.array(self.detector()).T)
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
                    self.cluster_blocks[x][y].cycle()


ClusterIDs: list = []


field = Field(10,10)
cluster = Field.Cluster(field,0,0,3,"aaa",0,[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]),[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])
field.add_cluster(0,0,cluster)
cell = Field.Cluster.Cell(cluster,0,[0,0,0,0,0],[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[3,1,1,1,0],[3,1,1,1,0],np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]),1)
field.cluster_blocks[0][0].add_cell_stage(0,cell)
leaf = Field.Cluster.Leaf(cluster,0,[0,0,0,0,0],[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[30,1,1,1,0],[3,1,1,1,0],np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,-0.5,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0.5,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]),2)
field.cluster_blocks[0][0].add_cell_stage(0,leaf)


for k in range(5):
    field.cluster_blocks[0][0].add_stage(0,k,10)
for k in range(5):
    field.cycle()


##TODO:自動Cell,Cluster生成