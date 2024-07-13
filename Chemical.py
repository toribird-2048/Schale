class Chemical() :
    def __init__(self,*elements:str) :
        elements_dict = {"A":1+1j,"B":1-1j,"C":-1+1j,"D":-1-1j,"a":0.5+0.5j,"b":0.5-0.5j,"c":-0.5+0.5j,"d":-0.5-0.5j}
        self.__elements = elements
        real_energy = 0
        imag_energy = 0
        
        for element in list(elements) :
            if element not in elements_dict.keys():
                raise ValueError(f"Invalid element: {element}")
            real_energy += elements_dict[element].real
            imag_energy += elements_dict[element].imag
            
        self.__complex = real_energy + imag_energy*1j
        self.__energy = abs(real_energy) + abs(imag_energy) + 2 ** (len(elements)-5)
        
    def get_energy(self) :
        return self.__energy
    
    def get_elements(self) :
        return self.__elements
    
    def get_complex(self) :
        return self.__complex
        
def reaction(c1:Chemical,c2:Chemical,energy):
    if type(energy) == int :
        energy = float(energy)
    if not (energy == "inf" or type(energy) == float) :
        raise ValueError()        
    E_reaction = abs(c1.get_complex() - c2.get_complex())
    c3 = Chemical(*(c1.get_elements() + c2.get_elements()))
    c3_E_stable = c3.get_energy()
    c12_E_stable = c1.get_energy() + c2.get_energy()
    if energy == "inf" :
        return [c3]
    if energy < c3_E_stable - c12_E_stable + E_reaction :
        return [c1,c2], energy
    elif c3_E_stable - c12_E_stable + E_reaction <= energy :
        return [c3], energy + c3_E_stable - c12_E_stable
    else :
        raise ValueError()
    
c1 = Chemical("A","D")
c2 = Chemical("c","d")
print(c1.get_elements(),c1.get_energy())
print(c2.get_elements(),c2.get_energy())
reac = reaction(c1,c2,1.25)

for c in reac[0] :
    print(c.get_elements(),c.get_energy())
    
print(reac[1])
    