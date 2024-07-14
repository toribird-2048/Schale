import itertools

class Chemical() :
    def __init__(self,elements:list[str]) :
        elements_dict = {"A":1+1j,"B":1-1j,"C":-1+1j,"D":-1-1j,"a":0.5+0.5j,"b":0.5-0.5j,"c":-0.5+0.5j,"d":-0.5-0.5j}
        self.__elements = elements
        real_energy = 0
        imag_energy = 0
        
        for element in elements :
            if element not in elements_dict.keys():
                raise ValueError(f"Invalid element: {element}")
            real_energy += elements_dict[element].real
            imag_energy += elements_dict[element].imag
            
        self.__complex = real_energy + imag_energy*1j
        self.__energy = abs(real_energy) + abs(imag_energy) + 2 ** (len(elements)-4)
        
    def get_energy(self) :
        return self.__energy
    
    def get_elements(self) :
        return self.__elements
    
    def get_complex(self) :
        return self.__complex
    
    
def calc_reaction_energy(reactants,products):
    reactants_energy = sum(r.get_energy() for r in reactants)
    products_energy = sum(p.get_energy() for p in products)
    return reactants_energy - products_energy
            
def reaction(energy,chemicals:list[Chemical]):
    if type(energy) == int :
        energy = float(energy)
    if not (energy == "inf" or type(energy) == float) :
        raise ValueError()        
    E_need = abs(sum([c.get_complex() for c in chemicals]))
    elements = [c.get_elements() for c in chemicals]
    c3 = Chemical(list(itertools.chain.from_iterable(elements)))
    E_reaction = calc_reaction_energy(chemicals,[c3])
    if energy == "inf" :
        return [c3]
    if energy < E_need :
        return elements, energy
    elif E_need <= energy :
        return [c3], energy + E_reaction
    else :
        raise ValueError()
    
def division(chem:Chemical,tempelature) :
    if type(tempelature) == int :
        tempelature = float(tempelature)
    if not (tempelature == "inf" or type(tempelature) == float) :
        raise ValueError()
    
    
#reac = reaction(1,[Chemical("A"),Chemical("D"),Chemical("c"),Chemical("d")])
#for c in reac[0] :
#    print(c.get_elements(),c.get_energy())
#reac[1]
