import os
from ase.io import read,write
import numpy as np
from ase.build import add_adsorbate, fcc100, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
list =[]
images = []
#energy = []
energy2 = []


remove=np.empty((12870,8))
for i in range(1,14):       #generate a combination of 8 different numbers ranging from 1 to 16 which are the indexes of atoms not to remove 
    for j in range(i+1,15):
        for k in range(j+1,16):
            for l in range(k+1,17):
                for m in range(l+1,17):
                    for n in range(m+1,17):
                        for o in range(n+1,17):
                            for p in range(o+1,17):
                                list.append([i,j,k,l,m,n,o,p])  
rmv=np.array(list)
#print(list)

alll = np.full((12870,16),fill_value=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
#print(alll)
#alll.shape
#rmv.shape
#alll[1][2]=0
for i in range(0,12870):
    a=np.delete(alll[i], rmv[i]-1) # subtract the atoms to remove from the full list of the 16 atoms
    remove[i]=a
remove =remove+95 #correct the indexes to the actual indexes of the oxygen atoms to remove
remove.astype(int)  #indexes of atoms must be integers not floats


for x in range(0,1301):  #run the ML potential calculation for the first batch of 1300 out of the 12870 total

# Construct a sample structure, similar to the EMT relaxation example!
    adslab = read('init.traj')
    del adslab[[int(y) for y in np.ndarray.tolist(remove[x])]]   #remove corresponding atoms
    adslab.set_tags(np.ones(len(adslab)))
# Define the calculator
#checkpoint_path = '/home/solusegu/CASFER/1new_ft_gly/fine-tuning/checkpoints/2024-03-20-15-52-16-ft-oxides/checkpoint.pt'
    checkpoint_path = '/home/solusegu/Pythonmodules/ocp-main/checkpoints/gnoc_oc22_oc20_all_s2ef.pt'
# checkpoint_path = '/home/josegaut/PythonModules/ocp-main/checkpoints/gnoc_finetune_all_s2ef.pt'
    calc = OCPCalculator(checkpoint=checkpoint_path)

# Set up the calculator
    adslab.set_calculator(calc)

# os.makedirs("data", exist_ok=True)
    opt = BFGS(adslab, trajectory='qn.traj',logfile='qn.log')

    opt.run(fmax=0.05, steps=500)
    images.append(adslab)
    e2 = adslab.get_potential_energy()
#    e= [x,np.ndarray.tolist(rmv[x]+63),e2]
#    energy.append(e)
    energy2.append(e2)
#    print(e)
#energy = np.array(energy)
write('images.traj', images)
#print(energy)
f = open('E.txt', 'w')
f.write(str(energy2)[1:-1])
f.close()
