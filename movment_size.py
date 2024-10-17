import numpy as np
import matplotlib.pyplot as plt


file_path = r"D:\PhD thesis\SimulationSV\cmake-build-debug\builds"
time = r"\2024-06-05_18-46-08"



file = r"\Vesicles_movement.xyz"
#file = "\Vesicles_movement0.xyz"
#file = "\VDW_force_vesicle.txt"
#file = "\VDW_force_protein.txt"
#file = "\total_force_vesicle.txt"
#file = "\total_force_protein.txt"
#file = "\SynapsinI_movement.xyz"
#file = "\electro_force_vesicle_protein.txt"
#file = "\electro_force_protein_vesicle.txt"
#file = "\electr_force_protein_protein.txt"
file_path = file_path + time + file

particle_movements = []

with open(file_path, 'r') as file:

    file.readline()
    file.readline()
    for line in file:
        if line == 300:
            continue
    for line in file:
        if line.startswith("Time"):
            continue  
        
    data = line.split()
    x, y, z = map(float, data[0:3]) 
    print(x, y, z)
    movement = np.sqrt(x**2 + y**2 + z**2)
    particle_movements.append(movement)
    
    
plt.hist(particle_movements, bins=2, density=True, alpha=0.7, color='blue')
plt.title('Histogram of Particle Movements')
plt.xlabel('Movement Magnitude(nm)')
plt.ylabel('Frequency')
plt.show()



