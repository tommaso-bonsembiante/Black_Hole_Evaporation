import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

#Load data
percorso = r"C:\Users\tomma\OneDrive\Desktop\Master Thesis\blackhawk_v2.3\blackhawk_v2.3\src\tables\gamma_tables"
file_0 = os.path.join(percorso, "spin_0.txt")
file_half= os.path.join(percorso, "spin_0.5.txt")
file_1= os.path.join(percorso, "spin_1.txt")
file_2= os.path.join(percorso, "spin_2.txt")
with open(file_0, 'r') as f:
    header = f.readline().split()
    # Skip 1st element 'a/x' and take the energies 
    x0_grid = np.array([float(val) for val in header[1:]]) 
data0 = np.loadtxt(file_0, skiprows=1)
with open(file_half, 'r') as f:
    header = f.readline().split()
    xhalf_grid = np.array([float(val) for val in header[1:]]) 
datahalf = np.loadtxt(file_half, skiprows=1)
with open(file_1, 'r') as f:
    header = f.readline().split()
    x1_grid = np.array([float(val) for val in header[1:]]) 
data1 = np.loadtxt(file_1, skiprows=1)
with open(file_2, 'r') as f:
    header = f.readline().split()
    x2_grid = np.array([float(val) for val in header[1:]]) 
data2 = np.loadtxt(file_2, skiprows=1)
#Parameters in Natural Units
G = 1.0
M_p = 1.0 
# BH Mass
Mg = float(input("Enter the black hole mass M_BH (in grams): "))
MGev=Mg*5.609*10**23 # In GeV
M=MGev/(1.22*10**19) # In Plank Units
# BH Temperature
T_BH = 1.0 / (8 * np.pi * G * M)
# BH Spin
a=data0[0,0]
# Energy Grid
E = x0_grid / (2.0 * G * M)
x_i=E/(T_BH)
# Gamma
Qs0 = data0[0,1:] 
Qshalf = datahalf[0,1:]
Qs1 = data1[0,1:]
Qs2 = data2[0,1:]
def gamma(s,Mi):
    stat_sign = 1.0 if s == 0.5 else -1.0
    T_BHi=1.0 / (8 * np.pi * G * Mi)
    Ei=x0_grid / (2.0 * G * Mi)
    if s == 0:    gamma = (np.exp(Ei/T_BHi)+stat_sign)*Qs0
    elif s == 0.5: gamma = (np.exp(Ei/T_BHi)+stat_sign)*Qshalf
    elif s == 1:   gamma = (np.exp(Ei/T_BHi)+stat_sign)*Qs1
    elif s == 2:   gamma = (np.exp(Ei/T_BHi)+stat_sign)*Qs2
    else: return 0.0
    return gamma
# sigma
def sigma(s,Mi):
    T_BHi=1.0 / (8 * np.pi * G * Mi)
    Ei=x0_grid / (2.0 * G * Mi)
    sigma=(np.pi * gamma(s,Mi)) / (Ei**2)
    return sigma
# psi
def psi(s,Mi):
    sigma_GO = 27.0 * np.pi * (G*Mi)**2
    psi=sigma(s,Mi)/sigma_GO
    return psi
results = {0:[],0.5:[],1:[],2:[]}
results[0]   = psi(0,M)
results[0.5] = psi(0.5,M)
results[1]   = psi(1,M)
results[2]   = psi(2,M)
# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_i, results[0],    color='tab:green',   linewidth=2, label='Spin 0')
plt.plot(x_i, results[0.5], color='tab:purple', linewidth=2, label='Spin 1/2')
plt.plot(x_i, results[1],    color='tab:blue',  linewidth=2, label='Spin 1')
plt.plot(x_i, results[2],    color='tab:orange',    linewidth=2, label='Spin 2')
plt.axhline(1.0, color='black', linestyle='--', alpha=0.6)
plt.xlim(0, 35)
plt.title("Reduced Absorption Cross Section $\psi(E)$", fontsize=14)
plt.xlabel(r"Energy $E/T_H$", fontsize=12)
plt.ylabel(r"$\psi$(E,0)", fontsize=12)
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
