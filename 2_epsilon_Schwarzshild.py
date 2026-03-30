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
# epsilon
def epsilon(mi,s,Mi):
    T_BHi=1.0 / (8 * np.pi * G * Mi)
    Ei=x0_grid / (2.0 * G * Mi)
    x=Ei/(T_BHi)
    zi=mi/T_BHi
    mask = x >= zi
    if not np.any(mask):
        return 0.0
    x_int = x[mask]
    psi_val=psi(s,Mi)[mask]
    stat_sign = 1.0 if s == 0.5 else -1.0
    integrand = (psi_val*x_int*(x_int**2 - zi**2))/(np.exp(x_int)+stat_sign)
    prefactor=(27.0 / (8192.0 * np.pi**5))
    return prefactor*simpson(y=integrand, x=x_int)
def epsilon_go(mi,s,Mi):
    T_BHi=1.0 / (8 * np.pi * G * Mi)
    Ei=x0_grid / (2.0 * G * Mi)
    x=Ei/(T_BHi)
    zi=mi/T_BHi
    mask = x >= zi
    if not np.any(mask): 
        return 0.0
    x_int = x[mask]
    stat_sign = 1.0 if s == 0.5 else -1.0
    # psi_val=1
    integrand = (x_int*(x_int**2-zi**2))/(np.exp(x_int)+stat_sign)
    prefactor=(27.0/(8192.0*np.pi**5))
    return prefactor*simpson(y=integrand, x=x_int)
# Range of z
z_vals = np.linspace(0,6,1000)
m_vals=z_vals*T_BH
results = {0:[],0.5:[],1:[],2:[]}
go_limits = {0:[],0.5:[]} 
for m in m_vals:
    results[0].append(epsilon(m,0,M))
    results[0.5].append(epsilon(m,0.5,M))
    results[1].append(epsilon(m,1,M))
    results[2].append(epsilon(m,2,M))
    go_limits[0].append(epsilon_go(m,0,M))
    go_limits[0.5].append(epsilon_go(m,0.5,M))
# Plot
plt.figure(figsize=(6, 6))
# Plot Greybody
plt.plot(z_vals, results[0],   color='mediumspringgreen', label=r'$s_i=0$', lw=2)
plt.plot(z_vals, results[0.5], color='maroon',            label=r'$s_i=1/2$', lw=2)
plt.plot(z_vals, results[1],   color='sandybrown',        label=r'$s_i=1$', lw=2)
plt.plot(z_vals, results[2],   color='cornflowerblue',    label=r'$s_i=2$', lw=2)
# Plot GO limit 
plt.plot(z_vals, go_limits[0],   'k:',  alpha=0.5, label='GO-limit, Boson')
plt.plot(z_vals, go_limits[0.5], 'k--', alpha=0.5, label='GO-limit, Fermion')
plt.yscale('log')
plt.xlim(0,20)
plt.ylim(1e-10, 1e-4) 
plt.xlabel(r'$z_s$')
plt.ylabel(r'$\epsilon_i(z_s)$')
plt.title("Evaporation Function")
plt.legend(loc='lower left', ncol=2)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()