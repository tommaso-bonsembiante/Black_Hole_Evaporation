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
# Convert gev->g
conv_gev_to_Mp = 8.190746e-20
# SM Particles List,  Mass in GeV
sm_content = [
    # Scalari (Spin 0)
    {'name': 'Higgs', 's': 0, 'gi': 1, 'm': 125.10},
    # Fermions - Quarks (gi = 3 colors * 2 spin * 2 anti.)
    {'name': 'Top',    's': 0.5, 'gi': 12, 'm': 172.76},
    {'name': 'Bottom', 's': 0.5, 'gi': 12, 'm': 4.18},
    {'name': 'Charm',  's': 0.5, 'gi': 12, 'm': 1.27},
    {'name': 'Strange','s': 0.5, 'gi': 12, 'm': 0.096},
    {'name': 'Down',   's': 0.5, 'gi': 12, 'm': 0.0047},
    {'name': 'Up',     's': 0.5, 'gi': 12, 'm': 0.0022},
    # Fermions (Spin 0.5) - Leptons (gi = 2 spin * 2 anti.)
    {'name': 'Tau',      's': 0.5, 'gi': 4, 'm': 1.776},
    {'name': 'Muon',     's': 0.5, 'gi': 4, 'm': 0.1056},
    {'name': 'Electron', 's': 0.5, 'gi': 4, 'm': 0.000511},
    {'name': 'Neutrinos','s': 0.5, 'gi': 6, 'm': 1e-10}, # 3 neutrinos
    # Vettors (Spin 1)
    {'name': 'Z_boson', 's': 1, 'gi': 3,  'm': 91.187},
    {'name': 'W_boson', 's': 1, 'gi': 6,  'm': 80.379},
    {'name': 'Gluons',  's': 1, 'gi': 16, 'm': 0.0},
    {'name': 'Photon',  's': 1, 'gi': 2,  'm': 0.0},
    # Tensors (Spin 2)
    {'name': 'Graviton','s': 2, 'gi': 2,  'm': 0.0}
]
#dM_dt
dM_dt_total = 0.0
dM_dt_total_fraction = 0.0
t_planck = 5.391e-44
print(f"{'Particella':<12} | {'zi':<10} | {'Contributo dM/dt':<18}")
print("-" * 45)
for p_info in sm_content:
    mi = p_info['m']*conv_gev_to_Mp
    zi=mi/T_BH
    gi = p_info['gi']
    s = p_info['s']
    contributo = -gi * epsilon(mi,s,M)*(M_p**4 / M**2)
    contributo_norm=contributo/M
    dM_dt_total += contributo
    contributo_fraction = (abs(contributo) / M) * (1.0 / t_planck)
    dM_dt_total_fraction += contributo_fraction
    print(f"{p_info['name']:<12} |  {zi:<10.2e} | {abs(contributo_fraction):<18.2e} ")
print( dM_dt_total_fraction)
#dM_dt Behaviour
M_grafico = np.logspace(0, 35, 3000) 
dM_dt_valori = []
for M_test in M_grafico:
    dM_dt_temp = 0.0
    for p_info in sm_content:
        mi = p_info['m']*conv_gev_to_Mp
        gi = p_info['gi']
        s = p_info['s']
        contributo = -gi * epsilon(mi,s,M_test) * (M_p**4 / M_test**2)
        dM_dt_temp += contributo
    dM_dt_valori.append(abs(dM_dt_temp))
#Plot dM_dt    
plt.figure(figsize=(10, 6))
plt.plot(M_grafico, dM_dt_valori, color='red', linewidth=2)
plt.xscale('log')
plt.yscale('log') 
plt.xlabel('BH Mass [$M_p$]')
plt.ylabel('|dM/dt|') 
plt.title('|dM/dt| Evolution')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.show()
# dM_dt*M^2Plot
dM_dt_array = np.array(dM_dt_valori)
fM_valori = dM_dt_array * (M_grafico**2 / M_p**4)
plt.figure(figsize=(10, 6))
plt.plot(M_grafico, fM_valori, color='blue', linewidth=2)
plt.xscale('log')
plt.yscale('log') 
plt.xlim(1,10e30)
plt.xlabel('BH Mass [$M_p$]')
plt.ylabel('Evaporation Function $f(M) = |dM/dt| \\times M^2$') 
plt.title('Standard Model case')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.show()