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
#Psi
def Psi(mi,s,Mi):
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
    integrand = (psi_val * (x_int**2 - zi**2)) / (np.exp(x_int) +stat_sign)
    # Integration using Simpson rule
    return simpson(y=integrand, x=x_int)
print(Psi(0,0,M))
print(Psi(0,0.5,M))
print(Psi(0,1,M))
print(Psi(0,2,M))
# Convert gev->g
conv_gev_to_Mp = 8.190746e-20
# SM particles list
# mass in GeV
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
print(f"{'Particella':<12} | {'Gamma_i ':<18}")
print("-" * 45)
# Denominator
M_steps = np.logspace(0, np.log10(M), 1000)
denom_array = np.zeros_like(M_steps)
for i, M_test in enumerate(M_steps):
    eps_tot = 0.0
    for p_j in sm_content:
        m_j = p_j['m'] * conv_gev_to_Mp
        eps_tot += p_j['gi'] * epsilon(m_j, p_j['s'], M_test)
    denom_array[i] = eps_tot
#eta
def eta(p_info):
    s_i = p_info['s']
    m_i = p_info['m']*conv_gev_to_Mp 
    integrand_ni = []
    for M_test, denom in zip(M_steps, denom_array):
        Psi_val = Psi(m_i, s_i, M_test)
        if denom > 0:
            integrand_ni.append((Psi_val * M_test) / denom)
        else:
            integrand_ni.append(0.0)
    integral_val = simpson(y=integrand_ni, x=M_steps)
    pref = 27.0 / (1024.0 * np.pi**4)
    eta_val = (pref / (M**2)) * integral_val 
    return eta_val
print(f"\n{'Specie SM':<15} | {'N_i (Totale)':<18}")
print("-" * 50)
Ntot = 0
for p_info in sm_content:
    eta_val = eta(p_info)
    Ni = p_info['gi'] * (M**2 / M_p**2) * eta_val
    Ntot += Ni
    print(f"{p_info['name']:<15} | {Ni:<18.2e}")
print("-" * 50)
print(f"{'TOTALE EMESSE':<15} | {Ntot:<18.2e}")