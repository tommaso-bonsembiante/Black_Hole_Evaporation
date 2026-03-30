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
#d^2N/dpdt
def d2N_dpdt(p_array, mi, s, gi, Mi):
    if s == 0:     Qs_table = Qs0
    elif s == 0.5: Qs_table = Qshalf
    elif s == 1:   Qs_table = Qs1
    elif s == 2:   Qs_table = Qs2
    else:          return 0.0
    stat_sign = 1.0 if s == 0.5 else -1.0
    Ti = 1.0 / (8 * np.pi * G * Mi)
    E_array = np.sqrt(p_array**2 + mi**2)
    x_eval = 2.0 * G * Mi * E_array
    Qs_interp = np.interp(x_eval, x0_grid, Qs_table, left=0.0, right=0.0)
    gamma_val = (np.exp(E_array / Ti) + stat_sign) * Qs_interp
    sigma_val = np.pi * gamma_val / (E_array**2)
    d2N = (gi / (2 * np.pi**2)) * (sigma_val / (np.exp(E_array / Ti) + stat_sign)) * (p_array**3 / E_array)
    return d2N
#d^2N/dpdt GO Limit
def d2N_dpdtGO(p_array, mi, s, gi, Mi):
    if s == 0:     Qs_table = Qs0
    elif s == 0.5: Qs_table = Qshalf
    elif s == 1:   Qs_table = Qs1
    elif s == 2:   Qs_table = Qs2
    else:          return 0.0
    stat_sign = 1.0 if s == 0.5 else -1.0
    Ti = 1.0 / (8 * np.pi * G * Mi)
    E_array = np.sqrt(p_array**2 + mi**2)
    sigma_GO=27.0 * np.pi * (G * Mi)**2
    d2N = (gi / (2 * np.pi**2)) * (sigma_GO / (np.exp(E_array / Ti) + stat_sign)) * (p_array**3 / E_array)
    return d2N
#epsilon_tot
def epsilon_tot(M_curr, m_DM, s_DM, g_DM):
    T_curr = 1.0 / (8 * np.pi * G * M_curr)
    eps_tot = 0.0
    Ei = x0_grid / (2.0 * G * M_curr)
    xi = Ei / T_curr
    for p_info in sm_content:
        mi = p_info['m'] * conv_gev_to_Mp
        zi = mi / T_curr
        mask = xi >= zi
        if not np.any(mask): continue
        x_int = xi[mask]
        psi_val = psi(p_info['s'], M_curr)[mask]
        stat_sign = 1.0 if p_info['s'] == 0.5 else -1.0
        integrand = (psi_val * (x_int**2 - zi**2) * x_int) / (np.exp(x_int) + stat_sign)
        epsilon_i = (27.0 / (8192.0 * np.pi**5)) * simpson(y=integrand, x=x_int)
        eps_tot += p_info['gi'] * epsilon_i   
    zi_DM = m_DM / T_curr
    mask_DM = xi >= zi_DM
    if np.any(mask_DM):
        x_int_DM = xi[mask_DM]
        psi_val_DM = psi(s_DM, M_curr)[mask_DM]
        stat_sign_DM = 1.0 if s_DM == 0.5 else -1.0
        integrand_DM = (psi_val_DM * (x_int_DM**2 - zi_DM**2) * x_int_DM) / (np.exp(x_int_DM) + stat_sign_DM)
        epsilon_i_DM = (27.0 / (8192.0 * np.pi**5)) * simpson(y=integrand_DM, x=x_int_DM)
        eps_tot += g_DM * epsilon_i_DM  
    return eps_tot
#dN_dp
def dN_dp(p_array, mi, s, gi, Mi):
    M_int = np.geomspace(Mi, 1.0, 1500) 
    integrand_matrix = np.zeros((len(M_int), len(p_array)))
    for i, M_curr in enumerate(M_int):
        eps = epsilon_tot(M_curr, mi, s, gi)
        d2N = d2N_dpdt(p_array, mi, s, gi, M_curr) 
        dt_dM = (M_curr**2) / eps
        integrand_matrix[i, :] = d2N * dt_dM    
    return np.abs(simpson(y=integrand_matrix, x=M_int, axis=0))
#dN_dp_GO
def dN_dp_GO(p_array, mi, s, gi, Mi):
    M_int = np.geomspace(Mi, 1.0, 1500) 
    integrand_matrix = np.zeros((len(M_int), len(p_array)))
    for i, M_curr in enumerate(M_int):
        eps = epsilon_tot(M_curr, mi, s, gi)
        d2N = d2N_dpdtGO(p_array, mi, s, gi, M_curr) 
        dt_dM = (M_curr**2) / eps
        integrand_matrix[i, :] = d2N * dt_dM    
    return np.abs(simpson(y=integrand_matrix, x=M_int, axis=0))
p_over_T = np.linspace(0.1, 60.0, 400)    
p_array = p_over_T * T_BH
def Boltzman (p_array,mi,s):
    E_array = np.sqrt(p_array**2 + mi**2)
    stat_sign = 1.0 if s == 0.5 else -1.0
    boltzmann = (p_array**2) / (np.exp(E_array / T_BH) + stat_sign)
    return boltzmann
# DM Input
m_DM_T = float(input("Enter the DM particle mass (in T_BH units): "))
s_DM = float(input("Enter the DM particle spin: "))
g_DM = float(input("Enter the DM particle d.o.f: "))
m_DM = T_BH * m_DM_T
dNdp_full = dN_dp(p_array, m_DM, s_DM, g_DM, M)
dNdp_GO = dN_dp_GO(p_array, m_DM, s_DM, g_DM, M)
boltzmann_dist = Boltzman(p_array, m_DM, s_DM)
# Normalization
area_full = simpson(y=dNdp_full, x=p_array)
area_GO = simpson(y=dNdp_GO, x=p_array)
area_boltz = simpson(y=boltzmann_dist, x=p_array)
print(area_full)
print(area_GO)
print(area_boltz)
norm_full = (dNdp_full / area_full) * T_BH
norm_GO = (dNdp_GO / area_GO) * T_BH
norm_boltz = (boltzmann_dist / area_boltz) * T_BH
# Average Momentum
avg_p = simpson(y=p_array * dNdp_full, x=p_array) / area_full
#Plot
plt.figure(figsize=(9, 6))
plt.plot(p_over_T, norm_full, color='purple', linewidth=2.5, label='Full Greybody Factors')
plt.plot(p_over_T, norm_GO, color='blue', linestyle='--', linewidth=2, label='GO Limit')
plt.plot(p_over_T, norm_boltz, color='green', linestyle='--', linewidth=2, label='Boltzmann Distribution')
plt.axvline(avg_p / T_BH, color='gray', linestyle='--', linewidth=1.5, label=r'$\langle p \rangle$')
plt.yscale('log')
plt.xlim(0.1, 50)
plt.ylim(3e-3, 0.3) 
plt.xlabel(r'$p / T_{BH}^{in}$', fontsize=14)
plt.ylabel(r'$T_{BH}^{in} \times (\mathrm{normalized} \ dN/dp)$', fontsize=14)
plt.title(fr'Phase-space distribution ($m_{{DM}} = {m_DM_T} \, T_{{BH}}^{{in}}$)', fontsize=15)
plt.legend(loc='upper right', frameon=True, fontsize=12)
plt.grid(alpha=0.15)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()
