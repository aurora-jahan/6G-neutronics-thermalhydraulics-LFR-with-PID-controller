from scipy import signal
import numpy as np
from numpy.linalg import eig
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from sys import exit

# initial data
P_0 = 300e3  # kW thermal
Delta_T = 460 # °C
Tin_0 = 400 # °C
M_f = 2132 # kg
C_f = 376e-3 # kJ/(kg*K)
M_c = 5429 # kg
C_c = 146e-3 # kJ/(kg*K)
G = 25757 # kg/s

L = 0.8066e-6 # second, LAMBDA mean generation time, not L diffusion length
li = 1e-5 * np.array([0.0125, 0.0292, 0.0895, 0.2575, 0.6037, 2.6688])   # decay constant, per second
b = 319e-5
bi = 1e-5 * np.array([6.124, 71.40, 34.86, 114.1, 69.92, 22.68])

# feedback coefficients
# note, typical values for a PWR are:
alpha_f = - 0.151e-5 - 0.0429e-5          #      alpha_d + alpha_a   [pcm/°C --> pure number/K]
alpha_c = - 1.22671e-5 - 0.7741e-5        #      alpha_c + alpha_r   [pcm/°C --> pure number/K]
alpha_h = 10e-5             #      ah ~ 10-15   [pcm/cm --> pure number/K]

# Calculated parameters
K = P_0/Delta_T # kW/°C
tau_f = M_f*C_f/K #s
tau_c = M_c*C_c/K #s
tau_0 = M_c/G #s
l = (np.sum(np.divide(bi, li)) / b)**(-1)        # Lambda decay constant, not l neutron life time

# A,B,C,D Matrices
A = np.array([[-b/L, bi[0]/L, bi[1]/L, bi[2]/L, bi[3]/L, bi[4]/L, bi[5]/L, alpha_f/L, alpha_c/L],
              [li[0], -li[0], 0, 0, 0, 0, 0, 0, 0],
              [li[1], 0, -li[1], 0, 0, 0, 0, 0, 0],
              [li[2], 0, 0, -li[2], 0, 0, 0, 0, 0],
              [li[3], 0, 0, 0, -li[3], 0, 0, 0, 0],
              [li[4], 0, 0, 0, 0, -li[4], 0, 0, 0],
              [li[5], 0, 0, 0, 0, 0, -li[5], 0, 0],
              [P_0/(M_f*C_f), 0, 0, 0, 0, 0, 0, -1/tau_f, 1/tau_f],
              [0, 0, 0, 0, 0, 0, 0, 1/tau_c, -1/tau_c-2/tau_0]])

B = np.array([[alpha_h/L, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 2/tau_0]])

C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 2],
              [0, 0, 0, 0, 0, 0, 0, alpha_f, alpha_c]])

D = np.array([[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, -1],
              [alpha_h, 0]])

# SS Model
Dyn1_PWR_6G = signal.StateSpace(A, B, C, D)

# Eigenvalues calculation
# w, v = eig(A)
# w = eigvals(A)
# print('E-value:', w)
# # print('E-vector', v)
# for eigenvalue in w:    
#     plt.scatter(eigenvalue.real, eigenvalue.imag, label=f'{eigenvalue.real} + {eigenvalue.imag} j')
# plt.hlines([0], -5000, 2, 'black')
# plt.vlines([0], -5, 5, 'black')
# plt.xscale('symlog')
# plt.yscale('linear')
# plt.xlabel('Real part')
# plt.ylabel('Imaginary part')
# plt.title('Eigenvalues')
# plt.legend(loc='upper left')
# plt.show()

# Initial state of the system
X0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# time points
t = np.linspace(0, 50, 300000)

#input
U = [[0], [0]]

# Simulation
TF_h = signal.ss2tf(A, B, C, D, input=0)
TF_T0 = signal.ss2tf(A, B, C, D, input=1)
print(TF_h)
print(TF_T0)

# time points
# t = np.linspace(0,5,300000)
t, Y_h = signal.step(TF_h, X0=[0.01, 0, 0, 0, 0, 0, 0, 0, 0], T= t)
t, Y_T0 = signal.step(TF_T0, X0=[0.01, 0, 0, 0, 0, 0, 0, 0, 0], T= t)

fig, axs = plt.subplots(6, 2, figsize=(16, 9))
axs[0, 0].plot(t, Y_h[:, 0] * P_0/1000)
axs[0, 0].set_title('Power_P (MW)')
axs[0, 0].grid()

Y_h_eta = np.sum(Y_h[:, 1:7], axis=1)
axs[1, 0].plot(t, Y_h_eta)
axs[1, 0].set_title('Normalized total precursors (-)')
axs[1, 0].grid()

axs[2, 0].plot(t, Y_h[:, 7])
axs[2, 0].set_title('Tf_h')
axs[2, 0].grid()

axs[3, 0].plot(t, Y_h[:, 8])
axs[3, 0].set_title('Tc_h')
axs[3, 0].grid()

axs[4, 0].plot(t, Y_h[:, 9])
axs[4, 0].set_title('Tout_h')
axs[4, 0].grid()

axs[5, 0].plot(t, Y_h[:, 10]*10**5)
axs[5, 0].set_title('Total reactivity (pcm)')
axs[5, 0].grid()

axs[0, 1].plot(t, Y_T0[:, 0]*P_0/1000, 'tab:orange')
axs[0, 1].set_title('Power_T0 (MW)')
axs[0, 1].grid()

Y_T0_eta = np.sum(Y_T0[:, 1:7], axis=1)
axs[1, 1].plot(t, Y_T0_eta, 'tab:orange')
axs[1, 1].set_title('Normalized precursors (-)')
axs[1, 1].grid()

axs[2, 1].plot(t, Y_T0[:, 7], 'tab:orange')
axs[2, 1].set_title('Tf_T0')
axs[2, 1].grid()

axs[3, 1].plot(t, Y_T0[:, 8], 'tab:orange')
axs[3, 1].set_title('Tc_T0')
axs[3, 1].grid()

axs[4, 1].plot(t, Y_T0[:, 9], 'tab:orange')
axs[4, 1].set_title('Tout_T0')
axs[4, 1].grid()

axs[5, 1].plot(t, Y_T0[:, 10]*10**5, 'tab:orange')
axs[5, 1].set_title('Total reactivity (pcm)')
axs[5, 1].grid()

# #print("Dominant eigenvalue: " + str(h))
plt.show()