import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({'font.size': 16})
import seaborn as sns

#initial data
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

# Calculation Parameters
K=P_0/Delta_T # kW/°C 
l=(np.sum(np.divide(bi,li))/b)**(-1)
tau_f=M_f*C_f/K #s 
tau_c=M_c*C_c/K #s 
tau_0=M_c/G #s

# used values
alpha_f = - 0.151 - 0.0429          #      alpha_d + alpha_a   [pcm/K]
alpha_c = - 1.22671 - 0.7741        #      alpha_c + alpha_r   [pcm/K]
alpha_h = 10                        #      ah ~ 10-15          [pcm/K]


# Points for mapping stability
M=100
N=100
L_X=np.zeros(N) 
L_Y=np.zeros(M) 
Ma=np.zeros((M,N))
# Ma=np.zeros(M)
# Eigenvalues calculation
for KP_i in range(0, M): 
    for KI_i in range(0, N): 
        K_P=KP_i * 1e-7 
        K_I=KI_i * 1e-7 
        L_X[KI_i]=K_I
        L_Y[KP_i]=K_P 
        A = np.array([[-b/L, bi[0]/L, bi[1]/L, bi[2]/L, bi[3]/L, bi[4]/L, bi[5]/L, alpha_f/L, alpha_c/L, alpha_h * K_P / L, alpha_h * K_I / L],
                      [li[0], -li[0], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [li[1], 0, -li[1], 0, 0, 0, 0, 0, 0, 0, 0],
                      [li[2], 0, 0, -li[2], 0, 0, 0, 0, 0, 0, 0],
                      [li[3], 0, 0, 0, -li[3], 0, 0, 0, 0, 0, 0],
                      [li[4], 0, 0, 0, 0, -li[4], 0, 0, 0, 0, 0],
                      [li[5], 0, 0, 0, 0, 0, -li[5], 0, 0, 0, 0],
                      [P_0/(M_f*C_f), 0, 0, 0, 0, 0, 0, -1/tau_f, 1/tau_f, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1/tau_c, -1/tau_c-2/tau_0, 0, 0],
                      [b/L, -bi[0]/L, -bi[1]/L, -bi[2]/L, -bi[3]/L, -bi[4]/L, -bi[5]/L, -alpha_f/L, -alpha_c/L, -alpha_h * K_P / L, -alpha_h * K_I / L],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
        P1=np.linalg.eigvals(A) 
        h=np.max(np.real(P1)) 
        Ma[KP_i, KI_i]=h

# Plotting Results in term of stability map
plt.figure()
levels = [-10.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
contour = plt.contour(L_X, L_Y, Ma, levels, colors = 'black')
plt.clabel(contour, colors = 'pink', fmt = '%2.1f', fontsize=12)
contour_filled = plt.contourf(L_X, L_Y, Ma, cmap='RdGy')
plt.colorbar(contour_filled)
# plt.scatter(alpha_c_used, alpha_f_used, color='lime')
plt.title('Dominant eigenvalue')
plt.xlabel('K_I')
plt.ylabel('K_P')
plt.show()

# fig, ax = plt.subplots()
# ax = sns.heatmap(Ma)
# plt.show()