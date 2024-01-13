from scipy.integrate import odeint,solve_ivp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (16, 9)

# initial data
L = 0.8066e-6 # second, LAMBDA mean generation time, not L diffusion length
li = 1e-5 * np.array([0.0125, 0.0292, 0.0895, 0.2575, 0.6037, 2.6688])   # decay constant, per second
b = 319e-5
bi = 1e-5 * np.array([6.124, 71.40, 34.86, 114.1, 69.92, 22.68])

P_0 = 300e3  # kW thermal
Delta_T = 460 # °C
Tin_0 = 400 # °C
M_f = 2132 # kg
C_f = 376e-3 # kJ/(kg*K)
M_c = 5429 # kg
C_c = 146e-3 # kJ/(kg*K)
G = 25757 # kg/s

# feedback coefficients
# typical values for a PWR are:
alpha_c = - 1.22671e-5 - 0.7741e-5        #      alpha_c + alpha_r   [pcm/°C --> pure number/K]
alpha_f = - 0.151e-5 - 0.0429e-5          #      alpha_d + alpha_a   [pcm/°C --> pure number/K]
alpha_h = 10e-5                           #      ah ~ 10-15   [pcm/cm --> pure number/K]

# Calculation Parameters
l=(np.sum(np.divide(bi,li))/b)**(-1)
K=P_0/Delta_T # kW/°C
tau_f=M_f*C_f/K #s
tau_c=M_c*C_c/K #s
tau_0=M_c/G #s

# initial conditions
C_0=(P_0/L)*np.divide(bi,li)
TH_0=np.array([[Tin_0+P_0/(2*G*C_c)+Delta_T], [Tin_0+P_0/(2*G*C_c)]])
NEU_0=np.append(P_0,C_0)
PWR_0=np.append(NEU_0, TH_0)

# time points
t_0=0
t_f=50
t_span = (t_0, t_f)
t = np.linspace(t_0, t_f, 1000)

#Input
rho_0=0
dh=1
U = [dh, Tin_0]

# Model    
def Dyn6_model(t, state):
    P,C1,C2,C3,C4,C5,C6,Tf,Tc = state
    dPdt = (alpha_h*U[0]+alpha_f*(Tf-TH_0[0, 0])+alpha_c*(Tc-TH_0[1, 0])-b)*P/L + li[0]*C1+li[1]*C2+li[2]*C3+li[3]*C4+li[4]*C5+li[5]*C6
    dC1dt = (bi[0]/L)*P-li[0]*C1
    dC2dt = (bi[1]/L)*P-li[1]*C2
    dC3dt = (bi[2]/L)*P-li[2]*C3
    dC4dt = (bi[3]/L)*P-li[3]*C4
    dC5dt = (bi[4]/L)*P-li[4]*C5
    dC6dt = (bi[5]/L)*P-li[5]*C6
    dTfdt = P/(M_f*C_f) - K * (Tf-Tc) / (M_f*C_f)
    dTcdt = K*(Tf-Tc)/(M_c*C_c)-2*G*(Tc-U[1])/(M_c)
    dXdt = [dPdt,dC1dt,dC2dt,dC3dt,dC4dt,dC5dt,dC6dt,dTfdt,dTcdt]
    return dXdt

# ODE solver
sol = solve_ivp(Dyn6_model, t_span, PWR_0, t_eval=t, method='RK45', rtol = 1e-6, atol = 1e-8)

# Plotting Results
plt.plot(t[0:len(sol.y[0])],sol.y[0]/10**6,'b-',label=r'P')
plt.ylabel('Power (GW)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.show()
exit()
plt.savefig('Power')
#
plt.cla()
plt.plot(t,sol.y[1],'r--',label=r'C1')
plt.ylabel('Precursor C1')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C1')

#
plt.cla()
plt.plot(t,sol.y[2],'r--',label=r'C2')
plt.ylabel('Precursor C2')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C2')
#
plt.cla()
plt.plot(t,sol.y[3],'r--',label=r'C3')
plt.ylabel('Precursor C3')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C3')
#
plt.cla()
plt.plot(t,sol.y[4],'r--',label=r'C4')
plt.ylabel('Precursor C4')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C4')
#
plt.cla()
plt.plot(t,sol.y[5],'r--',label=r'C5')
plt.ylabel('Precursor C5')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C5')
#
plt.cla()
plt.plot(t,sol.y[6],'r--',label=r'C6')
plt.ylabel('Precursor C6')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C6')
#
#
plt.cla()
plt.plot(t,sol.y[7],'g-',label=r'Tf')
plt.ylabel('Tf (°C)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Tf')
#
#
plt.cla()
plt.plot(t,sol.y[8],'g-',label=r'Tc')
plt.ylabel('Tc (°C)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Tc')
#
#
plt.cla()
plt.plot(t,2*sol.y[8]-U[1],'g-',label=r'Tout')
plt.ylabel('Tout (°C)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Tout')
#
#
plt.cla()
plt.plot(t,(alpha_h*U[0]+alpha_f*(sol.y[7]-TH_0[0,0])+alpha_c*(sol.y[8]-TH_0[1,0]))*10**5,'k--',label=r'Reactivity (pcm)')
plt.ylabel('Reactivity (pcm)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('reactivity')
#
# Plotting Results in terms of variations
plt.cla()
plt.plot(t,(sol.y[0]-PWR_0[0])/10**3,'b-',label=r'P')
plt.ylabel('Power (MW)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Power 1')
#
plt.cla()
plt.plot(t,(sol.y[1]-PWR_0[1]),'r--',label=r'C1')
plt.ylabel('Precursor C1')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C1 1')
#
plt.cla()
plt.plot(t,(sol.y[2]-PWR_0[2]),'r--',label=r'C2')
plt.ylabel('Precursor C2')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C2 1')
#
plt.cla()
plt.plot(t,(sol.y[3]-PWR_0[3]),'r--',label=r'C3')
plt.ylabel('Precursor C3')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C3 1')
#
plt.cla()
plt.plot(t,(sol.y[4]-PWR_0[4]),'r--',label=r'C4')
plt.ylabel('Precursor C4')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C4 1')
#
plt.cla()
plt.plot(t,(sol.y[5]-PWR_0[5]),'r--',label=r'C5')
plt.ylabel('Precursor C5')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C5 1')
#
plt.cla()
plt.plot(t,(sol.y[6]-PWR_0[6]),'r--',label=r'C6')
plt.ylabel('Precursor C6')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Precursor C6 1')
#
plt.cla()
plt.plot(t,(sol.y[7]-PWR_0[7]),'g-',label=r'Tf')
plt.ylabel('Tf (°C)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Tf 1')
#
plt.cla()
plt.plot(t,(sol.y[8]-PWR_0[8]),'g-',label=r'Tc')
plt.ylabel('Tc (°C)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Tc 1')
#
#
plt.cla()
plt.plot(t,2*sol.y[8]-U[1]-(2*PWR_0[8]-Tin_0),'g-',label=r'Tout')
plt.ylabel('Tout (°C)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Tout 1')
#
#
plt.cla()
plt.plot(t,(alpha_h*U[0]+alpha_f*(sol.y[7]-TH_0[0,0])+alpha_c*(sol.y[8]-TH_0[1,0]))*10**5,'k--',label=r'Reactivity (pcm)')
plt.ylabel('Reactivity (pcm)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.savefig('Reactivity 1')
##