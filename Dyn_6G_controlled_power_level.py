from scipy.integrate import odeint,solve_ivp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (16, 9)

# initial data
P_K = 300e3     # kW thermal
P_0 = 280e3  # kW thermal
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


# Calculation Parameters
l=(np.sum(np.divide(bi,li))/b)**(-1)
K=P_K/Delta_T # kW/°C
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
t_f=1500
t_span = (t_0, t_f)
t = np.linspace(t_0,t_f,100000)

#PID controller parameters
P_ref = 300e3  # kW thermal
Kp = 7.5e-6 # 7.5e-6 # 0.00001
Ki = 7.5e-7 # 7.5e-7 # 0.00001
Con_0_P = np.array(P_ref-P_0)
Con_0_PI = np.append(np.array(P_ref-P_0), 0)
Max_reactivity = 0.8*b

# Initial conditions
IC_P = np.append(PWR_0, Con_0_P)
IC_PI=np.append(PWR_0, Con_0_PI)
rho_0=0
U = Tin_0

# Model
def Dyn6_PI_model(t, state):
    P,C1,C2,C3,C4,C5,C6,Tf,Tc,e,E = state
    dPdt = (rho_0+min(alpha_h*(Kp*e+Ki*E),Max_reactivity)+alpha_f*(Tf-TH_0[0, 0])+alpha_c*(Tc-TH_0[1, 0])-b)*P/L + li[0]*C1+li[1]*C2+li[2]*C3+li[3]*C4+li[4]*C5+li[5]*C6
    dC1dt = (bi[0]/L)*P-li[0]*C1
    dC2dt = (bi[1]/L)*P-li[1]*C2
    dC3dt = (bi[2]/L)*P-li[2]*C3
    dC4dt = (bi[3]/L)*P-li[3]*C4
    dC5dt = (bi[4]/L)*P-li[4]*C5
    dC6dt = (bi[5]/L)*P-li[5]*C6
    dTfdt = P/(M_f*C_f) - K * (Tf-Tc) / (M_f*C_f)
    dTcdt = K*(Tf-Tc)/(M_c*C_c)-2*G*(Tc-U)/(M_c)
    dedt = -dPdt
    dEdt = e
    dXdt = [dPdt,dC1dt,dC2dt,dC3dt,dC4dt,dC5dt,dC6dt,dTfdt,dTcdt,dedt,dEdt]
    return dXdt

# ODE solver
sol_PI = solve_ivp(Dyn6_PI_model,t_span,IC_PI,t_eval=t,method='LSODA',rtol = 1e-6, atol = 1e-8)

# Control action
dh_PI = np.minimum(Kp * sol_PI.y[-2] + Ki * sol_PI.y[-1], Max_reactivity/alpha_h)

fig, ax1 = plt.subplots()

dh = dh_PI
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Position of control rods (cm)', color=color)
ax1.plot(t,dh,color=color, label='CR height')
ax1.vlines(0, 0, dh[0], color=color)
ax1.tick_params(axis='y', labelcolor=color)

sol = sol_PI
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Power (MW)')  # we already handled the x-label with ax1
ax2.plot(t, sol.y[0]/10**3, color=color, label='P')
# ax2.plot(t, sol.y[-2]/10**3, color=color, label='error')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.grid()
plt.show()
exit()
#
plt.plot(t,sol.y[1],'r--',label=r'C')
plt.ylabel('Precursor C')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.show()
#
plt.plot(t,sol.y[2],'r--',label=r'Tf')
plt.ylabel('Fuel temperature (°C)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.show()
#
plt.plot(t,sol.y[3],'r--',label=r'Tc')
plt.ylabel('Coolant temperature (°C)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.show()
#
plt.plot(t,2*sol.y[3]-U,'g--',label=r'Tout')
plt.ylabel('Outlet coolant temperature (°C)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.show()
#
plt.plot(t,(rho_0+alpha_h*dh+alpha_f*(sol.y[2]-TH_0[0,0])+alpha_c*(sol.y[3]-TH_0[1,0]))*10**5,'k--',label=r'rho')
plt.ylabel('Total reactivity (pcm)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.show()
#
#
# Plotting Results in terms of variations
plt.plot(t,(sol.y[0]-PWR_0[0])/10**3,'b-',label=r'P')
plt.ylabel('Power (MW)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.show()
#
plt.plot(t,(sol.y[1]-PWR_0[1]),'r--',label=r'C')
plt.ylabel('Precursor C')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.show()
#
plt.plot(t,(sol.y[2]-PWR_0[2]),'r--',label=r'Tf')
plt.ylabel('Fuel temperature (°C)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.show()
#
plt.plot(t,(sol.y[3]-PWR_0[3]),'r--',label=r'Tc')
plt.ylabel('Coolant temperature (°C)')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.show()
#
plt.plot(t,2*sol.y[3]-U-(2*PWR_0[3]-Tin_0),'g--',label=r'delta_Tout')
plt.ylabel('response')
plt.xlabel('time')
plt.legend(loc='best')
plt.grid()
plt.show()
#
plt.plot(t,(rho_0+alpha_h*dh+alpha_f*(sol.y[2]-TH_0[0,0])+alpha_c*(sol.y[3]-TH_0[1,0]))*10**5,'k--',label=r'rho')
plt.ylabel('Total reactivity')
plt.xlabel('time (s)')
plt.legend(loc='best')
plt.grid()
plt.show()
#
