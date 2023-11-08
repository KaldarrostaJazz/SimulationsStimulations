import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import argparse

INTEGRATORS = ['RK45','Radau','BDF']
parser = argparse.ArgumentParser()
parser.add_argument('-mu', '--parameter', type=float, default=1)
parser.add_argument('-tf', '--t_bound',type=float, default=10)
parser.add_argument('-y0', '--initial_state', type=str, default='0.0,0.0')
parser.add_argument('--integrator',type=str,choices=INTEGRATORS,default='Radau')
args = parser.parse_args()
mu = args.parameter
tf = args.t_bound
method = args.integrator
y0 = np.array([float(x) for x in args.initial_state.split(',')])

def van_der_pol(t,y):
    return np.array([y[1], mu*(1-y[0]*y[0])*y[1]-y[0]])
solution = solve_ivp(van_der_pol,(0,tf),y0,method,dense_output=True)
time = solution.t
state = solution.y
fig, axes = plt.subplots(1,2)
fig.suptitle("Van der Pol oscillator")
axes[0].set_title("Phase plane")
axes[1].set_title("Dynamics")
axes[0].grid()
axes[1].grid()
axes[0].plot(state[0],state[1],color='k')
axes[1].plot(time,state[0],color='k')
plt.show()
