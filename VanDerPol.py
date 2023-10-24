import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import argparse
import time as tm

t_init = tm.time()
parser = argparse.ArgumentParser()
parser.add_argument('-mu', '--parameter', type=float, default=1)
parser.add_argument('-tf', '--t_bound',type=float, default=10)
parser.add_argument('-y0', '--initial_state', type=str, default='0.0,0.0')
args = parser.parse_args()
mu = args.parameter
tf = args.t_bound
y0 = np.array([float(x) for x in args.initial_state.split(',')])

def van_der_pol(t,y):
    return np.array([y[1], mu*(1-y[0]*y[0])*y[1]-y[0]])
def jacobian(t,y):
    return np.array([[0,1],[-2*mu*y[0]*y[1]-1, mu*(1-y[0]*y[0])]])
lintime = np.linspace(0,tf,int(tf*100))
t_start = tm.time()
solution = solve_ivp(van_der_pol,(0,tf),y0,'Radau',lintime,True,jac=jacobian)
t_calc = tm.time()
print(t_start-t_init, t_calc-t_start,t_calc-t_init)
time = solution.t
state = solution.y
fig, axes = plt.subplots(1,2)
axes[0].grid()
axes[1].grid()
axes[0].plot(state[0],state[1])
axes[1].plot(time, state[0])
plt.show()