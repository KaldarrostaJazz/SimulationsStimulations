import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-tf','--t_bound',default=100,type=float)
parser.add_argument('-V0','--rest_voltage',default=-0.68,type=float)
parser.add_argument('-I','--I_ext',default=0,type=float)
args = parser.parse_args()
tf = args.t_bound
V0 = args.rest_voltage
I = args.I_ext
c = np.array([17.81, 47.71, 32.63,1.35,1.03])
E = np.array([0.55,-0.92])
y0 = np.array([V0,c[3]*V0+c[4]])

def f(t,y):
    dy0dt=(-(c[0]+c[1]*y[0]+c[2]*y[0]*y[0])*(y[0]-E[0])-26*y[1]*(y[0]-E[1])+I)/0.8
    dy1dt=(-y[1]+c[3]*y[0]+c[4])/1.9
    return np.array([dy0dt,dy1dt])
solution = solve_ivp(f,(0,tf),y0,'Radau')
fig, ax = plt.subplots(1,2)
ax[0].grid()
ax[0].plot(solution.y[0], solution.y[1], 'k')
ax[1].grid()
ax[1].plot(solution.t, solution.y[0], 'k')
plt.show()
