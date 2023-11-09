import matplotlib.pyplot as plt
import argparse
import numpy as np
from math import exp, sin, pi
from scipy.integrate import solve_ivp

parser = argparse.ArgumentParser(prog='FitzHugh-Nagumo')
VALID_TYPES = ['const','tri','sin','gauss','linear']
parser.add_argument('--type', choices=VALID_TYPES,required=True)
parser.add_argument('-i','--impulse',default=7/8,type=float)
parser.add_argument('-tc','--t_crit',default=1,type=float)
parser.add_argument('-tf','--t_bound',default=100,type=float)
args = parser.parse_args()
I_ext = args.impulse
tc = args.t_crit
tf = args.t_bound
impulse = args.type

a = 0.7
b = 0.8
T = 12.5

def I(time, t_crit, i_type):
    match i_type:
        case 'const':
            return I_ext
        case 'tri':
            if (time <= t_crit):
                return I_ext*time/t_crit
            elif (time <= 2*t_crit):
                return I_ext*(2-time/t_crit)
            else:
                return 0
        case 'sin':
            return I_ext*sin(2*pi*time/T)
        case 'gauss':
            return I_ext*exp(-(time-t_crit)*(time-t_crit)/(2*2))
        case 'linear':
            return I_ext*time*0.1
        case _:
            raise ValueError("results: status must be one of %r." % VALID_TYPES)

def f(t,y):
    y0 = y[0]-y[0]*y[0]*y[0]/3-y[1]+I(t,tc,impulse)
    y1 = (y[0]+a-b*y[1])/T
    return np.array([y0,y1])

y0 = np.array([-1.1995,-0.62427])
solution = solve_ivp(f,(0,tf),y0,'RK45')
xes = np.linspace(-2.5,2.5,500)
if (impulse == 'const'):
    null_clines = np.array([[x-x*x*x/3 + I_ext for x in xes],[(x+a)/b for x in xes]])
else:
    null_clines = np.array([[x-x*x*x/3 for x in xes],[(x+a)/b for x in xes]])
plt.grid()
plt.xlabel('t (ms)')
plt.ylabel('V')
#plt.ylim([-2.1,2])
plt.plot(solution.t,solution.y[0],'k')
plt.plot(solution.t, [I(i,tc,impulse) for i in solution.t], color='tab:gray', linestyle='--')
plt.figure()
plt.grid()
plt.xlabel('V')
plt.ylabel('W')
plt.plot(solution.y[0],solution.y[1],color='k')
plt.plot(xes, null_clines[0], color='tab:gray', linestyle='--')
plt.plot(xes, null_clines[1], color='tab:gray', linestyle='--')
plt.show()
