import matplotlib.pyplot as plt
import argparse
from math import exp, sin, pi
from scipy.integrate import simpson

parser = argparse.ArgumentParser(prog='FitzHugh-Nagumo')
VALID_TYPES = ['const','tri','sin','gauss']
parser.add_argument('--type', choices=VALID_TYPES,required=True)
parser.add_argument('-i','--impulse',default=1,type=float)
parser.add_argument('-tc','--t_crit',default=1,type=int)
parser.add_argument('-t','--t_bound',default=1000,type=int)
args = parser.parse_args()
I_ext = args.impulse
dt = 0.1
tc = args.t_crit
N = args.t_bound
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
            return I_ext*sin(5*pi*time*dt/T)
        case 'gauss':
            return I_ext*exp(-(time-t_crit)*(time-t_crit)/(2*2))
        case _:
            raise ValueError("results: status must be one of %r." % VALID_TYPES)

def f1(x, w, I):
    return x - x*x*x/3 - w + I

def f2(x, w, I):
    return (x + a - b*w)/T

x = [-1.1995]
w = [-0.62427]
t = [0]

for i in range(N):
    k11 = f1(x[i], w[i], I(t[i], tc, impulse))*dt
    k12 = f2(x[i], w[i], I(t[i], tc, impulse))*dt
    k21 = f1(x[i] + 0.5*k11, w[i] + 0.5*k12, I(t[i], tc, impulse))*dt
    k22 = f2(x[i] + 0.5*k11, w[i] + 0.5*k12, I(t[i], tc, impulse))*dt
    k31 = f1(x[i] + 0.5*k21, w[i] + 0.5*k22, I(t[i], tc, impulse))*dt
    k32 = f2(x[i] + 0.5*k21, w[i] + 0.5*k22, I(t[i], tc, impulse))*dt
    k41 = f1(x[i] + k31, w[i] + k32, I(t[i], tc, impulse))*dt
    k42 = f2(x[i] + k31, w[i] + k32, I(t[i], tc, impulse))*dt
    dx = (k11 + 2*k21 + 2*k31 + k41)/6
    dw = (k12 + 2*k22 + 2*k32 + k42)/6
    x.append(x[i] + dx)
    w.append(w[i] + dw)
    t.append(i*dt)

integral = [simpson(x[i],t[i]) for i in range(len(t))]
plt.grid()
#plt.ylim([-2.1,2])
plt.plot(t,x)
plt.plot(t,integral)
plt.plot(t, [I(i,tc,impulse) for i in t], linestyle='--')
plt.figure()
plt.grid()
plt.plot(x,w,color='tab:blue',marker='o',linewidth=0,markersize=2)
plt.show()
