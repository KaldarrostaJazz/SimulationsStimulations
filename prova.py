import matplotlib.pyplot as plt
import math

x = [-1]
y = [-1]
t = [0]
N = 2000
dt = 0.01

def f1(x,y,t):
    dxdt = y
    return dxdt

def f2(x,y,t):
    dydt = -x
    return dydt

for i in range(N):
    dx = (y[i] + f2(x[i],y[i],t[i])*dt/4)*dt/2
    dy = - (x[i] + f1(x[i],y[i],t[i])*dt/4)*dt/2
    x.append(x[i] + dx)
    y.append(y[i] + dy)
    t.append((i+1)*dt)

plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.plot(x,y)
plt.figure()
plt.plot(t,x)
plt.plot(t,y)
plt.show()
