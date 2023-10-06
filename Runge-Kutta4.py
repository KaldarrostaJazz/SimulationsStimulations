import matplotlib.pyplot as plt
import math

u = 1.5
x = [4.0]
y = [3.0]
t = [0]
dt = 0.1

def f1(x, y, t):
  dxdt = y
  return dxdt

def f2(x ,y ,t):
  dydt = u*(1-x*x)*y - x
  return dydt

for j in range(1000):
  k11 = dt*f1(x[j],y[j],t[j])
  k21 = dt*f2(x[j],y[j],t[j])
  k12 = dt*f1(x[j]+0.5*k11,y[j]+0.5*k21,t[j]+0.5*dt)
  k22 = dt*f2(x[j]+0.5*k11,y[j]+0.5*k21,t[j]+0.5*dt)
  k13 = dt*f1(x[j]+0.5*k12,y[j]+0.5*k22,t[j]+0.5*dt)
  k23 = dt*f2(x[j]+0.5*k12,y[j]+0.5*k22,t[j]+0.5*dt)
  k14 = dt*f1(x[j]+k13,y[j]+k23,t[j]+dt)
  k24 = dt*f2(x[j]+k13,y[j]+k23,t[j]+dt)
  x.append(x[j] + (k11+2*k12+2*k13+k14)/6)
  y.append(y[j] + (k21+2*k22+2*k23+k24)/6)
  t.append(t[j] + dt)

plt.grid()
plt.plot(x,y)
plt.figure()
plt.grid()
plt.plot(t,x)
plt.show()
