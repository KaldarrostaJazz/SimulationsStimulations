import matplotlib.pyplot as plt

x = [4]
y = [1.5]
t = [0]
dt = 0.01
N = 2000

def f1(x, y, t):
    dxdt = x*(1-0.5*y)
    return dxdt

def f2(x, y, t):
    dydt = -y*(1.5-0.7*x)
    return dydt

for i in range(N):
    dx = ((x[i] + f1(x[i],y[i],t[i])*dt/4) * (1-0.5*(y[i] + f2(x[i],y[i],t[i])*dt/4))) * dt/2
    dy = (- (y[i] + f2(x[i],y[i],t[i])*dt/4) * (1.5-0.7*(x[i] + f1(x[i],y[i],t[i])*dt/4))) * dt/2
    x.append(x[i] + dx)
    y.append(y[i] + dy)
    t.append((i+1)*dt)

plt.plot(x,y)
plt.figure()
plt.plot(t,x)
plt.plot(t,y)
plt.show()
