import matplotlib.pyplot as plt
import math

I = 0.15
g = 0.3
b = 0.6
T = 150

def F(x):
    return 1/(1+math.exp(-(x-0.1)*10))

def f1(x1, g1, x2, g2):
    return -x1 + F(I-b*x2*g2)

def f2(x1, g1, x2, g2):
    return (1-g1-g*g1*x1)/T

def f3(x1, g1, x2, g2):
    return -x2 + F(I-b*x1*g1)

def f4(x1, g1, x2, g2):
    return (1-g2-g*g2*x2)/T

x1 = [0.9]
g1 = [0.4]
x2 = [0.8]
g2 = [0.3]
time = [0]
dt = 0.1

for i in range(10000):
    k11 = f1(x1[i],g1[i],x2[i],g2[i])*dt
    k12 = f2(x1[i],g1[i],x2[i],g2[i])*dt
    k13 = f3(x1[i],g1[i],x2[i],g2[i])*dt
    k14 = f4(x1[i],g1[i],x2[i],g2[i])*dt
    k21 = f1(x1[i]+0.5*k11,g1[i]+0.5*k12,x2[i]+0.5*k13,g2[i]+0.5*k14)*dt
    k22 = f2(x1[i]+0.5*k11,g1[i]+0.5*k12,x2[i]+0.5*k13,g2[i]+0.5*k14)*dt
    k23 = f3(x1[i]+0.5*k11,g1[i]+0.5*k12,x2[i]+0.5*k13,g2[i]+0.5*k14)*dt
    k24 = f4(x1[i]+0.5*k11,g1[i]+0.5*k12,x2[i]+0.5*k13,g2[i]+0.5*k14)*dt
    k31 = f1(x1[i]+0.5*k21,g1[i]+0.5*k22,x2[i]+0.5*k23,g2[i]+0.5*k24)*dt
    k32 = f2(x1[i]+0.5*k21,g1[i]+0.5*k22,x2[i]+0.5*k23,g2[i]+0.5*k24)*dt
    k33 = f3(x1[i]+0.5*k21,g1[i]+0.5*k22,x2[i]+0.5*k23,g2[i]+0.5*k24)*dt
    k34 = f4(x1[i]+0.5*k21,g1[i]+0.5*k22,x2[i]+0.5*k23,g2[i]+0.5*k24)*dt
    k41 = f1(x1[i]+k31,g1[i]+k32,x2[i]+k33,g2[i]+k34)*dt
    k42 = f2(x1[i]+k31,g1[i]+k32,x2[i]+k33,g2[i]+k34)*dt
    k43 = f3(x1[i]+k31,g1[i]+k32,x2[i]+k33,g2[i]+k34)*dt
    k44 = f4(x1[i]+k31,g1[i]+k32,x2[i]+k33,g2[i]+k34)*dt
    x1.append(x1[i]+(k11+2*k21+2*k31+k41)/6)
    g1.append(g1[i]+(k12+2*k22+2*k32+k42)/6)
    x2.append(x2[i]+(k13+2*k23+2*k33+k43)/6)
    g2.append(g2[i]+(k14+2*k24+2*k34+k44)/6)
    time.append(i*dt)

plt.grid()
plt.title("Andamento temporale delle variabili di stato")
plt.xlabel("tempo")
plt.plot(time, x1, '-c', label='Attività media popolazione 1')
plt.plot(time,x2, '-m', label='Attività media popolazione 2')
plt.plot(time, g1, '--c', label='Depressione sinaptica 1')
plt.plot(time,g2, '--m', label='Depressione sinaptica 2')
plt.legend()
plt.figure()
plt.grid()
plt.title("Piano delle fasi")
plt.xlabel("u1 / g1")
plt.ylabel("u2 / g2")
plt.axis([0,1,0,1])
plt.plot(x1,x2, '-c')
plt.plot(g1,g2, '-m')
plt.figure()
plt.grid()
plt.title("Piano delle fasi (u, g)")
plt.xlabel("Attività media")
plt.ylabel("Depressione sinaptica")
plt.axis([0,1,0,1])
plt.plot(x1,g1, '-c', label='Popolazione 1')
plt.plot(x2,g2, '-m', label='Popolazione 2')
plt.legend()
plt.show()
