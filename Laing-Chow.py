import matplotlib.pyplot as plt
import math

b = 0.4
a = 0.2
Ta = 20
Tg = 40
dt = 0.1

def H(x):
    if (x > 0):
        return 1
    else:
        return 0
def I1(t):
    return 0.5
def I2(t):
    return 0.3
    
def f1(I1, I2, u1, a1, g1, u2, a2, g2):
    return -u1 + H(I1 + a*u1*g1 - b*u2*g2 - a1)
def f2(I1, I2, u1, a1, g1, u2, a2, g2):
    return (-a1 + 0.6*H(I1 + a*u1*g1 - b*u2*g2 - a1))/Ta
def f3(I1, I2, u1, a1, g1, u2, a2, g2):
    return (1 - g1*(1 + 0.6*H(I1 + a*u1*g1 - b*u2*g2 - a1)))/Tg
def f4(I1, I2, u1, a1, g1, u2, a2, g2):
    return -u2 + H(I2 + a*u2*g2 - b*u1*g1 - a2)
def f5(I1, I2, u1, a1, g1, u2, a2, g2):
    return (-a2 + 0.6*H(I2 + a*u2*g2 - b*u1*g1 - a2))/Ta
def f6(I1, I2, u1, a1, g1, u2, a2, g2):
    return (1 - g2*(1 + 0.6*H(I2 + a*u2*g2 - b*u1*g1 - a2)))/Tg

u1 = [0.8]
a1 = [0.2]
g1 = [0.1]
u2 = [0.4]
a2 = [0.1]
g2 = [0.2]
time = [0]

for i in range(1500):
    k11 = f1(I1(i*dt), I2(i*dt), u1[i], a1[i], g1[i], u2[i], a2[i], g2[i])*dt
    k12 = f2(I1(i*dt), I2(i*dt), u1[i], a1[i], g1[i], u2[i], a2[i], g2[i])*dt
    k13 = f3(I1(i*dt), I2(i*dt), u1[i], a1[i], g1[i], u2[i], a2[i], g2[i])*dt
    k14 = f4(I1(i*dt), I2(i*dt), u1[i], a1[i], g1[i], u2[i], a2[i], g2[i])*dt
    k15 = f5(I1(i*dt), I2(i*dt), u1[i], a1[i], g1[i], u2[i], a2[i], g2[i])*dt
    k16 = f6(I1(i*dt), I2(i*dt), u1[i], a1[i], g1[i], u2[i], a2[i], g2[i])*dt
    k21 = f1(I1(i*dt), I2(i*dt), u1[i]+0.5*k11, a1[i]+0.5*k12, g1[i]+0.5*k13, u2[i]+0.5*k14, a2[i]+0.5*k15, g2[i]+0.5*k16)*dt
    k22 = f2(I1(i*dt), I2(i*dt), u1[i]+0.5*k11, a1[i]+0.5*k12, g1[i]+0.5*k13, u2[i]+0.5*k14, a2[i]+0.5*k15, g2[i]+0.5*k16)*dt
    k23 = f3(I1(i*dt), I2(i*dt), u1[i]+0.5*k11, a1[i]+0.5*k12, g1[i]+0.5*k13, u2[i]+0.5*k14, a2[i]+0.5*k15, g2[i]+0.5*k16)*dt
    k24 = f4(I1(i*dt), I2(i*dt), u1[i]+0.5*k11, a1[i]+0.5*k12, g1[i]+0.5*k13, u2[i]+0.5*k14, a2[i]+0.5*k15, g2[i]+0.5*k16)*dt
    k25 = f5(I1(i*dt), I2(i*dt), u1[i]+0.5*k11, a1[i]+0.5*k12, g1[i]+0.5*k13, u2[i]+0.5*k14, a2[i]+0.5*k15, g2[i]+0.5*k16)*dt
    k26 = f6(I1(i*dt), I2(i*dt), u1[i]+0.5*k11, a1[i]+0.5*k12, g1[i]+0.5*k13, u2[i]+0.5*k14, a2[i]+0.5*k15, g2[i]+0.5*k16)*dt
    k31 = f1(I1(i*dt), I2(i*dt), u1[i]+0.5*k21, a1[i]+0.5*k22, g1[i]+0.5*k23, u2[i]+0.5*k24, a2[i]+0.5*k25, g2[i]+0.5*k26)*dt
    k32 = f2(I1(i*dt), I2(i*dt), u1[i]+0.5*k21, a1[i]+0.5*k22, g1[i]+0.5*k23, u2[i]+0.5*k24, a2[i]+0.5*k25, g2[i]+0.5*k26)*dt
    k33 = f3(I1(i*dt), I2(i*dt), u1[i]+0.5*k21, a1[i]+0.5*k22, g1[i]+0.5*k23, u2[i]+0.5*k24, a2[i]+0.5*k25, g2[i]+0.5*k26)*dt
    k34 = f4(I1(i*dt), I2(i*dt), u1[i]+0.5*k21, a1[i]+0.5*k22, g1[i]+0.5*k23, u2[i]+0.5*k24, a2[i]+0.5*k25, g2[i]+0.5*k26)*dt
    k35 = f5(I1(i*dt), I2(i*dt), u1[i]+0.5*k21, a1[i]+0.5*k22, g1[i]+0.5*k23, u2[i]+0.5*k24, a2[i]+0.5*k25, g2[i]+0.5*k26)*dt
    k36 = f6(I1(i*dt), I2(i*dt), u1[i]+0.5*k21, a1[i]+0.5*k22, g1[i]+0.5*k23, u2[i]+0.5*k24, a2[i]+0.5*k25, g2[i]+0.5*k26)*dt
    k41 = f1(I1(i*dt), I2(i*dt), u1[i]+k31, a1[i]+k32, g1[i]+k33, u2[i]+k34, a2[i]+k35, g2[i]+k36)*dt
    k42 = f2(I1(i*dt), I2(i*dt), u1[i]+k31, a1[i]+k32, g1[i]+k33, u2[i]+k34, a2[i]+k35, g2[i]+k36)*dt
    k43 = f3(I1(i*dt), I2(i*dt), u1[i]+k31, a1[i]+k32, g1[i]+k33, u2[i]+k34, a2[i]+k35, g2[i]+k36)*dt
    k44 = f4(I1(i*dt), I2(i*dt), u1[i]+k31, a1[i]+k32, g1[i]+k33, u2[i]+k34, a2[i]+k35, g2[i]+k36)*dt
    k45 = f5(I1(i*dt), I2(i*dt), u1[i]+k31, a1[i]+k32, g1[i]+k33, u2[i]+k34, a2[i]+k35, g2[i]+k36)*dt
    k46 = f6(I1(i*dt), I2(i*dt), u1[i]+k31, a1[i]+k32, g1[i]+k33, u2[i]+k34, a2[i]+k35, g2[i]+k36)*dt
    u1.append(u1[i]+(k11+2*k21+2*k31+k41)/6)
    a1.append(a1[i]+(k12+2*k22+2*k32+k42)/6)
    g1.append(g1[i]+(k13+2*k23+2*k33+k43)/6)
    u2.append(u2[i]+(k14+2*k24+2*k34+k44)/6)
    a2.append(a2[i]+(k15+2*k25+2*k35+k45)/6)
    g2.append(g2[i]+(k16+2*k26+2*k36+k46)/6)
    time.append(i*dt)

plt.grid()
#plt.axis([0,1000*dt,0,1])
plt.plot(time,u1)
plt.plot(time,u2)
plt.show()