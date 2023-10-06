import matplotlib.pyplot as plt
import math

a = 0.7
b = 1
h = 0.8
T = 12.5
I_ext = 20
x_0 = -1.1995
w_0 = -0.62427
dt = 0.1
tc = 30
N = 3000

def I(time, t_crit):
    if (time == t_crit):
        return I_ext
    else:
        return 0

def f1(x1, w1, x2, w2, I):
    return x1 - x1*x1*x1/3 - w1 + I - b*(x1-x2)


def f2(x1, w1, x2, w2, I):
    return (x1 + a - h*w1)/T

def f3(x1, w1, x2, w2, I):
    return x2 - x2*x2*x2/3 - w2 + b*(x1-x2)

def f4(x1, w1, x2, w2, I):
    return (x2 + a - h*w2)/T

x1 = [x_0]
w1 = [w_0]
x2 = [x_0]
w2 = [w_0]
t = [0]

for i in range(N):
    k11 = f1(x1[i], w1[i], x2[i], w2[i], I(t[i], tc))*dt
    k12 = f2(x1[i], w1[i], x2[i], w2[i], I(t[i], tc))*dt
    k13 = f3(x1[i], w1[i], x2[i], w2[i], I(t[i], tc))*dt
    k14 = f4(x1[i], w1[i], x2[i], w2[i], I(t[i], tc))*dt
    k21 = f1(x1[i] + 0.5*k11, w1[i] + 0.5*k12, x2[i] + 0.5*k13, w2[i] + 0.5*k14, I(t[i], tc))*dt
    k22 = f2(x1[i] + 0.5*k11, w1[i] + 0.5*k12, x2[i] + 0.5*k13, w2[i] + 0.5*k14, I(t[i], tc))*dt
    k23 = f3(x1[i] + 0.5*k11, w1[i] + 0.5*k12, x2[i] + 0.5*k13, w2[i] + 0.5*k14, I(t[i], tc))*dt
    k24 = f4(x1[i] + 0.5*k11, w1[i] + 0.5*k12, x2[i] + 0.5*k13, w2[i] + 0.5*k14, I(t[i], tc))*dt
    k31 = f1(x1[i] + 0.5*k21, w1[i] + 0.5*k22, x2[i] + 0.5*k23, w2[i] + 0.5*k24, I(t[i], tc))*dt
    k32 = f2(x1[i] + 0.5*k21, w1[i] + 0.5*k22, x2[i] + 0.5*k23, w2[i] + 0.5*k24, I(t[i], tc))*dt
    k33 = f3(x1[i] + 0.5*k21, w1[i] + 0.5*k22, x2[i] + 0.5*k23, w2[i] + 0.5*k24, I(t[i], tc))*dt
    k34 = f4(x1[i] + 0.5*k21, w1[i] + 0.5*k22, x2[i] + 0.5*k23, w2[i] + 0.5*k24, I(t[i], tc))*dt
    k41 = f1(x1[i] + k31, w1[i] + k32, x2[i] + k33, w2[i] + k34, I(t[i], tc))*dt
    k42 = f2(x1[i] + k31, w1[i] + k32, x2[i] + k33, w2[i] + k34, I(t[i], tc))*dt
    k43 = f3(x1[i] + k31, w1[i] + k32, x2[i] + k33, w2[i] + k34, I(t[i], tc))*dt
    k44 = f4(x1[i] + k31, w1[i] + k32, x2[i] + k33, w2[i] + k34, I(t[i], tc))*dt
    dx1 = (k11 + 2*k21 + 2*k31 + k41)/6
    dw1 = (k12 + 2*k22 + 2*k32 + k42)/6
    dx2 = (k13 + 2*k23 + 2*k33 + k43)/6
    dw2 = (k14 + 2*k24 + 2*k34 + k44)/6
    x1.append(x1[i] + dx1)
    w1.append(w1[i] + dw1)
    x2.append(x2[i] + dx2)
    w2.append(w2[i] + dw2)
    t.append(i*dt)

plt.grid()
plt.plot(t,x1)
plt.plot(t,x2, "o", markersize=1)
plt.show()
