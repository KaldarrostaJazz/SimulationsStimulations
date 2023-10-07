import matplotlib.pyplot as plt

a = 0.7
b = 0.8
T = 12.5
I_ext = 20
dt = 0.1
tc = 30
N = 3000

def I(time, t_crit):
    if (time == t_crit):
        return I_ext
    else:
        return 0

def f1(x, w, I):
    return x - x*x*x/3 - w + I

def f2(x, w, I):
    return (x + a - b*w)/T

x = [-1.1995]
w = [-0.62427]
t = [0]

for i in range(N):
    k11 = f1(x[i], w[i], I(t[i], tc))*dt
    k12 = f2(x[i], w[i], I(t[i], tc))*dt
    k21 = f1(x[i] + 0.5*k11, w[i] + 0.5*k12, I(t[i], tc))*dt
    k22 = f2(x[i] + 0.5*k11, w[i] + 0.5*k12, I(t[i], tc))*dt
    k31 = f1(x[i] + 0.5*k21, w[i] + 0.5*k22, I(t[i], tc))*dt
    k32 = f2(x[i] + 0.5*k21, w[i] + 0.5*k22, I(t[i], tc))*dt
    k41 = f1(x[i] + k31, w[i] + k32, I(t[i], tc))*dt
    k42 = f2(x[i] + k31, w[i] + k32, I(t[i], tc))*dt
    dx = (k11 + 2*k21 + 2*k31 + k41)/6
    dw = (k12 + 2*k22 + 2*k32 + k42)/6
    x.append(x[i] + dx)
    w.append(w[i] + dw)
    t.append(i*dt)

plt.grid()
plt.ylim([-2.1,2])
plt.plot(t,x)
plt.figure()
plt.grid()
plt.plot(x,w,'bo--', linewidth=1, markersize=1)
plt.show()
