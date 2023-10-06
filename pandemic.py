import matplotlib.pyplot as plt
import math

s = [0.97]
S = [0.97]
i = [0.03]
I = [0.03]
t = [0]
beta = 3
gamma = 1
dt = 0.01
N = 25/dt

def f1(b, g, s):
    dsdt = s*(b*s - b - g*math.log(s))
    return dsdt

for j in range(int(N)):
    ds1 = s[j] + f1(beta, gamma, s[j])*dt/4
    ds = (ds1*(beta*ds1-beta-gamma*math.log(ds1)))*dt/2
    s.append(s[j] + ds)
    i.append(1 - (s[j]+ds) + gamma*math.log((s[j]+ds)/s[0])/beta)
    t.append((j+1)*dt)
    S.append(S[j] - beta*S[j]*I[j]*dt)
    I.append(I[j] + beta*S[j]*I[j]*dt - gamma*I[j]*dt)

plt.plot(t,s)
plt.plot(t,i)
plt.plot(t,S, 'bo', markersize=1)
plt.plot(t,I, 'ro', markersize=1)
plt.figure()
plt.plot(s,i)
plt.plot(S,I, 'bo', markersize=1)
plt.show()
