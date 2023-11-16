import matplotlib.pyplot as plt
import math

I = 1.75
g = 0.5
b = 1.1
T = 100
#u = 0.8569
#A = 0.5*(99-1100*u+math.sqrt(9801-219800*u+1210000*u*u))
#B = 0.5*(99-1100*u-math.sqrt(9801-219800*u+1210000*u*u))

def F(x):
     return 1/(1+math.exp(-(x-0.2)*10))

def f1(x1, a1, x2, a2):
    return -x1 + F(I-b*x2-g*a1)

def f2(x1, a1, x2, a2):
    return (-a1 + x1)/T

def f3(x1, a1, x2, a2):
    return -x2 + F(I-b*x1-g*a2)

def f4(x1, a1, x2, a2):
    return (-a2 + x2)/T

x1 = [0.9]
a1 = [0.4]
x2 = [0.8]
a2 = [0.3]
time = [0]
dt = 0.1

for i in range(10000):
    k11 = f1(x1[i],a1[i],x2[i],a2[i])*dt
    k12 = f2(x1[i],a1[i],x2[i],a2[i])*dt
    k13 = f3(x1[i],a1[i],x2[i],a2[i])*dt
    k14 = f4(x1[i],a1[i],x2[i],a2[i])*dt
    k21 = f1(x1[i]+0.5*k11,a1[i]+0.5*k12,x2[i]+0.5*k13,a2[i]+0.5*k14)*dt
    k22 = f2(x1[i]+0.5*k11,a1[i]+0.5*k12,x2[i]+0.5*k13,a2[i]+0.5*k14)*dt
    k23 = f3(x1[i]+0.5*k11,a1[i]+0.5*k12,x2[i]+0.5*k13,a2[i]+0.5*k14)*dt
    k24 = f4(x1[i]+0.5*k11,a1[i]+0.5*k12,x2[i]+0.5*k13,a2[i]+0.5*k14)*dt
    k31 = f1(x1[i]+0.5*k21,a1[i]+0.5*k22,x2[i]+0.5*k23,a2[i]+0.5*k24)*dt
    k32 = f2(x1[i]+0.5*k21,a1[i]+0.5*k22,x2[i]+0.5*k23,a2[i]+0.5*k24)*dt
    k33 = f3(x1[i]+0.5*k21,a1[i]+0.5*k22,x2[i]+0.5*k23,a2[i]+0.5*k24)*dt
    k34 = f4(x1[i]+0.5*k21,a1[i]+0.5*k22,x2[i]+0.5*k23,a2[i]+0.5*k24)*dt
    k41 = f1(x1[i]+k31,a1[i]+k32,x2[i]+k33,a2[i]+k34)*dt
    k42 = f2(x1[i]+k31,a1[i]+k32,x2[i]+k33,a2[i]+k34)*dt
    k43 = f3(x1[i]+k31,a1[i]+k32,x2[i]+k33,a2[i]+k34)*dt
    k44 = f4(x1[i]+k31,a1[i]+k32,x2[i]+k33,a2[i]+k34)*dt
    x1.append(x1[i]+(k11+2*k21+2*k31+k41)/6)
    a1.append(a1[i]+(k12+2*k22+2*k32+k42)/6)
    x2.append(x2[i]+(k13+2*k23+2*k33+k43)/6)
    a2.append(a2[i]+(k14+2*k24+2*k34+k44)/6)
    time.append(i*dt)

#y1 = []
#y2 = []
#y3 = []
#y4 = []
#for i in range(len(x1)):
#    y1.append(A*x1[i] - x1[i] - A* x2[i] + a2[i])
#    y2.append(B*x1[i] - x1[i] - B* x2[i] + a2[i])
#    y3.append(A*x1[i] + x1[i] + A* x2[i] + a2[i])
#    y4.append(B*x1[i] + x1[i] + B* x2[i] + a2[i])

#plt.grid()
#plt.plot(time, y1)
#plt.plot(time, y2)
#plt.plot(time, y3)
#plt.plot(time, y4)
#plt.figure()
#plt.grid()
#plt.plot(y1,y3)
#plt.plot(y2,y4)
#plt.figure()

plt.grid()
plt.title("Andamento temporale delle variabili di stato")
plt.xlabel("tempo")
plt.plot(time, x1, '-c', label='Attività media popolazione 1')
plt.plot(time,x2, '-m', label='Attività media popolazione 2')
plt.plot(time, a1, '--c', label='Adattamento popolazione 1')
plt.plot(time,a2, '--m', label='Adattamento popolazione 2')
plt.legend()
plt.figure()
plt.grid()
plt.title("Piano delle fasi")
plt.xlabel("u1 / a1")
plt.ylabel("u2 / a2")
plt.axis([0,1,0,1])
plt.plot(x1,x2, '-c')
plt.plot(a1,a2, '-m')
plt.figure()
plt.grid()
plt.title("Piano delle fasi (u, a)")
plt.xlabel("Attività media")
plt.ylabel("Adattamento")
plt.axis([0,1,0,1])
plt.plot(x1,a1, '-c', label='Popolazione 1')
plt.plot(x2,a2, '-m', label='Popolazione 2')
plt.legend()
plt.show()
