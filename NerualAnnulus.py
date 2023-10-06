import matplotlib.pyplot as plt
import numpy as np
import time as tm
from math import exp
from scipy.integrate import Radau

# Fixed parameters
e = 0.01
a = 1.1
# Parameters
N = 100
J = 1.5
K = 1.1

#Initial state and ODEs of the system
mean = N/2
sigma = N/100
restState = [-a, -a + a*a*a/3]

y0 = np.ndarray(2*N, float)
for i in range(0, len(y0), 2):
    y0[i] = 2*exp(-0.5*(i/2-mean)*(i/2-mean)/(sigma*sigma)) + restState[0]
    y0[i+1] = restState[1]

def f(t, y):
    sysEquations = np.ndarray(2*N)
    for i in range(0, len(y), 2):
        if (i == 0):
            sysEquations[i]=(y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[2*N-2]-y[i]))/e
            sysEquations[i+1]=y[i]+a
        else:
            sysEquations[i]=(y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[i-2]-y[i]))/e
            sysEquations[i+1]=y[i]+a
    return sysEquations

"""elif (i == 2*N-2):
            sysEquations[i]=(y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[i-2]-y[i]+y[0]))/e
            sysEquations[i+1]=y[i]+a"""

#Calculations
t0 = tm.time()

solution = Radau(f, 0, y0, 10, rtol=0.0001, atol=0.0001, first_step=0.000001)
state = []
time = []
stepSize = []
while(True):
    time.append(solution.t)
    state.append(solution.y)
    if (solution.status == 'finished'):
        break
    solution.step()
    stepSize.append(solution.step_size)

t1 = tm.time()

#Propagation speed calculations
maxLocation = []
for i_state in state:
    i_list = i_state.tolist()
    maxLocation.append(i_list.index(max(i_list[::2])))
prev = maxLocation[0]
deltaX = []
for x in maxLocation[1:]:
    if (x < prev):
        deltaX.append((2*N+x-prev)*0.5/N)
    else:
        deltaX.append( (x-prev)*0.5/N)
    prev = x
deltaT = np.array(stepSize)
waveVel = np.divide(deltaX, deltaT)
print(len(maxLocation),maxLocation)
print(len(deltaX),deltaX)
print(len(deltaT),deltaT)
t2 = tm.time()

#Final drawings
i = int(len(time)/4)
print("Time elapsed during the calculation:", t1 - t0)
print("Time elapsed for speed calculations:", t2 - t1)
print("Dimesion of the system:\t", solution.n, "\nNumber of neurons per annulus:\t", N)
print("Time:\t10s\nTime steps:\t", len(time), "\nPlotted at:\t", i)
plt.title("State of the sistem at time "+str(i)+"/"+str(len(time)))
plt.plot(state[i][::2], 'o', markersize=2)
plt.plot(state[i][1::2], 'o', markersize=2)
plt.figure()
plt.title("5th Neuron dynamics")
plt.plot(time, [row[10] for row in state])
plt.plot(time, [row[11] for row in state], '--')
plt.figure()
plt.title("Wavefront propagation speed")
plt.plot(time[1::], waveVel, '+')
plt.figure()
plt.title("Initial conditions")
plt.plot(y0[::2], 'x')
plt.plot(y0[1::2], 'x')
plt.show()
