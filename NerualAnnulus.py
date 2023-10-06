import matplotlib.pyplot as plt
import numpy as np
import time as tm
from math import exp
from scipy.integrate import Radau
from scipy.stats import linregress

# Fixed parameters
e = 0.01
a = 1.1
T = 10
# Parameters
N = 100
J = 0.05
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

solution = Radau(f, 0, y0, T, rtol=0.0001, atol=0.0001, first_step=0.000001)
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
    maxLocation.append((i_list.index(max(i_list[::2])))/2)
"""for i in range(len(maxLocation)-1):
    if (maxLocation[i+1] < maxLocation[i]):
        if (maxLocation[i] - maxLocation[i+1] < 101):
            maxLocation[i+1] += N
        else:
            maxLocation[i+1] += 2*N
    else:
        maxLocation[i+1] += 0
regression = linregress(time, maxLocation)
x_fit = [t*regression.slope + regression.intercept for t in time]"""
t2 = tm.time()

#Final drawings
i = int(len(time)/4)
print("Time elapsed during the calculation:", t1 - t0)
print("Time elapsed for speed calculations:", t2 - t1)
print("Dimesion of the system:\t", solution.n, "\nNumber of neurons per annulus:\t", N)
print("Time:\t10s\nTime steps:\t", len(time), "\nPlotted at:\t", i)
#print("\nRegression result:\nSlope:\t",regression.slope,"\nIntercept:\t", regression.intercept,"\nR-value:\t",regression.rvalue)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("State of the sistem at time "+str(i)+"/"+str(len(time)))
ax1.plot(state[i][::2], 'o', markersize=2)
ax1.plot(state[i][1::2], 'o', markersize=2)
ax2.set_title("5th Neuron dynamics")
ax2.plot(time, [row[10] for row in state])
ax2.plot(time, [row[11] for row in state], '--')
plt.figure()
plt.title("Wavefront propagation speed")
plt.plot(time, maxLocation, '+')
#plt.plot(time, x_fit)
#plt.figure()
#plt.title("Initial conditions")
#plt.plot(y0[::2], 'x')
#plt.plot(y0[1::2], 'x')
plt.show()
