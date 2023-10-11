import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from functools import partial
import numpy as np
import time as tm
from math import exp
from scipy.integrate import Radau
from scipy.stats import linregress

# Fixed parameters
e = 0.01
a = 0.9
T = 25
# Parameters
N = 100
J = 0.05
K = 1.1

#Some utilities
def find_max(vector_state):
    a = []
    b = []
    for i_state in vector_state:
        i_list = i_state.tolist()
        a.append(i_list.index(max(i_list[::4]))/4)
        b.append(i_list.index(max(i_list[2::4]))/4)
    resultArray = np.array([a,b])
    return resultArray
def account_cycles(vector, prev):
    resultArray = [prev,]
    for x in vector[1::]:
        if (x < prev):
            while (x < prev):
                x += 100
            resultArray.append(x)
        else:
            resultArray.append(x)
        prev = x
    return resultArray

#Initial state and ODEs of the system
mean = N/2
sigma = N/100
restState = [-a, -a + a*a*a/3, -a, -a + a*a*a/3]

y0 = np.ndarray(4*N, float)
for i in range(0, len(y0), 4):
    y0[i] = 2*exp(-0.5*(i/4-mean/2)*(i/4-mean/2)/(sigma*sigma)) + restState[0]
    y0[i+1] = restState[1]
    y0[i+2] = 2*exp(-0.5*(i/4-3*mean/2)*(i/4-3*mean/2)/(sigma*sigma)) + restState[2]
    y0[i+3] = restState[3]

def f(t, y):
    sysEquations = np.ndarray(4*N)
    for i in range(0, len(y), 4):
        if (i == 0):
            sysEquations[i]=(y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[4*N-4]-y[i])-K*(y[i+2]-y[i]))/e
            sysEquations[i+1]=y[i]+a
            sysEquations[i+2]=(y[i+2]-y[i+2]*y[i+2]*y[i+2]/3-y[i+3]+J*(y[4*N-2]-y[i+2])-K*(y[i]-y[i+2]))/e
            sysEquations[i+3]=y[i+2]+a
        else:
            sysEquations[i]=(y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[i-4]-y[i])-K*(y[i+2]-y[i]))/e
            sysEquations[i+1]=y[i]+a
            sysEquations[i+2]=(y[i+2]-y[i+2]*y[i+2]*y[i+2]/3-y[i+3]+J*(y[i-2]-y[i+2])-K*(y[i]-y[i+2]))/e
            sysEquations[i+3]=y[i+2]+a
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
tMax, sMax = find_max(state)
target = np.array(account_cycles(tMax, tMax[0]))
suppressed = np.array(account_cycles(sMax, sMax[0]))
tRegression = linregress(time, target)
sRegression = linregress(time, suppressed)
t_fit = [t*tRegression.slope + tRegression.intercept for t in time]
s_fit = [t*tRegression.slope + sRegression.intercept for t in time]
t2 = tm.time()

#Final drawings
i = int(len(time)/5)
print("Time elapsed during the calculation:", t1 - t0)
print("Time elapsed for speed calculations:", t2 - t1)
print("Dimesion of the system:\t", solution.n, "\nNumber of neurons per annulus:\t", N)
print("Time:\t",T,"s\nTime steps:\t", len(time), "\nPlotted at:\t", i)
print(tRegression)
print(sRegression)
#print("\nRegression result:\nSlope:\t",regression.slope,"\nIntercept:\t", regression.intercept,"\nR-value:\t",regression.rvalue)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.grid()
ax1.set_title("State of the sistem at time "+str(i)+"/"+str(len(time)))
ax1.plot(state[i][::4], 'o', markersize=2)
ax1.plot(state[i][2::4], 'o', markersize=2)
ax2.grid()
ax2.plot(time, [row[20] for row in state])
ax2.plot(time, [row[22] for row in state])
plt.figure()
plt.title("Wave fronts propagation")
plt.grid()
plt.plot(time, target, 'c+')
plt.plot(time, t_fit, 'c')
plt.plot(time, suppressed, 'm+')
plt.plot(time, s_fit, 'm')
#plt.figure()
#plt.title("Initial conditions")
#plt.plot(y0[::4], 'x')
#plt.plot(y0[2::4], 'x')
#plt.show()

#Animation
fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(-5, 5)
ax.grid()
fig.suptitle("Soliton wave animation")
line1 = ax.plot()
def init():
    return line1
def update(frame, data):
    ax.clear()
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 5)
    ax.grid()
    ax.plot(data[frame][::4], 'o', markersize=3)
    ax.plot(data[frame][2::4], 'o', markersize=3)
animation = FuncAnimation(fig, partial(update,data=state), frames=range(len(state)), init_func=init, interval=10)
#Writer = writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
#animation.save('SolitonWave.mp4', writer=writer)
t3 = tm.time()
print("Time elapsed to draw the graphs and save the animations:\t",t3-t2)
plt.show()