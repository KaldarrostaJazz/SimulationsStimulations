from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from functools import partial
import numpy as np
import time as tm
from math import exp
from scipy.integrate import Radau
from scipy.stats import linregress
from scipy.sparse import csr_array

# Fixed parameters --------------------------------------------------
e = 0.01
a = 1.3
T = 25
# Parameters --------------------------------------------------
N = 100
J = 1.5
K = 1.0

# Some utilities --------------------------------------------------
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

# Initial state, jacobian and ODEs of the system --------------------------------------------------
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
    yDot = np.ndarray(4*N)
    for i in range(0, len(y), 4):
        if (i == 0):
            yDot[i]=(y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[4*N-4]-y[i])-K*(y[i+2]-y[i]))/e
            yDot[i+1]=y[i]+a
            yDot[i+2]=(y[i+2]-y[i+2]*y[i+2]*y[i+2]/3-y[i+3]+J*(y[4*N-2]-y[i+2])-K*(y[i]-y[i+2]))/e
            yDot[i+3]=y[i+2]+a
        else:
            yDot[i]=(y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[i-4]-y[i])-K*(y[i+2]-y[i]))/e
            yDot[i+1]=y[i]+a
            yDot[i+2]=(y[i+2]-y[i+2]*y[i+2]*y[i+2]/3-y[i+3]+J*(y[i-2]-y[i+2])-K*(y[i]-y[i+2]))/e
            yDot[i+3]=y[i+2]+a
    return yDot
def jacobian(t, y):
    indptr = np.array([0,4,5,9,10])
    ind_module = np.array([0,4,5,6,4,2,4,6,7,6])
    indices = np.array([0,1,2,4,0,0,2,3,6,2])
    data = np.array([1-y[0]*y[0]-J+K, -1, -K, J, 1, -K, 1-y[2]*y[2]-J+K, -1, J, 1])
    for i in range(N-1):
        module = np.array([J, 1-y[i]*y[i]-J+K, -1, -K, 1, J, -K, 1-y[i+2]*y[i+2]-J+K, -1, 1])
        ptr_append = np.array([indptr[-1]+4,indptr[-1]+5,indptr[-1]+9,indptr[-1]+10])
        ind_append = ind_module + 4*i
        data = np.append(data, module)
        indptr = np.append(indptr, ptr_append)
        indices = np.append(indices, ind_append)
    return csr_array((data,indices,indptr),shape=(4*N,4*N))

# Calculations --------------------------------------------------
t0 = tm.time()
with threadpool_limits(limits=1, user_api='blas'):
    solution = Radau(f, 0, y0, T, rtol=0.0001, atol=0.0001, jac=jacobian, first_step=0.000001)
state = []
time = []
while(True):
    time.append(solution.t)
    state.append(solution.y)
    if (solution.status == 'finished'):
        break
    solution.step()
t1 = tm.time()

# Propagation speed calculations --------------------------------------------------
tMax, sMax = find_max(state)
target = np.array(account_cycles(tMax, tMax[0]))
suppressed = np.array(account_cycles(sMax, sMax[0]))
tRegression = linregress(time, target)
sRegression = linregress(time, suppressed)
t_fit = [t*tRegression.slope + tRegression.intercept for t in time]
s_fit = [t*tRegression.slope + sRegression.intercept for t in time]
t2 = tm.time()

# Final drawings --------------------------------------------------
print("Time elapsed during the calculation:", t1 - t0)
print("Time elapsed for speed calculations:", t2 - t1)
print("Dimesion of the system:\t", solution.n, "\nNumber of neurons per annulus:\t", N)
print("Time:",T,"s\tTime steps:", len(time))
print(tRegression)
print(sRegression)

fig1, (ax1, ax2) = plt.subplots(1, 2)
fig1.suptitle("50th neuron activity")
ax1.grid()
ax1.set_xlabel("Voltage")
ax1.set_ylabel("Spike adptation variable")
ax1.plot([row[200] for row in state], [row[201] for row in state])
ax1.plot([row[202] for row in state], [row[203] for row in state])
ax2.grid()
ax2.set_xlabel("Time")
ax2.set_ylabel("Voltage")
ax2.plot(time, [row[200] for row in state])
ax2.plot(time, [row[202] for row in state])

fig2, (ax3,ax4) = plt.subplots(1,2)
ax3.grid()
ax3.set_title("Wave fronts propagation")
ax3.set_xlabel("Time")
ax3.set_ylabel("Position along the annulus (n+100=n)")
ax3.plot(time, target, marker='o', linewidth=0, color='tab:blue', markersize=1)
ax3.plot(time, t_fit, color='black', linestyle='dashed')
ax3.plot(time, suppressed, marker='o', linewidth=0, color='tab:orange', markersize=1)
ax3.plot(time, s_fit, color='black', linestyle='dashed')
ax4.set_title("Initial conditions")
ax4.set_xlabel("Neuron")
ax4.set_ylabel("Voltage")
ax4.plot(y0[::4], 'x')
ax4.plot(y0[2::4], 'x')

plt.show()

# Animation --------------------------------------------------
fig3, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(-5, 5)
ax.grid()
fig3.suptitle("Soliton wave animation")
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
animation = FuncAnimation(fig3, partial(update,data=state), frames=range(len(state)), init_func=init, interval=10)
Writer = writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
animation.save('SolitonWave.mp4', writer=writer)
t3 = tm.time()
print("Time elapsed to draw the graphs and save the animations:",t3-t2)
#plt.show()
