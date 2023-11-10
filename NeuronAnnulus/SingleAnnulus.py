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
e = 0.08
a = 0.7
b = 0.8
T = 200
# Parameters --------------------------------------------------
N = 100
J = 0.5

# Some utilities --------------------------------------------------
def find_max(vector_state):
    a = []
    for i_state in vector_state:
        i_list = i_state.tolist()
        a.append(i_list.index(max(i_list[::2]))/2)
    resultArray = np.array(a)
    return resultArray
def account_cycles(vector, prev):
    resultArray = [prev,]
    for x in vector[1::]:
        if (x < prev):
            while (x < prev):
                x += N
            resultArray.append(x)
        else:
            resultArray.append(x)
        prev = x
    return resultArray

# Initial state, jacobian and ODEs of the system --------------------------------------------------
mean = N/2
sigma = N/100
restState = [-1.199408035, -0.624260044]

y0 = np.ndarray(2*N, float)
for i in range(0, len(y0), 2):
    y0[i] = 2*exp(-0.5*(i/2-mean)*(i/2-mean)/(sigma*sigma)) + restState[0]
    y0[i+1] = restState[1]
def f(t, y):
    yDot = np.ndarray(2*N)
    for i in range(0, len(y), 2):
        if (i == 0):
            yDot[i]=y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[2*N-2]-y[i])
            yDot[i+1]=(y[i]+a-b*y[i+1])*e
        else:
            yDot[i]=y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[i-2]-y[i])
            yDot[i+1]=(y[i]+a-b*y[i+1])*e
    return yDot
def jacobian(t, y):
    indptr = np.array([0,3,5])
    ind_module = np.array([0,2,3,2,3])
    indices = np.array([0,1,2,0,1])
    data = np.array([1-y[0]*y[0]-J, -1, J, 1,-b])
    for i in range(N-1):
        module = np.array([J, 1-y[i]*y[i]-J, -1, 1,-b])
        ptr_append = np.array([indptr[-1]+3,indptr[-1]+5])
        ind_append = ind_module + 2*i
        data = np.append(data, module)
        indptr = np.append(indptr, ptr_append)
        indices = np.append(indices, ind_append)
    return csr_array((data,indices,indptr),shape=(2*N,2*N))

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
tMax = find_max(state)
target = np.array(account_cycles(tMax, tMax[0]))
tRegression = linregress(time, target)
t_fit = [t*tRegression.slope + tRegression.intercept for t in time]
t2 = tm.time()

# Final drawings --------------------------------------------------
print("Time elapsed during the calculation:", t1 - t0)
print("Time elapsed for speed calculations:", t2 - t1)
print("Dimesion of the system:\t", solution.n, "\nNumber of neurons per annulus:\t", N)
print("Time:",T,"s\tTime steps:", len(time))
print(tRegression)

fig1, (ax1, ax2) = plt.subplots(1, 2)
fig1.suptitle("75th neuron activity")
ax1.grid()
ax1.set_xlabel("Voltage")
ax1.set_ylabel("Spike adptation variable")
ax1.plot([row[150] for row in state], [row[151] for row in state])
ax2.grid()
ax2.set_xlabel("Time")
ax2.set_ylabel("Voltage")
ax2.plot(time, [row[150] for row in state])

fig2, (ax3,ax4) = plt.subplots(1,2)
ax3.grid()
ax3.set_title("Wave fronts propagation")
ax3.set_xlabel("Time")
ax3.set_ylabel("Position along the annulus (n+N=n)")
ax3.plot(time, target, marker='o', linewidth=0, color='tab:blue', markersize=1)
ax3.plot(time, t_fit, color='black', linestyle='dashed')
ax4.set_title("Initial conditions")
ax4.set_xlabel("Neuron")
ax4.set_ylabel("Voltage")
ax4.plot(y0[::2], 'x')

#plt.show()

# Animation --------------------------------------------------
fig3, ax = plt.subplots()
ax.set_xlim(0, N)
ax.set_ylim(-5, 5)
ax.grid()
fig3.suptitle("Soliton wave animation")
line1 = ax.plot()
def init():
    return line1
def update(frame, data):
    ax.clear()
    ax.set_xlim(0, N)
    ax.set_ylim(-5, 5)
    ax.grid()
    ax.plot(data[frame][::2], 'o', markersize=3)
    ax.plot(data[frame][1::2], 'o', markersize=3)
animation = FuncAnimation(fig3, partial(update,data=state), frames=range(len(state)), init_func=init, interval=10)
#Writer = writers['pillow']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
#animation.save('SolitonWave.gif', writer=writer)
t3 = tm.time()
print("Time elapsed to draw the graphs and save the animations:",t3-t2)
plt.show()
