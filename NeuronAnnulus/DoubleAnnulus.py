from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation, writers
from functools import partial
import numpy as np
import time as tm
from scipy.integrate import solve_ivp
from scipy.stats import linregress
from scipy.sparse import csr_array
import argparse

# Parsing the command line --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-J', type=float,default=0.2,help='Synaptic strenght J')
parser.add_argument('-K', type=float,default=0.4,help='Inhibition strenght K')
parser.add_argument('-N',type=int,default=100,help='Number of neurons per annulus')
parser.add_argument('-T',type=float,default=300,help='Integration final time')
parser.add_argument('--impulses',type=str,default='0.5,0.6',help='Stimuli to each annulus')
args = parser.parse_args()

# Fixed parameters --------------------------------------------------
e = 0.08
a = 0.7
b = 0.8
T = args.T
# Parameters --------------------------------------------------
N = args.N
J = args.J
K = args.K
Stimuli = [float(x) for x in args.impulses.split(',')]
I1 = Stimuli[0]
I2 = Stimuli[1]

# Initial state, jacobian and ODEs of the system --------------------------------------------------
restState = [-1.199408035, -0.624260044, -1.199408035, -0.624260044]

y0 = np.ndarray(4*N, float)
lintime = np.linspace(0,T,int(T*5))
for i in range(0, len(y0), 4):
    y0[i] = restState[0]
    y0[i+1] = restState[1]
    y0[i+2] = restState[2]
    y0[i+3] = restState[3]

def f(t, y):
    yDot = np.ndarray(4*N)
    for i in range(0, len(y), 4):
        if (i == 0):
            yDot[i]=y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[4*N-4]+y[i+4]-2*y[i])-K*(y[i+2]-y[i])+I1
            yDot[i+1]=(y[i]+a-b*y[i+1])*e
            yDot[i+2]=y[i+2]-y[i+2]*y[i+2]*y[i+2]/3-y[i+3]+J*(y[4*N-2]+y[i+6]-2*y[i+2])-K*(y[i]-y[i+2])+I2
            yDot[i+3]=(y[i+2]+a-b*y[i+3])*e
        elif (i == 4*N-4):
            yDot[i]=y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[i-4]+y[0]-2*y[i])-K*(y[i+2]-y[i])+I1
            yDot[i+1]=(y[i]+a-b*y[i+1])*e
            yDot[i+2]=y[i+2]-y[i+2]*y[i+2]*y[i+2]/3-y[i+3]+J*(y[i-2]+y[2]-2*y[i+2])-K*(y[i]-y[i+2])+I2
            yDot[i+3]=(y[i+2]+a-b*y[i+3])*e
        else:
            yDot[i]=y[i]-y[i]*y[i]*y[i]/3-y[i+1]+J*(y[i-4]+y[i+4]-2*y[i])-K*(y[i+2]-y[i])+I1
            yDot[i+1]=(y[i]+a-b*y[i+1])*e
            yDot[i+2]=y[i+2]-y[i+2]*y[i+2]*y[i+2]/3-y[i+3]+J*(y[i-2]+y[i+6]-2*y[i+2])-K*(y[i]-y[i+2])+I2
            yDot[i+3]=(y[i+2]+a-b*y[i+3])*e
    return yDot
def jacobian(t, y):
    indptr = np.array([0,5,7,12,14])
    ind_module = np.array([0,4,5,6,8,4,5,2,2,6,7,10,6,7])
    indices = np.array([0,1,2,4,4*N-4,0,1,0,2,3,6,4*N-2,2,3])
    data = np.array([1-y[0]*y[0]-2*J+K,-1,-K,J,J,1,-b,-K,1-y[2]*y[2]-2*J+K,-1,J,J,1,-b])
    for i in range(1,N-1):
        module = np.array([J,1-y[i*4]*y[i*4]-2*J+K,-1,-K,J,1,-b,J,-K,1-y[i*4+2]*y[i*4+2]-2*J+K,-1,J,1,-b])
        ptr_append = np.array([indptr[-1]+5,indptr[-1]+7,indptr[-1]+12,indptr[-1]+14])
        ind_append = ind_module + 4*(i-1)
        data = np.append(data, module)
        indptr = np.append(indptr, ptr_append)
        indices = np.append(indices, ind_append)
    data = np.append(data,[J,J,1-y[4*N-4]*y[4*N-4]-2*J+K,-1,-K,1,-b,J,J,-K,1-y[4*N-2]*y[4*N-2]-2*J+K,-1,1,-b])
    indices = np.append(indices,[0,4*N-8,4*N-4,4*N-3,4*N-2,4*N-4,4*N-3,2,4*N-6,4*N-4,4*N-2,4*N-1,4*N-2,4*N-1])
    indptr = np.append(indptr,[indptr[-1]+5,indptr[-1]+7,indptr[-1]+12,indptr[-1]+14])
    return csr_array((data, indices, indptr),shape=(4*N,4*N))

# Calculations --------------------------------------------------
t0 = tm.time()
with threadpool_limits(limits=1, user_api='blas'):
    solution = solve_ivp(f,(0,T),y0,'Radau',dense_output=True,jac=jacobian, rtol=0.0001, atol=0.0001, first_step=0.000001)
t1 = tm.time()
state = solution.y
time = solution.t
dense_state = np.array([solution.sol(t) for t in lintime])

# Final drawings --------------------------------------------------
print("Time elapsed during the calculation:", t1 - t0)
print("Time:",T,"s\tTime steps:", len(time))

fig1, ax = plt.subplots(figsize=(4.2,2.1))
fig1.suptitle("50th neurons activities.")
ax.grid()
ax.set_xlabel("Time")
ax.set_ylabel("Voltage")
ax.plot(time, state[200],'k',label="Anello A")
ax.plot(time, state[202],'k--',label="Anello B")
fig1.legend()

# Animation --------------------------------------------------
"""fig3, ax = plt.subplots(figsize=(10.2,5.1))
fig3.suptitle("Soliton wave animation")
line1 = ax.plot()
def init():
    return line1
def update(frame, data):
    text = 'time='+str(lintime[frame])+'\nJ='+str(J)+'K='+str(K)
    ax.clear()
    ax.set_xlim(0, N)
    ax.set_ylim(-5, 5)
    ax.grid()
    ax.text(40,3.5,text)
    ax.plot(data[frame][::4], 'o', markersize=3)
    ax.plot(data[frame][2::4], 'o', markersize=3)
animation = FuncAnimation(fig3, partial(update,data=dense_state),frames=range(len(dense_state)), init_func=init, interval=10)
#Writer = writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=10000)
#animation.save('ConstantImpulse.mp4', writer=writer)
t3 = tm.time()"""
plt.show()
