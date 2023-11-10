import multiprocessing
from threadpoolctl import threadpool_limits
import numpy as np
import time as tm
from scipy.integrate import solve_ivp
from scipy.sparse import csr_array
import argparse

# Parsing the command line --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-J', type=float,default=0.2,help='Synaptic strenght J')
parser.add_argument('-K', type=float,default=0.4,help='Inhibition strenght K')
parser.add_argument('-N',type=int,default=100,help='Number of neurons per annulus')
parser.add_argument('-T',type=float,default=250,help='Integration final time')
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
I1 = 0.1
I2 = 0.11

# Initial state, jacobian and ODEs of the system --------------------------------------------------
restState = [-1.199408035, -0.624260044, -1.199408035, -0.624260044]
y0 = np.ndarray(4*N, float)
lintime = np.linspace(0,T,int(T*5))
for i in range(0, len(y0), 4):
    y0[i] = restState[0]
    y0[i+1] = restState[1]
    y0[i+2] = restState[2]
    y0[i+3] = restState[3]