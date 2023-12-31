import multiprocessing
from threadpoolctl import threadpool_limits
import numpy as np
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

# Initial state, jacobian and ODEs of the system --------------------------------------------------
restState = [-1.199408035, -0.624260044, -1.199408035, -0.624260044]
y0 = np.ndarray(4*N, float)
lintime = np.linspace(0,T,int(T*5),endpoint=False)
for i in range(0, len(y0), 4):
    y0[i] = restState[0]
    y0[i+1] = restState[1]
    y0[i+2] = restState[2]
    y0[i+3] = restState[3]

def f(t, y,I1,I2):
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
def jacobian(t, y,I1,I2):
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
print('I1,I2,T1,E1,T2,E2')
I_ref=0.61
def simulation(I):
    with threadpool_limits(limits=1, user_api='blas'):
        solution = solve_ivp(f,(0,T),y0,'Radau',dense_output=True,args=(I_ref,I),jac=jacobian, rtol=0.0001, atol=0.0001, first_step=0.000001)
        dense_state = np.array([solution.sol(t) for t in lintime])
        periods = []
        for i in range(0,4*N,4):
            state_i = np.array([row[i] for row in dense_state])
            time_series = []
            for j in range(1,state_i.size-1):
                if (state_i[j]>state_i[j-1] and state_i[j]>state_i[j+1]):
                    time_series.append(lintime[j])
                elif (state_i[j]<state_i[j+1] and state_i[j]<state_i[j-1]):
                      time_series.append(lintime[j])
            a = np.mean(np.diff(time_series)[::2])
            b = np.mean(np.diff(time_series)[1::2])
            periods.append([a,b])
        means = [np.mean([row[0] for row in periods]),np.mean([row[1] for row in periods])]
        std_devs = [np.std([row[0] for row in periods]),np.std([row[1] for row in periods])]
        print(I_ref,I,means[0],std_devs[0],means[1],std_devs[1],sep=',')
        return means
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=None)
    I_range = np.linspace(0.05,0.60,55,endpoint=False)
    results = pool.map(simulation,I_range)
    pool.close()
    pool.join()