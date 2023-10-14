import numpy as np
from scipy.sparse import csr_array

J = 0.05
K = 1.5
a = 1 -1.3*1.3 - J + K
N = 10
indptr = np.array([0,4,5,9,10])
ind_module = np.array([0,4,5,6,4,2,4,6,7,6])
indices = np.array([0,1,2,4,0,0,2,3,6,2])
module = np.array([J,a,-1,-K,1,J,-K,a,-1,1])
data = np.array([a,-1,-K,J,1,-K,a,-1,J,1])
for i in range(N-1):
    ptr_append = np.array([indptr[-1]+4,
        indptr[-1]+5,indptr[-1]+9,indptr[-1]+10])
    ind_append = ind_module + 4*i
    data = np.append(data, module)
    indptr = np.append(indptr, ptr_append)
    indices = np.append(indices, ind_append)
matrix = csr_array((data,indices,indptr),shape=(4*N,4*N))
print(matrix)
print(matrix.toarray())
