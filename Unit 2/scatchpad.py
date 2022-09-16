#PCA vs MDS
import numpy as np
x = np.matrix([[1,1],[1,-1],[-1,1]])
B = np.matmul(x, x.transpose())
#PCA of B
w, v = np.linalg.eig(B)
v1 = v[:,0]/np.linalg.norm(v[:,0])
print(v1)