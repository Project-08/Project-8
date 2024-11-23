import numpy as np 
import matplotlib.pyplot as plt 
import scipy.sparse as sp 

def FDLaplacian2D(steps, stepsize, dimensions):
	diagonal = np.ones(steps)/stepsize
	D = sp.diags([diagonal, -diagonal], [0,-1], shape=(steps,steps-1))
	L = D.T@D

	sub_matrices = list()
	for dim in range(dimensions):
		I1 = sp.eye(int((steps-1)**dim))
		I2 = sp.eye(int((steps-1)**(dimensions-dim-1)))
		sub_matrices.append( sp.kron(sp.kron(I1, L), I2)) 
	return sum(sub_matrices)
plt.spy(FDLaplacian2D(4, 1, 10), marker='o', markerfacecolor='green', markersize=0.25, markeredgecolor='green')
plt.show()
