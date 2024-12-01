import numpy as np 
import matplotlib.pyplot as plt 
import scipy.sparse as sp 
import scipy.sparse.linalg as la 
from multiprocessing import Pool



def FDLaplacianND(steps, stepsize, dimensions):
	diagonal = np.ones(steps)/stepsize
	D = sp.diags([diagonal, -diagonal], [0,-1], shape=(steps,steps-1))
	L = D.T@D

	sub_matrices = list()
	for dim in range(dimensions):
		I1 = sp.eye(int((steps-1)**dim))
		I2 = sp.eye(int((steps-1)**(dimensions-dim-1)))
		sub_matrices.append( sp.kron(sp.kron(I1, L), I2)) 
	return sum(sub_matrices)

def func(x,y,z):
	return -np.pi*np.pi*x*y*np.sin(np.pi*z)

def Get_3D_source_func(func, steps, stepsize):
	result = np.zeros((steps-1)**3)
	for z in range(1,steps):
		for y in range(1,steps):
			for x in range(1,steps):
				result[(x-1)+(steps-1)*(y-1)+(steps-1)*(steps-1)*(z-1)] = func(x*stepsize,y*stepsize,z*stepsize) 
				if(y==(steps-1)): result[(x-1)+(steps-1)*(y-1)+(steps-1)*(steps-1)*(z-1)] =+  x*stepsize*np.sin(np.pi*z*stepsize)/(stepsize**2)  #BC
				if(x==(steps-1)): result[(x-1)+(steps-1)*(y-1)+(steps-1)*(steps-1)*(z-1)] =+  y*stepsize*np.sin(np.pi*z*stepsize)/(stepsize**2)   #BC

	return result

def Get_3D_analytical(steps, stepsize):
	result = np.zeros((steps-1)**3)
	for z in range(1,steps):
		for y in range(1,steps):
			for x in range(1,steps):
				result[(x-1)+(steps-1)*(y-1)+(steps-1)*(steps-1)*(z-1)] = stepsize*stepsize*x*y*np.sin(np.pi*z*stepsize)
	return result

def Get_error(size):
	n = size
	i = size
	u_num = la.spsolve(FDLaplacianND(n, 1/n, 3), Get_3D_source_func(func, n, 1/n))
	u_exact = Get_3D_analytical(n, 1/n)
	squares = (u_num-u_exact)**2
	mean = np.mean(squares)
	root = np.sqrt(mean)
	return root

def Generate_plot():
	steps_array = np.array([4,8,16,32,64,128,256])
	with Pool() as pool:
		RMS = pool.map(Get_error, steps_array)
	plt.semilogy(steps_array,RMS, label="RMS")
	plt.legend()
	#plt.show()
	plt.savefig("3D Error until N=60")
	#return array
if __name__ == "__main__":
	Generate_plot()
