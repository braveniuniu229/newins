import matplotlib
import matplotlib.pylab as plt
import time
from math import sinh
import solvergussian
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


Nx, Ny = 50, 50
nx, ny = Nx - 1, Ny - 1
n = nx * ny

d = np.ones(n) # diagonals
b = np.zeros(n) #RHS
d0 = d * -4
d1 = d[0:-1]
d5 = d[0:-ny]
A = scipy.sparse.diags([d0, d1, d1, d5, d5], [0, 1, -1, ny, -ny], format='csc')
Ttop=0
Tbottom=0
Tleft=0
Tright=0

xmax=1.0
ymax=1.0
x = np.linspace(0, xmax, Nx + 1)
y = np.linspace(0, ymax, Ny + 1)
type_points = [tuple(np.random.randint(0, Nx, 2)) for _ in range(4)]
T,_=solvergussian.generate_sample(type_points)

def draw(T):
    plt.imshow(T, cmap='hot', origin='lower')
    plt.colorbar(label='Temperature')
    plt.title('Steady-State Temperature Distribution')
    plt.show()

draw(T)