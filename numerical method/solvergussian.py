import numpy as np
import scipy 
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pylab as plt
import time

# Change some default values to make plots more readable on the screen


# Set temperature at the top
Ttop=0
Tbottom=0
Tleft=0
Tright=0
xmax=1
ymax=1

# Set simulation parameters
Nx = 50
Ny=50
nx = Nx-1
ny = Ny-1
n = nx*ny

d = np.ones(n) # diagonals
b = np.zeros(n) #RHS
d0 = d*-4
d1 = d[0:-1]
d5 = d[0:-ny]
A = scipy.sparse.diags([d0, d1, d1, d5, d5], [0, 1, -1, ny, -ny], format='csc')

# Add Gaussian heat sources to b
def add_gaussian_heat(x, y, amplitude, sigma):
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            r = np.sqrt((i - x)**2 + (j - y)**2)
            b[j + (i-1)*ny - 1] += amplitude * np.exp(-r**2/(2*sigma**2))
    return b

# Randomly generate four Gaussian sources
for _ in range(4):
    x, y = np.random.randint(0, nx), np.random.randint(0, ny)
    sigma = 1.0
    amplitude = 10.0
    b = add_gaussian_heat(x, y, amplitude, sigma)

# Modify A matrix and b for boundary conditions
for k in range(1, nx):
    j = k * ny
    i = j - 1
    A[i, j], A[j, i] = 0, 0
    b[i] = -Ttop

b[-ny:] -= Tright
b[-1] -= Ttop
b[0:ny-1] -= Tleft
b[::ny] -= Tbottom

# Solve
theta = scipy.sparse.linalg.spsolve(A, b)

# Extract temperature to matrix T
x = np.linspace(0, xmax, Nx + 1)
y = np.linspace(0, ymax, Ny + 1)
X, Y = np.meshgrid(x, y)
T = np.zeros_like(X)
T[-1, :] = Ttop
T[0, :] = Tbottom
T[:, 0] = Tleft
T[:, -1] = Tright

for j in range(1, ny + 1):
    for i in range(1, nx + 1):
        T[j, i] = theta[j + (i-1)*ny - 1]

# Plotting
plt.imshow(T, cmap='hot', origin='lower')
plt.colorbar()
plt.show()
