import numpy as np
import h5py
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

X, Y = np.meshgrid(x, y)

T = np.zeros_like(X)

def add_gaussian_heat(x, y, amplitude, sigma, b_vector):
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            r = np.sqrt((i - x)**2 + (j - y)**2)
            b_vector[j + (i-1)*ny - 1] -= amplitude * np.exp(-r**2/(2*sigma**2))
    return b_vector
n_samples_per_type = 10000

#这里是选择的16个观测点
points_indices = np.random.choice(Nx*Ny, 16, replace=False)
points_coordinates = [(idx // Nx, idx % Ny) for idx in points_indices]


# 用于生成一个t样本
def generate_sample(type_points):
    b_local = np.zeros(nx * ny)
    # 这四个点各放置一个相同的高斯热源
    amplitude = np.random.uniform(10, 50)
    sigma = np.random.uniform(0.5, 3)
    for point in type_points:
        x, y = point
        
        b_local = add_gaussian_heat(x, y, amplitude, sigma, b_local)
    theta = scipy.sparse.linalg.spsolve(A, b_local)
    T[-1,:] = Ttop
    T[0,:] = Tbottom
    T[:,0] = Tleft
    T[:,-1] = Tright

    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            T[j, i] = theta[j + (i-1)*ny - 1]
    vector = np.array([T[coord[0], coord[1]] for coord in points_coordinates])
    return T, vector

with h5py.File('dataset.h5', 'w') as f:
    for type_idx in range(5):
        type_points = [tuple(np.random.randint(0, Nx, 2)) for _ in range(4)]
        type_grp = f.create_group(f'type{type_idx}')
        n_xl_per_type = n_samples_per_type // 5
        for xl_idx in range(n_samples_per_type // 5):
            xl_grp = type_grp.create_group(f'xl{xl_idx}')
            encoder_grp = xl_grp.create_group('encoder')
            decoder_grp = xl_grp.create_group('decoder')



            for sample_idx_within_xl in range(5):
                T, vector = generate_sample(type_points)
                if sample_idx_within_xl < 4:  # encoder samples
                    sample_grp = encoder_grp.create_group(f'sample{sample_idx_within_xl}')
                else:  # decoder sample
                    sample_grp = decoder_grp.create_group(f'sample{sample_idx_within_xl}')
                
                sample_grp.create_dataset('T', data=T)
                sample_grp.create_dataset('vector', data=vector)

            if (xl_idx + 1) % 10 == 0:
                print(f"Type {type_idx + 1}, generated {xl_idx + 1} out of {n_xl_per_type} xl groups")