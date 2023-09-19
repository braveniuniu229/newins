import numpy as np
import scipy 
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
import matplotlib.pylab as plt
import time
from math import sinh
import drawtruth
import h5py

#import matplotlib.pyplot as plt

# Change some default values to make plots more readable on the screen
# 用于打乱，并生成41个坐标
def generate_all_unique_coordinates(nx, ny, group_num=40, per_group=41):
    """
    Generate all unique (x, y) coordinates from a grid of size (nx, ny) for all groups.
    """
    total_coords_needed = group_num * per_group
    coords = [(i, j) for i in range(1, nx + 1) for j in range(1, ny + 1)]
    np.random.shuffle(coords)
    
    # 确保我们有足够的坐标来分配给所有组合
    if len(coords) < total_coords_needed:
        raise ValueError(f"Not enough unique coordinates! Needed: {total_coords_needed}, Got: {len(coords)}")
    
    return [coords[i:i + per_group] for i in range(0, total_coords_needed, per_group)]


def msolver():

    with h5py.File('data.hdf5', 'w') as f:
        # 网格是（0,50）共51个点，边界温度固定，所以要求解的是49*49
        Nx = 50
        xmax=1.0
        ymax=1.0
        h=xmax/Nx
        Ny = int(ymax/h)
        nx = Nx-1
        ny = Ny-1
        n = (nx)*(ny) #未知点数量
        Ttop=10.0
        Tbottom=10.0
        Tleft=10.0
        Tright=10.0
       
        b = np.zeros(n) 
        
        # 初始化T矩阵
        T= np.zeros((Nx + 1,Ny + 1))
        T[-1,:] = Ttop
        T[0,:] = Tbottom
        T[:,0] = Tleft
        T[:,-1] = Tright
        for j in range(1,ny+1):
                        for i in range(1, nx + 1):
                            T[j, i] = 0
        
        
        T_solved= np.zeros((Nx + 1,Ny+1))
        
        
        
        # 初始化拉普拉斯算子，顺便把top边界温度设置好
        for k in range(1,nx):
                        j = k*(ny)
                        i = j - 1
                        
                        b[i] = -Ttop
        
        b[-ny:]+=-Tright  #set the last ny elements to -Tright       
        b[-1]+=-Ttop      #set the last element to -Ttop
        b[0:ny-1]+=-Tleft #set the first ny elements to -Tleft 
        b[0::ny]+=-Tbottom #set every ny-th element to -Tbottom
        # 最外层的参数，大问题定义，共十个类别
        for t in range(10):
            type_group = f.create_group(f"type_{t}")
            # 随机生成四个位置
            random_coords = [(np.random.randint(1, nx+1 ), np.random.randint(1, ny+1)) for _ in range(4)]
            
            
            for p in range(1000):       
                param_group = type_group.create_group(f"param_{p}")
                # 对于每一组确定的位置生成1000种不同的温度
        
                for pos_x, pos_y in random_coords:
                    rand_temp = np.random.uniform(0, 100)
                    T[pos_y, pos_x] = rand_temp
                    b[pos_y + (pos_x-1)*ny - 1]=T[pos_y, pos_x]
                
                d = np.ones(n) # diagonals
                d0 = d*-4
                d1 = d[0:-1]
                d5 = d[0:-ny]

                A = scipy.sparse.diags([d0, d1, d1, d5, d5], [0, 1, -1, ny, -ny], format='csc')
                for k in range(1,nx):
                    A[i, j], A[j, i] = 0, 0
                #alternatively (scalar broadcasting version:)
                #A = scipy.sparse.diags([1, 1, -4, 1, 1], [-5, -1, 0, 1, 5], shape=(15, 15)).toarray()

                # 求解矩阵，二阶微分用拉普拉斯算子代替
            


                
                theta = scipy.sparse.linalg.spsolve(A,b) #theta=sc.linalg.solve_triangular(A,d)
                
                
                
                

                # surfaceplot:
                # x = np.linspace(0, xmax, Nx + 1)
                # y = np.linspace(0, ymax, Ny + 1)

                # X,Y= np.meshgrid(x, y)

                


                # set the imposed boudary values
                T_solved[-1,:] = Ttop
                T_solved[0,:] = Tbottom
                T_solved[:,0] = Tleft
                T_solved[:,-1] = Tright


                for j in range(1,ny+1):
                    for i in range(1, nx + 1):
                        T_solved[j, i] = theta[j + (i-1)*ny - 1]
                
                all_unique_coords = generate_all_unique_coordinates(nx, ny, 40, 41)
                for group_num1 in range(40):
                    grp = param_group.create_group(f"group_{group_num1}")
                    unique_coords=all_unique_coords[group_num1]

                    
                    # 这里是key矩阵的创建
                    key_matrix = np.zeros((5, 45))
                    for col, (x, y) in enumerate(unique_coords + random_coords):
                        key_matrix[0, col] = group_num1
                        key_matrix[1, col] = x
                        key_matrix[2, col] = y
                        key_matrix[3, col] = T[y, x]
                        if col >= 41:
                            key_matrix[4, col] = 1


                    # 创建value矩阵
                    value_matrix = np.zeros((5, 45), dtype=np.complex)
                    for col, (x, y) in enumerate(unique_coords):
                        value_matrix[0, col] = group_num1 + 1j
                        value_matrix[1, col] = x
                        value_matrix[2, col] = y
                        value_matrix[3, col] = T_solved[y, x]  # 可以根据需要更改为其他求解后的值
                    for col, (x, y) in enumerate(random_coords, start=41):
                            value_matrix[1, col] = x
                            value_matrix[2, col] = y
                            value_matrix[3, col] = T_solved[y, x]  # 可以根据需要更改为其他求解后的值
                    

                    # 创建query矩阵
                    query_matrix = key_matrix.copy()
                    query_matrix[3, :] = 0
                    
                    # 将每一组数据写入数据集中
                    
                    grp.create_dataset("key", data=key_matrix)
                    grp.create_dataset("value", data=value_matrix)
                    grp.create_dataset("query", data=query_matrix)
                    
                    print(f"type:{t+1}\tpara:{p+1}\tgroup:{group_num1+1}\n")
