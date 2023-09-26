import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import data

train_dataset = data.CustomHDF5Dataset('E:\code\incontextSimulator\dataset.h5', train=True)
train_loader = data.DataLoader(train_dataset, batch_size=5, shuffle=True)
i=0
# 迭代 DataLoader
for encoder_T, encoder_vector, decoder_vector, target in train_loader:
    # 打印数据的形状和一些具体的数据值
    # print("encoder_T shape:", encoder_T.shape)
    # print("encoder_vector shape:", encoder_vector.shape)
    # print("decoder_vector shape:", decoder_vector.shape)
    # print("target shape:", target.shape)
    
    # print("encoder_T data:", encoder_T)
    # print("encoder_vector data:", encoder_vector)
    # print("decoder_vector data:", decoder_vector)
    # print("target data:", target)
    
    # 由于我们只是验证数据，所以我们可以在检查第一批数据后终止循环。
   i+=1
   print(i)

