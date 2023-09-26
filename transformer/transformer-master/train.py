"""
@author : zwz
@when : 2023-9-5
@homepage : https://github.com/braveniuniu229
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import data



# model parameter setting
batch_size = 5

d_model = 2601
n_layers = 6
n_heads = 1
ffn_hidden = 2048
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
num_epoch = 10
clip = 1.0
weight_decay = 5e-4
inf = float('inf')

train_dataset = data.CustomHDF5Dataset('E:\code\incontextSimulator\dataset.h5', train=True)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomHDF5Dataset('path_to_your_test_dataset', train=False)  # 请确保这里是您的测试集
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


model = Transformer(
                 
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.MSELoss()

num_epochs = 1 

# 在训练开始前定义一个变量来存储最低损失，并初始化为无穷大
min_loss = inf
# 定义模型保存路径
model_save_path = "best_model.pth"

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    total_loss = 0.0
    
    model.train()
    for i, (encoder_T, encoder_vector, decoder_vector, target) in enumerate(train_loader):
        encoder_T, encoder_vector, decoder_vector, target = encoder_T.to(device), encoder_vector.to(device), decoder_vector.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(encoder_T, encoder_vector, decoder_vector,target)
        
        loss = criterion(outputs, target.view(target.size(0),-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 每10个iteration打印一次损失
        if i % 100 == 0:
            print(f"Iter {i}, Loss: {loss.item()}")
    
    model.eval()  # 切换到评估模式
    test_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算
        for i, (encoder_T, encoder_vector, decoder_vector, target) in enumerate(test_loader):
            encoder_T, encoder_vector, decoder_vector, target = encoder_T.to(device), encoder_vector.to(device), decoder_vector.to(device), target.to(device)
            outputs = model(encoder_T, encoder_vector, decoder_vector,target)
            loss = criterion(outputs, target.view(target.size(0),-1))
            test_loss += loss.item()
    
    average_test_loss = test_loss / len(test_loader)
    print(f"Epoch {epoch+1}, Test Loss: {average_test_loss}")
    
    # 比较测试损失，保存在测试集上表现最好的模型
    if average_test_loss < min_test_loss:
        min_test_loss = average_test_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved on test set with loss: {min_test_loss}")
    
    model.train()  # 切换回训练模式

