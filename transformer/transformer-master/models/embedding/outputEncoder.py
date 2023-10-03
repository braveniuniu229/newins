"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
import torch.nn.functional as F


class outputEncoder(nn.Module):
    def __init__(self):
        super(outputEncoder,self).__init__()
    
    def forward(self,encoder_vector,encoder_T,decoder_vector):
        decoder_vector=decoder_vector.unsqueeze(1)
        mean_encv = torch.mean(encoder_vector, dim=2, keepdim=True) # 计算均值
        std_encv = torch.std(encoder_vector, dim=2, keepdim=True)   # 计算标准差
        encoder_vector = (encoder_vector - mean_encv)/(std_encv + 1e-7)
        mean_decv = torch.mean(decoder_vector, dim=2, keepdim=True) # 计算均值
        std_decv = torch.std(decoder_vector, dim=2, keepdim=True)   # 计算标准差
        decoder_vector = (decoder_vector-mean_decv)/(std_decv + 1e-7)
        similarity_matrix = torch.einsum('bik,bjk->bij', decoder_vector, encoder_vector)
       
        weighted_T = torch.einsum('bij,bjkl->bikl', [similarity_matrix, encoder_T]) 
        T_flatten = weighted_T.view(weighted_T.size(0),weighted_T.size(1),-1)
        mean_T = torch.mean(encoder_vector, dim=2, keepdim=True) # 计算均值
        std_T = torch.std(encoder_vector, dim=2, keepdim=True)
        T_flatten = (T_flatten - mean_T)/(std_T + 1e-7)
        return T_flatten
    