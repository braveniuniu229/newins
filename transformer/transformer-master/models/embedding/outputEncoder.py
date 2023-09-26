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
        similarity_matrix = torch.einsum('bik,bjk->bij', decoder_vector, encoder_vector)
        l2_norm = torch.norm(similarity_matrix, p=2, dim=-1, keepdim=True)
        epsilon = 1e-10  # Calculate L2 norm
        normalized_similarity_matrix = similarity_matrix / (l2_norm + epsilon)
        weighted_T = torch.einsum('bij,bjkl->bikl', [normalized_similarity_matrix, encoder_T]) 
        T_flatten = weighted_T.view(weighted_T.size(0),weighted_T.size(1),-1)
        
        return T_flatten
    