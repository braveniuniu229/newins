"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
import torch.nn.functional as F

class inputEncoder(nn.Module):
    def __init__(self):
        super(inputEncoder, self).__init__()
        
    def forward(self, encoder_vector, encoder_T):
         # encoder_vector shape: [5, 4, 16]
        similarity_matrix = torch.einsum('bik,bjk->bij', encoder_vector, encoder_vector)
        
        # Calculate L2 norm along the last dimension and expand it to have the same shape as similarity_matrix
        l2_norm = torch.norm(similarity_matrix, p=2, dim=-1, keepdim=True)  # Calculate L2 norm
        
        # Prevent division by zero by adding a small epsilon
        epsilon = 1e-10
        
        # Normalize the similarity_matrix by L2 norm
        normalized_similarity_matrix = similarity_matrix / (l2_norm + epsilon)

        # 应用权重并累加
        weighted_T = torch.einsum('bij,bjkl->bikl', [normalized_similarity_matrix, encoder_T])  # [5, 4, 51, 51]
        T_flatten = weighted_T.view(weighted_T.size(0),weighted_T.size(1),-1)
        
        return T_flatten

