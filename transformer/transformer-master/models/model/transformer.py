"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
from models.embedding.inputEncoder import inputEncoder
from models.embedding.outputEncoder import outputEncoder
from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self,d_model, n_head, 
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
       
        self.device = device
        self.enemb = inputEncoder()
        self.deemb = outputEncoder() 
        self.encoder = Encoder(
                                d_model=d_model,
                                ffn_hidden=ffn_hidden,
                                n_layers=n_layers,
                                n_head=n_head,
                                drop_prob=drop_prob,
                                device=device)
                            

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, encoder_T, encoder_vector, decoder_vector, target ):
        src=self.enemb(encoder_vector, encoder_T)
        enc_src = self.encoder(src)
        trg = self.deemb(encoder_vector,encoder_T,decoder_vector)
        output = self.decoder(trg, enc_src)
        output = output.squeeze(1)
        return output

    def make_src_mask(self, src): #src是稀疏表达的分类，sparse representation classifier
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask