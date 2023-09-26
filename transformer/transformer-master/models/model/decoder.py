"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer



class Decoder(nn.Module):
    def __init__(self ,d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, trg, enc_src):
        

        for layer in self.layers:
            decoderEmbedding = layer(trg, enc_src)

        # pass to LM head
        output = self.linear(decoderEmbedding)
        return output
