import torch
import torch.nn as nn
import torch
import torch.nn as nn
import math
from torch.nn import BatchNorm2d, Conv2d, Conv2d, ModuleList, Parameter, ModuleList
import numpy as np
import torch.nn.functional as F
import math
from transformer import LightformerLayer,Lightformer

class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 137):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=32,
        layer=8,
    ):
        super(Transformer, self).__init__()

        self.layers = layer
        self.hid_dim =d_model

        self.attention_layer = LightformerLayer(self.hid_dim, 8, self.hid_dim * 4)
        self.attention_norm = nn.LayerNorm(self.hid_dim)
        self.attention = Lightformer(self.attention_layer, self.layers, self.attention_norm)
        self.lpos = LearnedPositionalEncoding(self.hid_dim)

    def forward(self,input):
        x = input.permute(1,0,2)
        x = self.lpos(x)
        output = self.attention(x)

        return output.permute(1,0,2)
