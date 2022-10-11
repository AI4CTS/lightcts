import torch.nn as nn
from transformer import LightformerLayer,Lightformer

class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 170):
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
        n_heads=8,
        layers=6
    ):
        super(Transformer, self).__init__()

        self.layers = 4
        self.hid_dim =d_model
        self.heads = n_heads

        self.attention_layer = LightformerLayer(self.hid_dim, self.heads, self.hid_dim * 4)
        self.attention_norm = nn.LayerNorm(self.hid_dim)
        self.attention = Lightformer(self.attention_layer, self.layers, self.attention_norm)
        self.lpos = LearnedPositionalEncoding(self.hid_dim)

    def forward(self,input, mask):
        x = input.permute(1,0,2)
        x = self.lpos(x)
        output = self.attention(x, mask)

        return output.permute(1,0,2)
