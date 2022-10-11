import copy
from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import *
from torch.nn.init import constant_, xavier_uniform_
import numpy as np
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LScaledDotProductAttention(Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, groups=2):

        super(LScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_k = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_v = nn.Linear(d_model // groups, h * d_v // groups)
        self.fc_o = nn.Linear(h * d_v // groups, d_model // groups)
        self.dropout = nn.Dropout(dropout)
        self.groups = groups

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.fc_q.weight)
        xavier_uniform_(self.fc_k.weight)
        xavier_uniform_(self.fc_v.weight)
        xavier_uniform_(self.fc_o.weight)
        constant_(self.fc_q.bias, 0)
        constant_(self.fc_k.bias, 0)
        constant_(self.fc_v.bias, 0)
        constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        queries = queries.permute(1, 0, 2)
        keys = keys.permute(1, 0, 2)
        values = values.permute(1, 0, 2)
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries.view(b_s, nq, self.groups, -1)).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys.view(b_s, nk, self.groups, -1)).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values.view(b_s, nk, self.groups, -1)).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out.view(b_s, nq, self.groups, -1)).view(b_s, nq, -1)
        return out.permute(1, 0, 2)


class LMultiHeadAttention(Module):
    def __init__(self, d_model, h, dropout=.1, batch_first=False, groups=2, device=None, dtype=None):
        super(LMultiHeadAttention, self).__init__()

        self.attention = LScaledDotProductAttention(d_model=d_model, groups=groups, d_k=d_model // h, d_v=d_model // h,
                                                    h=h, dropout=dropout)

    def forward(self, queries, keys, values, attn_mask=None, key_padding_mask=None,need_weights=False,attention_weights=None):
        out = self.attention(queries, keys, values, attn_mask, attention_weights)
        return out, out



class Lightformer(Module):

    __constants__ = ['norm']

    def __init__(self, attention_layer, num_layers, norm=None):
        super(Lightformer, self).__init__()
        self.layers = _get_clones(attention_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for i, mod in enumerate(self.layers):
            if i % 2 ==0:
                output = mod(output)
            else:
                output = mod(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class LightformerLayer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LightformerLayer, self).__init__()
        self.self_attn = LMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                             **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward // 2, d_model // 2, **factory_kwargs)  ###

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(LightformerLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        b, l, d = x.size()
        x = self.linear2(self.dropout(self.activation(self.linear1(x))).view(b, l, 2, d*4 // 2)) ###
        x= x.view(b, l, d)
        return self.dropout2(x)



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    return F.gelu
