# -*- coding: UTF-8 -*-

import math
from sqlite3 import adapters
from typing import ForwardRef
from numpy.testing._private.utils import print_assert_equal
import torch
import torch.nn as nn
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=False, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention.
        """
        self.d_model = d_model
        self.h = n_heads
        self.d_k = self.d_model // self.h
        self.kq_same = kq_same

        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

    def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        return x.view(*new_x_shape).transpose(-2, -3)

    def forward(self, q, k, v, mask=None):
        origin_shape = q.size()
        
        # perform linear operation and split into h heads
        if not self.kq_same:
            
            q = self.head_split(self.q_linear(q))
        else:
            q = self.head_split(self.k_linear(q))
        k = self.head_split(self.k_linear(k))
        v = self.head_split(self.v_linear(v))

        # calculate attention using function we will define next
       
        output,scores = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)
        
        # concatenate heads and put through final linear layer
        output = output.transpose(-2, -3).reshape(origin_shape)
        return output,scores
    def forward_hypter_prefix(self, q, k, v, prefix_k,prefix_q,mask=None):
        origin_shape = v.size()
        
        # perform linear operation and split into h heads
        if not self.kq_same:
            q=self.q_linear(q)
         
            q=torch.cat((prefix_q[:,:,0,:],q),dim=1)
           
            q = self.head_split(self.q_linear(q))
        else:
            q = self.head_split(self.k_linear(q))
        k=self.k_linear(k)
        k=torch.cat((prefix_k[:,:,0,:],k),dim=1)

        k = self.head_split(self.k_linear(k))
        v = self.head_split(self.v_linear(v))

        # calculate attention using function we will define next
       
        output,scores = self.scaled_dot_product_attention(q, k, v, self.d_k, mask,prefix_len=prefix_k.shape[1])
       
        # concatenate heads and put through final linear layer
        output = output.transpose(-2, -3).reshape(origin_shape)
        
        return output,scores

    @staticmethod
    def scaled_dot_product_attention(q, k, v, d_k, mask=None,prefix_len=None):
        """
        This is called by Multi-head attention object to find the values.
        """
       
        scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # bs, head, q_len, k_len
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        
        scores = (scores - scores.max()).softmax(dim=-1)
        scores = scores.masked_fill(torch.isnan(scores), 0)

        if prefix_len is not None:
            scores=scores[:,:,prefix_len:,prefix_len:]
        
        output = torch.matmul(scores, v)  # bs, head, q_len, d_k
      
        
        return output,scores

class MultiHeadAttention_dimunchanged(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=False, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention.
        """
        self.d_model = d_model
        self.h = n_heads
        self.d_k = self.d_model
        self.kq_same = kq_same

        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model*self.h, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model*self.h, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model*self.h, bias=bias)

    def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
        
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        return x.view(*new_x_shape).transpose(-2, -3)

    def forward(self, q, k, v, mask=None):
        origin_shape = q.size()
        
        # perform linear operation and split into h heads
        
        if not self.kq_same:
            q = self.head_split(self.q_linear(q))
            
        else:
            q = self.head_split(self.k_linear(q))
        
        k = self.head_split(self.k_linear(k))
        v = self.head_split(self.v_linear(v))

        # calculate attention using function we will define next
       
        output = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)

        # concatenate heads and put through final linear layer
        output = output.transpose(-2, -3)
        
        return output

  

    @staticmethod
    def scaled_dot_product_attention(q, k, v, d_k, mask=None):
        """
        This is called by Multi-head attention object to find the values.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # bs, head, q_len, k_len
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        
        scores = (scores - scores.max()).softmax(dim=-1)
        scores = scores.masked_fill(torch.isnan(scores), 0)
        output = torch.matmul(scores, v)  # bs, head, q_len, d_k
        
        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0, kq_same=False):
        super().__init__()
        """
        This is a Basic Block of Transformer. It contains one Multi-head attention object. 
        Followed by layer norm and position wise feedforward net and dropout layer.
        """
        # Multi-Head Attention Block
       
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, seq, mask=None):
      
        context,scores = self.masked_attn_head(seq, seq, seq, mask)
        
        context = self.layer_norm1(self.dropout1(context) + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output

    def forward_adapter(self,seq,mask=None,adapter1=None,adapter2=None):
      
        context,scores = self.masked_attn_head(seq, seq, seq, mask)
      
        context=self.dropout1(context)
        context=adapter1(context)+context
        context = self.layer_norm1(context + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output=self.dropout2(output)
        output=adapter2(output)+output
        output = self.layer_norm2(output + context)
        return output
    def forward_adapter_prefix(self,seq,mask=None,adapter1=None,adapter2=None,prefix=None):
        '''
        prefix is a tuple (prefix_k,prefix_q)
        '''
        
        prefix_len=prefix[0].shape[1]
        # mask_prefix=torch.ones((mask.shape[0],mask.shape[1],prefix_len,prefix_len)).to(seq.device)
        mask=torch.nn.functional.pad(mask,(prefix_len,0,prefix_len,0),value=1)
        context,scores = self.masked_attn_head.forward_hypter_prefix(seq, seq, seq,prefix[0],prefix[1], mask)
        context=self.dropout1(context)
        context=adapter1(context)+context
        context = self.layer_norm1(context + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output=self.dropout2(output)
        output=adapter2(output)+output
        output = self.layer_norm2(output + context)
        return output
    def forward_prefix(self, seq, mask=None,prefix=None):
        prefix_len=prefix[0].shape[1]
        # mask_prefix=torch.ones((mask.shape[0],mask.shape[1],prefix_len,prefix_len)).to(seq.device)
        
        mask=torch.nn.functional.pad(mask,(prefix_len,0,prefix_len,0),value=1)
        
        context,scores = self.masked_attn_head.forward_hypter_prefix(seq, seq, seq,prefix[0],prefix[1], mask)
        
        context = self.layer_norm1(self.dropout1(context) + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output



class TransformerLayerv2(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0, kq_same=False):
        super().__init__()
        """
        This is a Basic Block of Transformer. It contains one Multi-head attention object. 
        Followed by layer norm and position wise feedforward net and dropout layer.
        """
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, seq, mask=None):
        norm_seq=self.layer_norm1(seq)
        context = self.masked_attn_head(norm_seq, norm_seq, norm_seq, mask)
        context = self.dropout1(context) + seq
        norm_context=self.layer_norm2(context)
        output = self.linear1(norm_context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output

class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob=0.0, hidden_act='relu', layer_norm_eps=1e-5):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": torch.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]
    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)
    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


import torch.nn as nn
import torch.nn.functional as F

class DilatedResBlock(nn.Module):
    def __init__(self,dilation,channel,max_len):
        super(DilatedResBlock,self).__init__()
        self.dilation = dilation
        self.channel = channel
        self.half_channel = int(channel/2)
        self.max_len = max_len
        
        self.reduce = nn.Conv1d(channel,self.half_channel,1)
        self.masked = nn.Conv1d(self.half_channel,self.half_channel,3,dilation=dilation)
        self.increase = nn.Conv1d(self.half_channel,channel,1)
        """
        self.reduce_norm = nn.LayerNorm(normalized_shape=[max_len])#channel)
        self.masked_norm = nn.LayerNorm(normalized_shape=[max_len])#self.half_channel)
        self.increase_norm = nn.LayerNorm(normalized_shape=[max_len])#self.half_channel)
        """
        self.reduce_norm = nn.LayerNorm(normalized_shape=channel)
        self.masked_norm = nn.LayerNorm(normalized_shape=self.half_channel)
        self.increase_norm = nn.LayerNorm(normalized_shape=self.half_channel)
        
    def forward(self,x):
        y = self.reduce_norm(x.permute(0,2,1)).permute(0,2,1)
        #y = self.reduce_norm(x)

        y = F.leaky_relu(x)
        y = self.reduce(y)
        
                
        y = self.masked_norm(y.permute(0,2,1)).permute(0,2,1)
        y = F.leaky_relu(y)
        y = F.pad(y,pad=(2 + (self.dilation-1)*2,0),mode='constant')
        y = self.masked(y)
      
        
        y = self.increase_norm(y.permute(0,2,1)).permute(0,2,1)
        #y = self.increase_norm(y)
        y = F.leaky_relu(y)
        y = self.increase(y)
        
        return x+y