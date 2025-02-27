import torch
from torch import nn
from typing import Tuple, List


inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)   
)
batch = torch.stack((inputs, inputs), dim=0)

class SelfAttentionV1(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.Wq = nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=True)
        self.Wk = nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=True)
        self.Wv = nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=True)

    def forward(self, x):
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        atten_scores = q @ k.T
        atten_weights = torch.softmax(atten_scores/self.dim_out ** 0.5, dim=-1)
        context_vecs = atten_weights @ v
        return context_vecs

class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout_rate, qkv_bias=False):
        super().__init__()
        self.dim_out = dim_out
        self.Wq = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Wk = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Wv = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, seq_len, dim = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        atten_scores = q @ k.transpose(1, 2)  # 将第1维和第2维转置，从第0维开始
        atten_scores = atten_scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], -torch.inf)
        atten_weights = torch.softmax(atten_scores/self.dim_out ** 0.5, dim=-1)
        context_vecs = self.dropout(atten_weights) @ v
        return context_vecs
    

def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1]), f"{pos_cis.shape} == {(x.shape[1], x.shape[-1])} ?"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout_rate, num_heads, qkv_bias=False):
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads   # 每个头的维度
        self.Wq = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Wk = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Wv = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Wo = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, pos_cis, attention_mask=None, use_kv_cache=False, past_kv:Tuple[torch.Tensor]=None):
        # 输入形状
        b, num_tokens, dim_in = x.shape

        # 求Q\K\V矩阵，形状变为： b, num_tokens, dim_out
        if use_kv_cache:
            current_token = x[:, -1:, :]
            # 如果past_kv存在，则只计算current_token的q\k\v
            if past_kv != None:
                past_k, past_v = past_kv
                q = torch.cat((torch.zeros_like(x[:, :-1, :]), self.Wq(current_token)), dim=1)
                k = torch.cat((past_k, self.Wk(current_token)), dim=1)
                v = torch.cat((past_v, self.Wv(current_token)), dim=1)
            else: # 如果past_kv不存在，则计算全部
                q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
            past_kv = (k, v)
        else:
            q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)

        # 变换形状，将最后一维拆成多头，每个头有head_dim维，矩阵形状由三维变为四维。
        q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, pos_cis) if pos_cis is not None else (q, k)
        
        # 交换第2维和第3维，这一步过后，形状变为：b, num_heads, num_tokens, head_dim
        q = q.transpose(1, 2)   
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数，形状变为: b, num_heads, num_tokens, num_tokens
        atten_scores = q @ k.transpose(2, 3)   
        atten_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        scaled_atten_scores = atten_scores/k.shape[-1]**0.5

        # 注意力掩码运算
        if attention_mask is not None:
            # filled the padding with -inf
            scaled_atten_scores = scaled_atten_scores.masked_fill(attention_mask==0, -torch.inf)

        atten_weights = torch.softmax(scaled_atten_scores, dim=-1)
        atten_weights = self.dropout(atten_weights)

        context_vecs = atten_weights @ v   # shape: b, num_heads, num_tokens, head_dim
        context_vecs = context_vecs.transpose(1,2)  # shape: b, num_tokens, num_heads, head_dim
        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.dim_out)
        output = self.Wo(context_vecs)

        return output, past_kv