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
    """自注意力机制模块
    """
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
    """因果注意力机制模块
    """
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
    

class MultiHeadAttention(nn.Module):
    """多头注意力机制模块
    """
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
        
    def forward(self, x):
        b, num_tokens, dim_in = x.shape
        # 求Q/K/V矩阵，shape为（b, num_tokens, dim_out）
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        
        # 多头注意力矩阵分割
        # 变换形状，使用view将dim_out -> num_heads, head_dim
        # shape=(b, num_tokens, num_heads, head_dim)
        q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # 由于计算注意力分数时，需要对每一个头并行计算，这里需要交换第2维和第3维
        # shape：（b, num_heads, num_tokens, self.dim_out）
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数,shape=(b, num_heads, num_tokens, num_tokens)
        atten_scores = q @ k.transpose(2, 3)
        atten_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        atten_weights = torch.softmax(atten_scores / k.shape[-1] ** 0.5, dim=-1)
        atten_weights = self.dropout(atten_weights)
        
        # 计算上下文张量
        context_vecs = atten_weights @ v  # shape=(b, num_heads, num_tokens, head_dim)
        context_vecs = context_vecs.transpose(1, 2)  # shape=(b, num_tokens, num_heads, head_dim)
        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.dim_out)
        # print(context_vecs.shape)
        # print(self.Wo)
        output = self.Wo(context_vecs)
        
        return output

    
class LayerNorm(nn.Module):
    """层归一化"""
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.eps = 1e-5
        self.scaler = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean) / (var + self.eps) ** 0.5
        x_norm = self.scaler * x_norm + self.shift
        
        return x_norm
        

class FeedForward(nn.Module):
    """前馈神经网络模块
    """
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    
class ExampleDeepNetwork(nn.Module):
    """一个简单的深度网络"""
    def __init__(self, layer_sizes, use_shortcut=True):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), nn.ReLU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), nn.ReLU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), nn.ReLU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), nn.ReLU())
        ])
        self.use_shortcut = use_shortcut
        
    def forward(self, x):
        for layer in enumerate(self.layers):
            out = layer(x)
            if self.use_shortcut and out.shape == x.shape:
                x = x + out
            else:
                x = out
        return x
    
    
class TransformerBlock(nn.Module):
    """Transformer模块
    """
    def __init__(self, cfg:dict):
        super().__init__()
        self.atten = MultiHeadAttention(
            dim_in=cfg["emb_dim"],
            dim_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout_rate=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        
        self.ffn = FeedForward(
            dim_in=cfg["emb_dim"],
            dim_hidden=4 * cfg["emb_dim"],
            dim_out=cfg["emb_dim"]
        )
        
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.layernorm1 = LayerNorm(cfg["emb_dim"])
        self.layernorm2 = LayerNorm(cfg["emb_dim"])
        
    def forward(self, x):
        # 在以前的某些架构中，层归一化是在MultiHeadAttention和FeedForward之后用的，这被称为Post-Layernorm，
        # 这种方法在较浅的网络中表现良好，但在更深的网络中会遇到训练不稳定的问题，
        # 总体来说，Pre-Layernorm在稳定性方面表现更好。
        
        # 多头自注意力计算
        shortcut = x
        x = self.layernorm1(x)
        x = self.atten(x)
        x = self.dropout(x)
        x = x + shortcut
        
        # FFN计算
        shortcut = x
        x = self.layernorm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut
        
        return x
    
    
MODEL_CONFIG = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

# torch.manual_seed(123)
# example_input = torch.randn(2, 4, 768)
# block = TransformerBlock(MODEL_CONFIG)
# output = block(example_input)
# output.shape


from transformers import PretrainedConfig, PreTrainedModel
class GPTConfig(PretrainedConfig):
    """模型配置封装"""
    # 每个模型都必须有一个独特的model_type,都则会报"Should have a `model_type` key in its config.json"
    model_type = 'minigpt'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 所有配置参数使用类的成员属性显式定义，这样可以提供默认值，并限制类型
        self.context_length = kwargs.get("context_length", 1024)
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.emb_dim = kwargs.get("emb_dim", 768)
        self.drop_rate = kwargs.get("drop_rate", 0.1)
        self.n_heads = kwargs.get("n_heads", 12)
        self.n_layers = kwargs.get("n_layers", 12)
        self.qkv_bias = kwargs.get("qkv_bias", False)
    


class MiniGPT(PreTrainedModel)