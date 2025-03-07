import torch
import torch.nn as nn
import importlib

import attention_v1
# 使用 importlib.reload 来重新加载该模块，可以在修改模块后，不用重启内核，直接重新加载模块。
importlib.reload(attention_v1)

from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from attention_v1 import MultiHeadAttention
# from attention_v1 import FlashMultiHeadAttention
from transformers.modeling_outputs import CausalLMOutputWithPast


# 模型配置
MODEL_CONFIG = {
    "vocab_size": 32000, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}


class LayerNorm(nn.Module):
    """层归一化"""
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scaler = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        """前向传播"""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / (var + self.eps).sqrt()
        return norm_x * self.scaler + self.shift
    
    
class FeedForward(nn.Module):
    """前馈神经网络
    网络结构：线性层 + GELU + 线性层
    """
    def __init__(self, emb_dim:int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.GELU(),
            nn.Linear(4*emb_dim, emb_dim)
        )
        
    def forward(self, x):
        """前向传播"""
        return self.layers(x)
    

class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, **kwargs):
        super().__init__()
        attn_kwargs = {
            "dim_in": kwargs["emb_dim"],
            "dim_out": kwargs["emb_dim"],
            "context_length": kwargs["context_length"],
            "num_heads": kwargs["n_heads"],
            "dropout_rate": kwargs["drop_rate"],
            "qkv_bias": kwargs["qkv_bias"]
        }
        if kwargs.get('flash_attn'):
            self.atten = FlashMultiHeadAttention(**attn_kwargs)
            print("use flash attention.")
        else:
            self.atten = MultiHeadAttention(**attn_kwargs)
        
        self.ffn = FeedForward(kwargs["emb_dim"])
        self.drop = nn.Dropout(kwargs["drop_rate"])
        self.layernorm1 = LayerNorm(kwargs["emb_dim"])
        self.layernorm2 = LayerNorm(kwargs["emb_dim"])
        
    def forward(self, x, pos_cis, attention_mask=None, use_kv_cache=False, past_kv=None):
        """前向传播"""
        shortcut = x
        x = self.layernorm1(x)
        x, past_kv = self.atten(x, pos_cis, attention_mask, use_kv_cache, past_kv)
        x = self.drop(x)
        x = x + shortcut
        
        shortcut = x
        x = self.layernorm2(x)
        x = self.ffn(x)
        x = self.drop(x)
        x = x + shortcut
        
        return x, past_kv
    
    
def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算位置编码"""
    # 计算频率向量，用于旋转位置编码
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成位置索引，从0到end-1
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # 计算外积，将位置索引与频率向量相乘
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # 将频率转换为复数形式的旋转矩阵
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # 返回预计算的位置编码
    return pos_cis


def attention_mask_to_4d(attention_mask, num_heads):
    """将2D的attention mask扩展为4D"""
    # 获取输入掩码的批次大小和序列长度
    batch_size, seq_len = attention_mask.size()
    # 将2D掩码扩展为4D，增加两个维度：(batch, 1, 1, seq_len)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    # 将掩码重复以匹配多头注意力的形状：(batch, num_heads, seq_len, seq_len)
    attention_mask = attention_mask.repeat(1, num_heads, seq_len, 1)
    # 反转掩码，使得值为1的位置被掩码为-inf
    return (1 - attention_mask)


class GPTConfig(PretrainedConfig):
    """GPT模型配置"""
    model_type = "minigpt" 
    
    def __init__(self, **kwargs):
        super().__init__()
        self.context_length = kwargs.get('context_length', 1024)
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.emb_dim = kwargs.get('emb_dim', 768)
        self.drop_rate = kwargs.get('drop_rate', 0.1)
        self.n_layers = kwargs.get('n_layers', 12)
        self.n_heads = kwargs.get('n_heads', 12)
        self.qkv_bias = kwargs.get('qkv_bias', False)
        self.flash_attn = kwargs.get('flash_attn', False)
        


class MiniGPT(PreTrainedModel):
    """GPT模型"""
    config_class = GPTConfig
    
    @classmethod  
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):  
        # 自定义加载逻辑，通常包括加载权重和配置  
        model = super(MiniGPT, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)  
        return model  

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.context_length = config.context_length
        self.num_heads = config.n_heads
        self.n_layers = config.n_layers
        
        self.token_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.decode_layers = nn.ModuleList([
            TransformerBlock(**config.to_dict()) for _ in range(config.n_layers)])
        
        pos_cis = precompute_pos_cis(config.emb_dim // config.n_heads, config.context_length)
        self.register_buffer("pos_cis", pos_cis, persistent=False)
        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size)
        self.out = CausalLMOutputWithPast()
        
    def forward(self, 
                inputs:Optional[torch.Tensor]=None, 
                attention_mask:Optional[torch.Tensor]=None, 
                use_kv_cache=False,
                past_kvs=None,
                return_dict=False,
                **kwargs):
        """前向传播"""
        if not past_kvs:
            past_kvs = [None] * self.n_layers
        if 'input_ids' in kwargs:
            inputs = kwargs['input_ids']
        
        assert isinstance(inputs, torch.Tensor), f"expect torch.Tensor, but got{type(inputs)}"
        
        b, seq_len = inputs.shape
        pos_cis = self.pos_cis[:seq_len]
        
        x = self.token_emb(inputs)
        x = self.drop_emb(x)
        
        # 支持注意力掩码计算
        if attention_mask is not None:
            assert isinstance(attention_mask, torch.Tensor), f"expect torch.Tensor, but got{type(attention_mask)}"
            assert attention_mask.size() == inputs.size(), f"size of inputs {inputs.size()} and attention_mask {attention_mask.size()} must be the same."
            attention_mask = attention_mask_to_4d(attention_mask, self.num_heads)
            
        for i, block in enumerate(self.decode_layers):
            x, past_kvs[i] = block(x, pos_cis, attention_mask, use_kv_cache, past_kvs[i])
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        if not return_dict:
            return logits
        
        self.out.__setitem__('logits', logits)
        self.out.__setitem__('past_kvs', past_kvs)
        return self.out
    
    @torch.inference_mode()
    def generate(self, input_ids, max_length=512, eos_token_id=-1, **kwargs):
        """生成文本"""
        assert isinstance(max_length, int) and max_length > 0
        eos_reached = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
        past_kvs = None
        return_dict = True
        attention_mask = None
        
        if 'use_kv_cache' in kwargs:
            use_kv_cache = kwargs['use_kv_cache']
            del kwargs['use_kv_cache']
        else:
            use_kv_cache = False
        
        for _ in range(max_length):
            # 如果生成序列过程中超出上下文长度，则由后往前截取context_length个token。
            context_ids = input_ids[:, -self.context_length:]  
            with torch.no_grad():
                # 前向传播计算输出
                output = self(context_ids, attention_mask, use_kv_cache, past_kvs, return_dict, **kwargs)  # shape: batch, n_tokens, vocab_size
            past_kvs = output["past_kvs"] if use_kv_cache else None
            # 只取每个序列最后一个token的输出向量作为logits, shape变为: batch, vocab_size
            logits = output["logits"][:, -1, :]        
            # 使用softmax函数将logits转换为下一个token的概率分布，shape仍是: batch, vocab_size
            probs = torch.softmax(logits, dim=-1)   
            # 解码策略：取概率最大的作为next_input_ids，形状变为：batch, 1
            next_token_ids = torch.argmax(probs, dim=-1, keepdim=True)
            # 将next_token_id连接到下一个token的结尾， 形状变为：batch, n_tokens+1
            input_ids = torch.cat((input_ids, next_token_ids), dim=1)
            # 更新 eos_reached
            eos_reached |= (next_token_ids.squeeze(-1) == eos_token_id)
            if eos_reached.all():
                break

        return input_ids