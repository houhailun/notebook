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