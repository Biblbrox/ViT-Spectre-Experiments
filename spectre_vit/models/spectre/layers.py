import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectre_vit.modules.spectre import FFT


class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, requires_grad=True):
        super().__init__()

        if requires_grad:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
        else:
            self.weight = nn.Parameter(torch.ones(out_features, in_features), requires_grad=False)

        self.scale = nn.Parameter(torch.ones(1), requires_grad=requires_grad)

    def forward(self, x):
        w_bin = self.weight.sign()
        return self.scale * (x @ w_bin.T)


# class SignPermuteMix(nn.Module):
#     def __init__(self, size: int, dim: int):
#         super().__init__()
#         self.size = size
#         self.dim = dim
#
#         signs = torch.randint(0, 2, (size,)).float() * 2 - 1
#         self.register_buffer("signs", signs)
#
#         perm = torch.randperm(size)
#         self.register_buffer("perm", perm)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         assert x.shape[self.dim] == self.size, (
#             f"Expected dim {self.dim} size {self.size}, got {x.shape[self.dim]}"
#         )
#
#         shape = [1] * x.ndim
#         shape[self.dim] = self.size
#         signs = self.signs.view(*shape)
#         x = x * signs
#
#         x = torch.index_select(x, dim=self.dim, index=self.perm)
#
#         return x


class MHPermutMix(nn.Module):
    def __init__(self, embed_dim: int, token_dim: int, num_heads: int, out_channels: int):
        super().__init__()
        d = embed_dim * token_dim
        self.num_heads = num_heads
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.concat_dim = self.embed_dim * self.num_heads
        signs = torch.randint(0, 2, (num_heads, d), dtype=torch.float32)
        signs = signs * 2 - 1
        self.register_buffer("signs", signs.unsqueeze(0))
        perms = torch.stack([torch.randperm(d) for _ in range(num_heads)])
        self.register_buffer("perms", perms)
        self.linear = SpectreLinear(embed_dim * num_heads, out_channels)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)
        x = x[:, self.perms] * self.signs
        x = x.view(B, self.token_dim, self.concat_dim)
        return self.linear(x)


class SpectreLinear(nn.Module):
    def __init__(self, in_channels, out_channels, tokens=None):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.sparsity = 1
        local_idx = torch.arange(0, in_channels, self.sparsity)
        self.register_buffer("local_idx", local_idx)  # [local_channels]
        local_channels = int(math.ceil(in_channels / self.sparsity))
        self.local_head = nn.Sequential(
            nn.Linear(local_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
        if self.in_channels == self.out_channels:
            self.avg_pool = lambda x: x
        else:
            self.avg_pool = nn.AdaptiveAvgPool1d(out_channels)

    def forward(self, x, dim=(-1)):
        """
        x: [B,N,E]
        """
        # local_feat = torch.index_select(x, dim=-1, index=self.local_idx)  # [B,N,E/s]
        local_feat = self.local_head(x)
        return local_feat + self.avg_pool(x)


class FFTApproximator(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.out_dim = dim // 2 + 1
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(self.out_dim, self.dim))

    def forward(self, x):
        """

        Args:
            x (_type_): (B, N, D)
        """
        # x = torch.exp(self.fft_param * x) + self.fft_bias
        # x = self.project(x)
        # weight = self.weight.sign()
        return x @ self.weight.T


class LearnedSigmoid(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.sharpness = 5000

    def forward(self, x):
        return 1 / (
            1 + torch.exp(1 / torch.sqrt(self.threshold**2 / self.sharpness) * (x + self.threshold))
        )
