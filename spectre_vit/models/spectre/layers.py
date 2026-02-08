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


class MHPermutMix(nn.Module):
    def __init__(
        self, embed_dim: int, token_dim: int, num_heads: int, out_channels: int, num_encoders: int
    ):
        super().__init__()
        d = embed_dim * token_dim
        self.num_heads = num_heads
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.concat_dim = self.embed_dim * self.num_heads
        signs = torch.stack([
            torch.randint(0, 2, (num_heads, d), dtype=torch.float32).unsqueeze(0) * 2 - 1
            for _ in range(num_encoders)
        ])
        self.register_buffer("signs", signs)
        perms = torch.stack([
            torch.stack([torch.randperm(d) for _ in range(num_heads)]) for _ in range(num_encoders)
        ])
        self.register_buffer("perms", perms)
        self.linear = nn.Linear(embed_dim * num_heads, out_channels)  # , bias=False)

    #
    def forward(self, x: torch.Tensor, encoder_num: int):
        B = x.shape[0]
        # B -- batch, N -- number of tokens, E -- embeddings, H -- number of heads
        x = x.view(B, -1)  # [B, N * E]
        x = x[:, self.perms[encoder_num]] * self.signs[encoder_num]  # [B, H, N * E]
        x = x.view(B, self.token_dim, self.concat_dim)  # [B, N, H * E]
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
