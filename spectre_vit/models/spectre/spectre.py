import math

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from debugpy.launcher.debuggee import process
from mpmath.tests.test_hp import b
from polars import Binary
from pytorch_wavelets import DWT1DForward, DWT1DInverse, DWTForward, DWTInverse
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_activation_fn, _get_clones
from torch.profiler import ProfilerActivity, profile, record_function

from spectre_vit.models.spectre.layers import SignPermuteMix, SpectreLinear
from spectre_vit.modules.patch_embeddings import PatchEmbedding

# from fast_hadamard_transform import hadamard_transform


def transform(x: torch.Tensor, dims, method="fft", pad=False):
    if method == "fft":
        if pad:
            # F.pad(x,)
            pass
        return torch.fft.fft(x, dims=dims)
    elif method == "hadamar":
        pass


class Transpose(nn.Module):
    def __init__(self, dims=(-2, -1)):
        super(Transpose, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(self.dims[0], self.dims[1])


def shifted_sigmoid(x, threshold):
    return 1 / (1 + torch.exp(x + threshold))


def shifted_sigmoid2(x, threshold):
    return 1 / (1 + torch.exp(1 / math.sqrt(threshold**2 / 5000) * (x + threshold)))


class NormalMask(nn.Module):
    def __init__(self, n_bins):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(n_bins / 2.0))
        self.std = nn.Parameter(torch.tensor(n_bins / 8.0))
        self.freqs = torch.linspace(0, n_bins - 1, n_bins)

    def forward(self, X):
        gauss = torch.exp(-0.5 * ((self.freqs.to(X.device) - self.mean) / (self.std + 1e-8)) ** 2)
        return X * gauss


class SpectreMix(nn.Module):
    def __init__(self, in_channels, num_heads, seq_length, method="fft"):
        super().__init__()

        self.num_heads = num_heads
        self.in_channels = in_channels
        self.method = method

        if method == "fft_bare":
            pass
        elif method == "fft_mh":
            self.head_linears = nn.ModuleList([
                nn.Linear(in_channels, in_channels) for _ in range(num_heads)
            ])
        elif method == "fft_mh_spectrelayers":
            scale = 2
            self.common_linear = SpectreLinear(in_channels, in_channels // scale)
            shrink = 6
            self.fused = False
            if self.fused:
                self.head_linears_fused = nn.Sequential(
                    SignPermuteMix(seq_length * self.num_heads, -2),
                    SpectreLinear(
                        in_channels // scale,
                        in_channels // shrink * self.num_heads,
                    ),
                )
            else:
                self.head_linears = nn.ModuleList([
                    nn.Sequential(
                        SignPermuteMix(seq_length, -2),
                        SpectreLinear(in_channels // scale, in_channels // shrink),
                    )
                    for _ in range(self.num_heads)
                ])
            self.proj_head = nn.Sequential(
                SpectreLinear(
                    in_channels // shrink * num_heads, in_channels // shrink * num_heads // 2
                ),
                SpectreLinear(in_channels // shrink * num_heads // 2, in_channels),
            )
        elif method == "fft_param":
            self.param = [nn.Parameter(torch.ones(seq_length, 1)) for _ in range(num_heads)]
        elif method == "dwt_embed":
            self.dwt = DWT1DForward(num_heads, wave="haar", mode="zero")
            self.head_linear = nn.ModuleList([
                nn.Linear(in_channels // 2 ** (i + 1), in_channels) for i in range(num_heads)
            ])
        elif method == "dwt_token":
            self.dwt = DWT1DForward(num_heads, wave="haar", mode="zero")
            self.head_linear = nn.ModuleList([
                nn.Linear(seq_length // 2 ** (i + 1) + 1, seq_length) for i in range(num_heads)
            ])

    def forward(self, x):
        """
        x: [B, N, E]
        """
        if self.method == "fft_bare":
            return torch.fft.fft2(x, dim=(-2, -1)).real
        elif self.method == "fft_param":
            feat = None
            for i in range(self.num_heads):
                if feat is None:
                    feat = torch.fft.fft2(x, dim=(-2, -1)).real * self.param[i]
                else:
                    feat += torch.fft.fft2(x, dim=(-2, -1)).real * self.param[i]
            return feat
        elif self.method == "fft_mh" or self.method == "fft_mh_spectrelayers":
            head_embed = self.common_linear(x)
            if self.fused:
                head_embed = torch.cat(
                    [head(head_embed) for head in self.permutation_heads], dim=-1
                )
                full_embed = self.head_linears_fused(head_embed)
            else:
                full_embed = [head(head_embed) for head in self.head_linears]
                full_embed = torch.cat(full_embed, dim=-1)
            projected = self.proj_head(full_embed)
            return projected
        elif self.method == "dwt_embed":
            approx, detail = self.dwt(x)
            full_embed = None
            for i in range(self.num_heads):
                head_embed = self.head_linear[i](detail[i])
                if full_embed is None:
                    full_embed = head_embed
                else:
                    full_embed += head_embed

            return full_embed
        elif self.method == "dwt_token":
            x = x.transpose(2, 1)
            approx, detail = self.dwt(x)
            full_embed = None
            for i in range(self.num_heads):
                head_embed = self.head_linear[i](detail[i])
                head_embed = head_embed.transpose(2, 1)
                if full_embed is None:
                    full_embed = head_embed
                else:
                    full_embed += head_embed
            return full_embed


class SpectreEncoderLayer(nn.Module):
    """Spectre encoding layer. The following configurations are supported:
    - fft_bare - FNet implementation
    - fft_mh - Multi-Head fft with individials linear layers for each head
    - dwt_embed - Wavelet transform in embedding dimension
    - dwt_token - Wavelet transform in token dimension
    - attention - Native ViT Self-Attention
    """

    def __init__(
        self,
        seq_length,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        method="fft_mh_spectrelayers",
    ):
        super().__init__()
        assert method in [
            "fft_bare",
            "fft_mh",
            "fft_param",
            "dwt_embed",
            "dwt_token",
            "fft_mh_spectrelayers",
        ]
        self.method = method
        bias = True
        layer_norm_eps = 1e-5
        self.mix_layer = SpectreMix(d_model, nhead, seq_length, method=method)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = SpectreLinear(d_model, dim_feedforward)
        self.linear2 = SpectreLinear(dim_feedforward, dim_feedforward)
        self.linear3 = SpectreLinear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation

    def forward(self, src):
        x = src
        old_x = x
        x = self.mix_layer(x)
        x = self.norm1(x) + old_x
        x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.linear1(x))
        x = self.linear2(x)
        # x = self.transform_norm(x)
        x = self.linear3(x)
        return self.dropout2(x)


class SpectreEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        norm=None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
    ):
        output = src
        for idx, mod in enumerate(self.layers):
            output = mod(
                output,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output + src


class SpectralMask(nn.Module):
    def __init__(self, n_freq):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.zeros(n_freq))

    def forward(self, x):
        # x: [B,N,E]
        X = torch.fft.fft(x, dim=-1).real

        m = torch.sigmoid(self.mask_logits)  # [F]
        X = X * m[None, None, :]  # broadcast

        return X


# class SpectralPatchEmbed(nn.Module):
#     def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels, keep_ratio=0.5):
#         super().__init__()
#         self.P = patch_size
#
#         fh = int(patch_size * keep_ratio)
#         fw = int((patch_size // 2 + 1) * keep_ratio)
#
#         self.fh, self.fw = fh, fw
#
#         self.proj = nn.Linear(in_channels * fh * fw, embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # x: [B, C, H, W]
#         B, C, H, W = x.shape
#
#         # Split patches
#         x = x.unfold(2, self.P, self.P).unfold(3, self.P, self.P)
#         x = x.contiguous().view(B, C, -1, self.P, self.P)
#
#         # FFT per patch
#         x_fft = torch.fft.rfft2(x, norm="ortho").real
#
#         # Keep low freqs
#         x_fft = x_fft[:, :, :, : self.fh, : self.fw]
#
#         # x_fft = x_fft.flatten(1, -1)
#         # Flatten channels + freq dimensions per patch
#         x_fft = x_fft.permute(0, 2, 1, 3, 4)  # [B, N, C, fh, fw]
#         x_fft = x_fft.flatten(2)  # [B, N, C*fh*fw]
#
#         x_proj = self.proj(x_fft)
#         cls_token = self.cls_token.expand(B, -1, -1)
#         x_out = torch.cat((cls_token, x_proj), dim=1)  # [B, N+1, E]
#         x_out = x_out + self.position_embeddings
#         x_out = self.dropout(x_out)
#
#         return x_out


class SpectralPatchEmbed(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.P = patch_size
        self.embed_dim = embed_dim

        # Learnable frequency importance
        self.freq_weight_h = nn.Parameter(torch.ones(self.P))
        self.freq_weight_w = nn.Parameter(torch.ones(self.P // 2 + 1))

        # Linear projection from spectral patch to embedding
        self.proj = nn.Linear(in_channels * self.P * (self.P // 2 + 1), embed_dim)

        # CLS token + positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        P = self.P

        # Split into patches
        x = x.unfold(2, P, P).unfold(3, P, P)  # [B, C, nH, nW, P, P]
        nH, nW = x.size(2), x.size(3)
        N = nH * nW
        x = x.contiguous().view(B, C, N, P, P)  # [B, C, N, P, P]

        # FFT per patch
        x_fft = torch.fft.rfft2(x, norm="ortho").real  # [B, C, N, P, P//2 + 1]

        # Apply learnable frequency importance
        freq_h = self.freq_weight_h.view(1, 1, 1, P, 1)
        freq_w = self.freq_weight_w.view(1, 1, 1, 1, P // 2 + 1)
        x_fft = x_fft * freq_h * freq_w  # [B, C, N, P, P//2 + 1]

        # Flatten C + freq dims per patch
        x_fft = x_fft.permute(0, 2, 1, 3, 4)  # [B, N, C, P, P//2+1]
        x_fft = x_fft.flatten(2)  # [B, N, C*P*(P//2+1)]

        # Linear projection per patch
        x_proj = self.proj(x_fft)  # [B, N, embed_dim]

        # CLS token + positional embeddings
        cls_token = self.cls_token.expand(B, -1, -1)
        x_out = torch.cat((cls_token, x_proj), dim=1)  # [B, N+1, embed_dim]
        x_out = x_out + self.position_embeddings
        x_out = self.dropout(x_out)

        return x_out


class SpectreViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=768,
        num_encoders=12,
        num_heads=12,
        hidden_dim=3072,
        dropout=0.1,
        activation="gelu",
        method="attention",
    ):
        super().__init__()

        num_patches = (img_size // patch_size) ** 2

        # self.embeddings_block = PatchEmbedding(
        #     embed_dim, patch_size, num_patches, dropout, in_channels
        # )

        self.embeddings_block = SpectralPatchEmbed(
            embed_dim, patch_size, num_patches, dropout, in_channels
        )

        encoder_layer = SpectreEncoderLayer(
            seq_length=num_patches + 1,
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            method=method,
        )

        self.encoder_blocks = SpectreEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(SpectreLinear(embed_dim, num_classes))
        # self.cls_norm = nn.LayerNorm(100)

    def forward(self, x, return_features=False):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)

        cls_token = x[:, 0, :]
        x = self.mlp_head(cls_token)
        if return_features:
            return x, cls_token
        return x
