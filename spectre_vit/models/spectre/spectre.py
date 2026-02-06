import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_activation_fn, _get_clones

from spectre_vit.models.spectre.layers import MHPermutMix, SpectreLinear


class Transpose(nn.Module):
    def __init__(self, dims=(-2, -1)):
        super(Transpose, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(self.dims[0], self.dims[1])


class NormalMask(nn.Module):
    def __init__(self, n_bins):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(n_bins / 2.0))
        self.std = nn.Parameter(torch.tensor(n_bins / 8.0))
        self.freqs = torch.linspace(0, n_bins - 1, n_bins)

    def forward(self, X):
        gauss = torch.exp(-0.5 * ((self.freqs.to(X.device) - self.mean) / (self.std + 1e-8)) ** 2)
        return X * gauss


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
    ):
        super().__init__()
        bias = True
        layer_norm_eps = 1e-5
        self.mix_layer = MHPermutMix(d_model, seq_length, nhead, d_model)
        self.linear1 = SpectreLinear(d_model, dim_feedforward)
        self.linear3 = SpectreLinear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation

    def forward(self, x):
        x = self.norm1(self.mix_layer(x)) + x
        x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.linear1(x))
        x = self.dropout2(self.linear3(x))
        return x


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
        B, C, _, _ = x.shape
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
    ):
        super().__init__()

        num_patches = (img_size // patch_size) ** 2

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
        )

        self.encoder_blocks = SpectreEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(SpectreLinear(embed_dim, num_classes))

    def forward(self, x, return_features=False):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)

        cls_token = x[:, 0, :]
        x = self.mlp_head(cls_token)
        if return_features:
            return x, cls_token
        return x
