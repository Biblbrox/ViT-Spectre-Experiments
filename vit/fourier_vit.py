from sklearn.externals.array_api_compat.torch import layer_norm
import torch
from torch import nn
from torch.nn import functional as F
import warnings
import copy
from torch.nn.modules.transformer import _detect_is_causal_mask, _get_clones, _get_activation_fn, _get_seq_len
from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
from vit.model import PatchEmbedding, transform


class SpectreBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_length, dims=(-2, -1)):
        super().__init__()
        self.seq_length = seq_length
        self.transform_weights = torch.nn.Parameter(torch.randn(seq_length, 1))
        self.dims = dims

        # Decompose mlp to separate layers
        self.linear1 = nn.Linear(out_channels, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.001)
        self.norm1 = nn.LayerNorm(in_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x, method='fft'):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = transform(x, method=method, dims=(-1)) * \
            self.transform_weights
        x = self.norm1(x)
        x = self.linear2(x)
        return x


class SpectreEncoder(nn.Module):
    def __init__(self, seq_length, num_layers, in_channels, out_channels, embed_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SpectreBlock(
            in_channels, embed_dim, seq_length=seq_length))
        for _ in range(1, num_layers - 1):
            self.layers.append(SpectreBlock(
                embed_dim, embed_dim, seq_length=seq_length))
        self.layers.append(SpectreBlock(
            embed_dim, out_channels, seq_length=seq_length))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
        use_spectre=False,
        spectre_threshold=None
    ):
        super().__init__()

        num_patches = (img_size // patch_size) ** 2

        self.embeddings_block = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        )

        self.encoder_block = SpectreEncoder(
            seq_length=num_patches + 1,
            num_layers=num_encoders,
            in_channels=embed_dim,
            out_channels=embed_dim,
            embed_dim=embed_dim)

        if use_spectre and spectre_threshold is not None:
            embed_dim = int(embed_dim * spectre_threshold)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_block(x)
        x = x[:, 0, :]
        x = self.mlp_head(x)  # CLS token
        return x
