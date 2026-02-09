import torch
from torch import nn
from torch.nn.modules.transformer import _get_activation_fn, _get_clones

from spectre_vit.modules.patch_embeddings import PatchEmbedding


class FNetEncoderLayer(nn.Module):
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
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear3 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation

    def forward(self, x):
        x = self.norm1(torch.fft.fft2(x).real) + x
        x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.linear1(x))
        x = self.dropout2(self.linear3(x))
        return x


class FNetEncoder(nn.Module):
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


class FNet(nn.Module):
    def __init__(self, c):
        super().__init__()

        num_patches = (c.img_size // c.patch_size) ** 2

        self.embeddings_block = PatchEmbedding(
            c.embed_dim, c.patch_size, num_patches, c.dropout, c.in_channels
        )

        encoder_layer = FNetEncoderLayer(
            seq_length=num_patches + 1,
            d_model=c.embed_dim,
            nhead=c.num_heads,
            dim_feedforward=c.hidden_dim,
            dropout=c.dropout,
            activation=c.activation,
        )

        self.encoder_blocks = FNetEncoder(encoder_layer, num_layers=c.num_encoders)

        self.mlp_head = nn.Sequential(nn.Linear(c.embed_dim, c.num_classes))

    def forward(self, x, return_features=False):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)

        cls_token = x[:, 0, :]
        x = self.mlp_head(cls_token)
        if return_features:
            return x, cls_token
        return x
