import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_activation_fn, _get_clones

from spectre_vit.models.spectre.spectre import Transpose
from spectre_vit.modules.patch_embeddings import PatchEmbedding


class SpectreMix(nn.Module):
    def __init__(self, in_channels, num_heads, seq_length):
        super().__init__()

        self.num_heads = num_heads
        self.in_channels = in_channels

        shrink = 4
        self.head_linears = nn.ModuleList([
            nn.Linear(in_channels, in_channels // shrink) for _ in range(self.num_heads)
        ])
        # self.token_proj = nn.Linear(seq_length // 2 + 1, seq_length)
        self.proj_head = nn.Linear(self.in_channels // shrink * self.num_heads, in_channels)

    def forward(self, x):
        """
        x: [B, N, E]
        """
        residual = x
        # full_embed = [torch.fft.rfft2(head(x), dim=(-2, -1)).real for head in self.head_linears]
        full_embed = [head(x) for head in self.head_linears]
        full_embed = torch.cat(full_embed, dim=-1)
        projected = self.proj_head(full_embed)
        return projected + residual


class SpectreBranchEncoderLayer(nn.Module):
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
        self.d_model = d_model
        bias = True
        layer_norm_eps = 1e-5
        self.dropout = nn.Dropout(dropout)
        self.mix_layer = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_feedforward)
        self.linear3 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # self.mix_heads = SpectreMix(in_channels=d_model, num_heads=nhead, seq_length=seq_length)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation

    def forward(self, src):
        x = src
        old_x = x
        # x = torch.fft.fft2(x).real
        x = self.norm1(x) + old_x
        x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.linear1(x))
        x = self.linear2(x)
        # x = self.transform_norm(x)
        x = self.linear3(x)  # [B, N, E]
        return self.dropout2(x)


class SpectreBranchEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self, encoder_layer, num_patches: int, num_layers: int, norm=None, reduction=1
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.spectre_branch = SpectreFeatExtractor(
            3, encoder_layer.d_model, num_patches, reduction=1, num_stages=num_layers
        )
        self.spectre_project = nn.ModuleList([nn.Linear(768 * 2, 768) for _ in range(num_layers)])

    def forward(self, src: torch.Tensor, img: torch.Tensor):
        output = src
        # Extract spectre features
        _, feats = self.spectre_branch(img)

        for idx, mod in enumerate(self.layers):
            output = torch.cat([mod(output), feats[idx]], dim=-1)
            output = self.spectre_project[idx](output)

        if self.norm is not None:
            output = self.norm(output)

        return output + src


class SpectreFeatExtractor(nn.Module):
    def __init__(self, in_channels, embed_dim, num_tokens, reduction=1, num_stages=1) -> None:
        super().__init__()

        self.reduction = reduction
        self.net = nn.ModuleList([])
        prev_channels = in_channels
        channel_scale = 3
        for stage in range(num_stages):
            self.net.append(
                nn.Sequential(
                    # FFT(),
                    nn.Conv2d(prev_channels, prev_channels * channel_scale, 3, stride=1),
                    # nn.BatchNorm2d(prev_channels * channel_scale),
                )
            )
            prev_channels *= channel_scale

        self.project = nn.ModuleList([])

        prev_channels = in_channels * channel_scale
        for _ in range(num_stages):
            self.project.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, embed_dim, 1, stride=1),
                    nn.Flatten(start_dim=2),
                    nn.AdaptiveAvgPool1d(num_tokens),
                    Transpose((-2, -1)),
                )
            )
            prev_channels *= channel_scale

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # x = torch.fft.rfft2(x, dim=(-2, -1))
        x = torch.log1p(torch.abs(torch.fft.rfft2(x, dim=(-2, -1))))
        # x = torch.view_as_real(x)  # [B,C,H,W,2]
        # x = x.permute(0, 1, 4, 2, 3).flatten(1, 2)

        if self.reduction > 1:
            width = x.shape[2]
            height = x.shape[3]
            x = x[..., : height // self.reduction, : width // self.reduction]

        feats = []
        for idx, stage in enumerate(self.net):
            x = stage(x)
            feats.append(self.project[idx](x))

        return x, feats


class SpectreBranch(nn.Module):
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

        self.embeddings_block = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        )

        encoder_layer = SpectreBranchEncoderLayer(
            seq_length=num_patches + 1,
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
        )

        self.encoder_blocks = SpectreBranchEncoder(
            encoder_layer, num_patches + 1, num_layers=num_encoders
        )

        self.mlp_head = nn.Sequential(nn.Linear(embed_dim, num_classes))
        # self.cls_norm = nn.LayerNorm(100)

    def forward(self, x, return_features=False):
        img = x
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x, img)

        cls_token = x[:, 0, :]
        x = self.mlp_head(cls_token)
        if return_features:
            return x, cls_token
        return x
