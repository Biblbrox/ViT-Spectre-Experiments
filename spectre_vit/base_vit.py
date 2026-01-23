from spectre_vit.spectre import SpectreLinear, SpectreEncoderLayer, SpectreEncoder
import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()

        self.embed_dim = embed_dim

        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2)  # [B, E, N]
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # if self.training:
        #    x = random_mask_pixels_batch(x, 200)

        B = x.shape[0]

        x = self.patcher(x)           # [B, E, N]
        x = x.permute(0, 2, 1)        # [B, N, E]

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.position_embeddings
        x = self.dropout(x)

        return x


class ViT(nn.Module):
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
        method="attention"
    ):
        super().__init__()

        num_patches = (img_size // patch_size) ** 2

        self.embeddings_block = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        )

        encoder_layer = SpectreEncoderLayer(
            seq_length=num_patches + 1,
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            method=method)

        self.encoder_blocks = SpectreEncoder(
            encoder_layer, num_layers=num_encoders
        )

        self.mlp_head = nn.Sequential(
            SpectreLinear(embed_dim, num_classes)
        )
        # self.cls_norm = nn.LayerNorm(100)

    def forward(self, x, return_features=False):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)

        cls_token = x[:, 0, :]
        x = self.mlp_head(cls_token)
        if return_features:
            return x, cls_token
        return x
