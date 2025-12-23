import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()

        self.embed_dim = embed_dim

        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Flatten(2)  # [B, E, N]
        )

        # FIXED: correct CLS token shape
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B = x.shape[0]

        x = self.patcher(x)           # [B, E, N]
        x = x.permute(0, 2, 1)        # [B, N, E]

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.position_embeddings
        x = self.dropout(x)

        return x


def transform(x, method='fft'):
    if method == 'fft':
        return torch.fft.fft2(x, dim=(-1, -2)).real
    else:
        raise NotImplementedError(f"Transform method '{method}' is not implemented.")


class FNetEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout,
                 activation,
                 batch_first,
                 norm_first,
                 use_spectre=False):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first
        )
        self.use_spectre = use_spectre

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.use_spectre:
            if self.norm_first:
                x = x + transform(self.norm1(x))
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + transform(x))
                x = self.norm2(x + self._ff_block(x))
        else:
            if self.norm_first:
                x = x + self._sa_block(
                    self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
                )
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(
                    x
                    + self._sa_block(x, src_mask,
                                     src_key_padding_mask, is_causal=is_causal)
                )
                x = self.norm2(x + self._ff_block(x))

        return x


# class FNet(nn.TransformerEncoder):
#     def __init__(
#         self, d_model=256, expansion_factor=2, dropout=0.5, num_layers=6,
#     ):
#         encoder_layer = FNetEncoderLayer(d_model, expansion_factor, dropout)
#         super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x


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
        use_spectre=False
    ):
        super().__init__()

        num_patches = (img_size // patch_size) ** 2

        self.embeddings_block = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        )

        encoder_layer = FNetEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
            use_spectre=use_spectre)

        self.encoder_blocks = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoders
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])  # CLS token
        return x
