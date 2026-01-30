from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from spectre_vit.modules.patch_embeddings import PatchEmbedding


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
        method="attention",
    ):
        super().__init__()

        num_patches = (img_size // patch_size) ** 2

        self.embeddings_block = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
        )

        self.encoder_blocks = TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(nn.Linear(embed_dim, num_classes, 5))
        # self.cls_norm = nn.LayerNorm(100)

    def forward(self, x, return_features=False):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)

        cls_token = x[:, 0, :]
        x = self.mlp_head(cls_token)
        if return_features:
            return x, cls_token
        return x
