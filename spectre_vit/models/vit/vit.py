from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from spectre_vit.modules.patch_embeddings import PatchEmbedding


class ViT(nn.Module):
    def __init__(self, c):
        super().__init__()

        num_patches = (c.img_size // c.patch_size) ** 2

        self.embeddings_block = PatchEmbedding(
            c.embed_dim, c.patch_size, num_patches, c.dropout, c.in_channels
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=c.embed_dim,
            nhead=c.num_heads,
            dim_feedforward=c.hidden_dim,
            dropout=c.dropout,
            activation=c.activation,
        )

        self.encoder_blocks = TransformerEncoder(encoder_layer, num_layers=c.num_encoders)

        self.mlp_head = nn.Sequential(nn.Linear(c.embed_dim, c.num_classes, 5))
        # self.cls_norm = nn.LayerNorm(100)

    def forward(self, x, return_features=False):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)

        cls_token = x[:, 0, :]
        x = self.mlp_head(cls_token)
        if return_features:
            return x, cls_token
        return x
