import torch.nn as nn
import torch

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
