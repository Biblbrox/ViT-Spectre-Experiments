# %% Performance checks
import sys
import time
import timeit
from itertools import product

sys.path.append("../..")

import matplotlib.pyplot as plt
import numpy as np
import polars as pd
import torch
import torch.nn as nn
import torch.profiler as profiler

from spectre_vit.configs.parser import parse_config
from spectre_vit.models.spectre.spectre import SpectreEncoderLayer, SpectreViT
from spectre_vit.profile.parser import ProfilerParser

# %% Cell 2
# Model configuration
config_path = "spectre_vit/configs/spectre_vit_cifar100.py"
c = parse_config(config_path)
device = "cuda"
# %% Overall model performance evaluation for different heads and patch sizes
# ---------------------------------------------------------------------------
input_tensor = torch.rand(
    (c.batch_size, c.in_channels, c.img_size, c.img_size), dtype=torch.float32
).to(device)
with torch.no_grad():
    for patch, heads in product([4, 8], [1, 2, 4, 8]):
        model = (
            SpectreViT(
                img_size=c.img_size,
                patch_size=c.patch_size,
                in_channels=c.in_channels,
                num_classes=c.num_classes,
                embed_dim=c.embed_dim,
                num_encoders=c.num_encoders,
                num_heads=c.num_heads,
                hidden_dim=c.hidden_dim,
                dropout=c.dropout,
                activation=c.activation,
            )
            .to(device)
            .eval()
        )
        # Warm-up
        for _ in range(100):
            _ = model(input_tensor)

        # Timing
        iterations = 1000
        start_time = time.time()
        for _ in range(iterations):
            _ = model(input_tensor)
        end_time = time.time()
        total_time = end_time - start_time
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Average latency over {iterations} iterations for batch: {c.batch_size} with params: {params}, patch: {patch}, heads: {heads}: {total_time / iterations * 1000:.2f} ms"
        )


# %% SpectreLinear performance check
# ---------------------------------------------------------------------------
dims = 2 ** np.array(np.linspace(8, 12, 4), dtype=np.int32)
iters = 500


class SpectreLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
        if self.in_channels == self.out_channels:
            self.avg_pool = lambda x: x
        else:
            self.avg_pool = nn.AdaptiveAvgPool1d(out_channels)

    def forward(self, x, dim=(-1)):
        """
        x: [B,N,E]
        """
        return self.linear(x) + self.avg_pool(x)


# Check approximation
with torch.no_grad():
    total_time_cpu_spectre = []
    total_time_cpu_layer = []
    total_time_gpu_spectre = []
    total_time_gpu_layer = []
    total_spectre_params = []
    total_linear_params = []
    for dim in dims:
        input = torch.randn(c.batch_size, c.num_patches, int(dim))
        input_gpu = torch.randn(c.batch_size, c.num_patches, int(dim)).cuda()
        spectre_linear_gpu = SpectreLinear(dim, dim).cuda().eval()
        spectre_linear = SpectreLinear(dim, dim).eval()
        layer = nn.Linear(dim, dim).eval()
        layer_gpu = nn.Linear(dim, dim).cuda().eval()

        total_spectre_params.append(
            sum(p.numel() for p in spectre_linear.parameters() if p.requires_grad)
        )
        total_linear_params.append(sum(p.numel() for p in layer.parameters() if p.requires_grad))

        total_time_cpu_spectre.append(
            timeit.timeit(lambda: spectre_linear(input), number=iters, timer=timeit.default_timer)
        )
        total_time_cpu_layer.append(
            timeit.timeit(lambda: layer(input), number=iters, timer=timeit.default_timer)
        )
        total_time_gpu_spectre.append(
            timeit.timeit(
                lambda: spectre_linear_gpu(input_gpu), number=iters, timer=timeit.default_timer
            )
        )
        total_time_gpu_layer.append(
            timeit.timeit(lambda: layer_gpu(input_gpu), number=iters, timer=timeit.default_timer)
        )

    results = pd.DataFrame({
        "Implementation": ["SpectreLinear", "nn.Linear", "SpectreLinear (GPU)", "nn.Linear (GPU)"],
        "Avg Inference Time (ms)": [
            total_time_cpu_spectre,
            total_time_cpu_layer,
            total_time_gpu_spectre,
            total_time_gpu_layer,
        ],
    })

plt.plot(dims, total_time_cpu_spectre)
# plt.plot(dims, total_time_gpu_spectre)
plt.plot(dims, total_time_cpu_layer)
# plt.plot(dims, total_time_gpu_layer)
# plt.plot(dims, total_spectre_params)
# plt.plot(dims, total_linear_params)
# plt.legend(["Spectre(CPU)", "Spectre(GPU)", "nn.Linear(CPU)", "nn.Linear(GPU"])
plt.legend(["Spectre(CPU)", "nn.Linear(CPU)"])
# plt.legend(["Spectre Params(CPU)", "nn.Linear Params(CPU)"])
# plt.yscale("log")
# plt.legend(["Spectre(GPU)", "nn.Linear(GPU)"])
plt.xlabel("Matrix size")
plt.ylabel("Time, s")
plt.tight_layout()

# %% Encoder layer performance check
# ---------------------------------------------------------------------------
layer = SpectreEncoderLayer(
    seq_length=c.num_patches,
    d_model=c.embed_dim,
    nhead=c.num_heads,
    dim_feedforward=c.hidden_dim,
    dropout=c.dropout,
    activation=c.activation,
).cuda()

x = torch.randn(c.batch_size, c.num_patches, c.embed_dim, device="cuda")  # (B, T, D)
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=False,
    profile_memory=True,
) as prof:
    with profiler.record_function("SpectreEncoderLayer_forward"):
        out = layer(x)
parser = ProfilerParser(prof).remove_idle().add_percentages().round(3).sort_by_cuda()
parser.show()
parser.to_csv("plots/encoder_layer.csv")
