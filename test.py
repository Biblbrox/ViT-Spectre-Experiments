# %% Performance checks
import time
import timeit

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import polars as pd
import torch
import torch.nn as nn
from numpy.testing import measure

from spectre_vit.configs.parser import parse_config
from spectre_vit.models.spectre.layers import SpectreLinear
from spectre_vit.models.spectre.spectre import SpectreViT

# %% Cell 2
# Model configuration
config_path = "spectre_vit/configs/spectre_vit.py"
config = parse_config(config_path)
random_seed = config.random_seed
batch_size = config.batch_size
epochs = config.epochs
learning_rate = config.learning_rate
num_classes = config.num_classes
patch_size = config.patch_size
img_size = config.img_size
in_channels = config.in_channels
num_heads = config.num_heads
dropout = config.dropout
hidden_dim = config.hidden_dim
adam_weight_decay = config.adam_weight_decay
adam_betas = config.adam_betas
activation = config.activation
num_encoders = config.num_encoders
embed_dim = config.embed_dim
num_patches = config.num_patches
use_spectre = config.use_spectre
spectre_threshold = config.spectre_threshold
method = config.method
experiment_name = "spectre_vit_fftmh16_spectrelayers_fusedheads"
use_distillation = False

# %%
model = SpectreViT(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    num_classes=num_classes,
    embed_dim=embed_dim,
    num_encoders=num_encoders,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    dropout=dropout,
    activation=activation,
    method=method,
)

# %% Overall performance check
input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
with torch.no_grad():
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)

    # Timing
    iterations = 100
    start_time = time.time()
    for _ in range(iterations):
        _ = model(input_tensor)
    end_time = time.time()
    total_time = end_time - start_time
    print(
        f"Average inference time over {iterations} iterations: {total_time / iterations * 1000:.2f} ms"
    )


# %% SpectreLinear performance check


dims = 2 ** np.array(np.linspace(8, 12, 4), dtype=np.int32)
iters = 50

# Check approximation
with torch.no_grad():
    total_time_cpu_spectre = []
    total_time_cpu_layer = []
    total_time_gpu_spectre = []
    total_time_gpu_layer = []
    total_spectre_params = []
    total_linear_params = []
    for dim in dims:
        input = torch.randn(batch_size, num_patches, int(dim))
        input_gpu = torch.randn(batch_size, num_patches, int(dim)).cuda()
        spectre_linear_gpu = SpectreLinear(dim, dim).cuda().eval()
        spectre_linear = SpectreLinear(dim, dim).eval()
        layer = nn.Linear(dim, dim)
        layer_gpu = nn.Linear(dim, dim).cuda()

        total_spectre_params.append(
            sum(p.numel() for p in spectre_linear.parameters() if p.requires_grad)
        )
        total_linear_params.append(sum(p.numel() for p in layer.parameters() if p.requires_grad))

        total_time_cpu_spectre.append(timeit.timeit(lambda: spectre_linear(input), number=iters))
        total_time_cpu_layer.append(timeit.timeit(lambda: layer(input), number=iters))
        total_time_gpu_spectre.append(
            timeit.timeit(lambda: spectre_linear_gpu(input_gpu), number=iters)
        )
        total_time_gpu_layer.append(timeit.timeit(lambda: layer_gpu(input_gpu), number=iters))

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
plt.ylabel("Time, ms")
