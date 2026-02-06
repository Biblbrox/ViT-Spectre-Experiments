# %%
import math
import random
import time
import timeit
from os import path

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.tensorboard import SummaryWriter

from spectre_vit.models.spectre.hadamar import fwht_fast
from spectre_vit.models.spectre.layers import SpectreLinear

# %%


class FNetAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.fft.fft2(x, dim=(-2, -1))


class MHPermutMix(nn.Module):
    def __init__(self, embed_dim: int, token_dim: int, num_heads: int, out_channels: int):
        super().__init__()
        d = embed_dim * token_dim
        self.num_heads = num_heads
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.concat_dim = self.embed_dim * self.num_heads
        signs = torch.randint(0, 2, (num_heads, d), dtype=torch.float32)
        signs = signs * 2 - 1
        self.register_buffer("signs", signs.unsqueeze(0))
        perms = torch.stack([torch.randperm(d) for _ in range(num_heads)])
        self.register_buffer("perms", perms)
        self.linear = SpectreLinear(embed_dim * num_heads, out_channels)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)
        x = x[:, self.perms] * self.signs
        x = x.view(B, self.token_dim, self.concat_dim)
        return x
        # return self.linear(x)


# %% Visualize orhtogonal features
batch_size = 16
num_patches = 64
embed_dim = 512
num_heads = 8
input = torch.randn(batch_size, num_patches, embed_dim)
mh_permut = MHPermutMix(embed_dim, num_patches, num_heads, embed_dim)
with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_modules=True) as prof:
    vecs = mh_permut(input)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(vecs[0].shape)
print(vecs[0].flatten(0).shape)
# print(
#     torch.dot(
#         vecs.view(batch_size, num_patches, embed_dim * num_heads)[..., :embed_dim].flatten(0),
#         vecs.view(batch_size, num_patches, embed_dim * num_heads)[
#             ..., embed_dim : embed_dim * 2
#         ].flatten(0),
#     )
# )

# %% Check performance in comparation with torch.fft.rfft

batch_size = 16
num_patches = 32
num_heads = 8
# dims = range(2, 512)
dims = 2 ** np.array(np.linspace(4, 13, 9), dtype=np.int32)
iters = 500

# Check approximation
with torch.no_grad():
    approx_gpu_time = []
    for dim in dims:
        input = torch.randn(batch_size, num_patches, int(dim)).cuda()
        perm = MHPermutMix(input.shape[-1], input.shape[-2], num_heads, embed_dim).cuda().eval()
        approx_gpu_time.append(
            timeit.timeit(lambda: perm(input), number=iters, timer=timeit.default_timer) / iters
        )

    approx_cpu_time = []
    for dim in dims:
        input = torch.randn(batch_size, num_patches, int(dim)).cpu()
        perm = MHPermutMix(input.shape[-1], input.shape[-2], num_heads, embed_dim).cpu().eval()
        approx_cpu_time.append(
            timeit.timeit(lambda: perm(input), number=iters, timer=timeit.default_timer) / iters
        )

    rfft_cpu_time = []
    for dim in dims:
        input = torch.randn(batch_size, num_patches, int(dim)).cpu()
        fft = FNetAttention().cpu().eval()
        rfft_cpu_time.append(
            timeit.timeit(lambda: fft(input), number=iters, timer=timeit.default_timer) / iters
        )

    rfft_gpu_time = []
    for dim in dims:
        input = torch.randn(batch_size, num_patches, int(dim)).cuda()
        fft = FNetAttention().cuda().eval()
        rfft_gpu_time.append(
            timeit.timeit(lambda: fft(input), number=iters, timer=timeit.default_timer) / iters
        )

plt.rcParams.update({"font.size": 12})
plt.plot(dims, approx_gpu_time, "r")
plt.plot(dims, approx_cpu_time, "g")
plt.plot(dims, rfft_gpu_time, "b")
plt.plot(dims, rfft_cpu_time, "c")
plt.plot(dims, approx_gpu_time, "rx")
plt.plot(dims, approx_cpu_time, "gx")
plt.plot(dims, rfft_gpu_time, "bx")
plt.plot(dims, rfft_cpu_time, "cx")
plt.yscale("log")
plt.xlabel("Dimension")
plt.ylabel("Time, s")
plt.grid()
plt.legend(["MHPermutMix (GPU)", "MHPermutMix (CPU)", "2D FFT (GPU)", "2D FFT (CPU)"])
plt.tight_layout()
plt.savefig(f"plots/pytorch_spectremix_h{num_heads}.png")

# %% ONNX performance benchmark
num_heads = 1
dims = 2 ** np.array(np.linspace(4, 13, 9), dtype=np.int32)

# Convert models
for dim in dims:
    input_tensor = torch.rand((batch_size, num_patches, int(dim)), dtype=torch.float32)
    fft_approx = MHPermutMix(int(dim), num_patches, num_heads, int(dim))
    approx_model = f"export/fft_approx{int(dim)}.onnx"
    fft_rfft = FNetAttention()
    rfft_model = f"export/fft_rfft{int(dim)}.onnx"
    torch.onnx.export(
        fft_approx,
        (input_tensor,),
        approx_model,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
        external_data=False,
    )
    torch.onnx.export(
        fft_rfft,
        (input_tensor,),
        rfft_model,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
        external_data=False,
    )


providers = [
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
    "CPUExecutionProvider",
]


def make_session(path):
    return ort.InferenceSession(path, providers=providers)


def inference_approx(x, ort_session):
    return ort_session.run(None, {"input": x})


def inference_cuda(input_np, session):
    ort_inputs = {session.get_inputs()[0].name: input_np}
    session.run(None, ort_inputs)


def benchmark_cuda(session, input_np, iters=50, warmup=10):
    for _ in range(warmup):
        inference_cuda(input_np, session)

    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        inference_cuda(input_np, session)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / iters


# Check approximation
with torch.no_grad():
    approx_cpu_time = []
    for dim in dims:
        input = np.ones((batch_size, num_patches, int(dim)), dtype=np.float32)
        approx_model = f"export/fft_approx{int(dim)}.onnx"
        fft_approx = ort.InferenceSession(approx_model)
        approx_cpu_time.append(
            timeit.timeit(
                lambda: inference_approx(input, fft_approx),
                number=iters,
                timer=timeit.default_timer,
            )
            / iters
        )

    rfft_cpu_time = []
    for dim in dims:
        input = np.ones((batch_size, num_patches, int(dim)), dtype=np.float32)
        rfft_model = f"export/fft_rfft{int(dim)}.onnx"
        fft_rfft = ort.InferenceSession(rfft_model)
        rfft_cpu_time.append(
            timeit.timeit(
                lambda: inference_approx(input, fft_rfft), number=iters, timer=timeit.default_timer
            )
            / iters
        )

# plt.plot(dims, approx_cpu_time, "g")
# plt.plot(dims, rfft_cpu_time, "c")
# plt.xlabel("Dimension")
# plt.ylabel("Time, s")
# plt.grid()
# plt.legend(["MHPermuteMix (CPU)", "2D FFT (CPU)"])
# plt.tight_layout()
# plt.savefig(f"plots/onnx_spectremix_h{num_heads}.png")

plt.figure(figsize=(7, 5))
plt.plot(dims, approx_cpu_time, "g")
plt.plot(dims, rfft_cpu_time, "c")
plt.xlabel("Dimension")
plt.ylabel("Time per inference (s)")
plt.grid()

plt.legend([
    "MHPermuteMix (CPU)",
    "2D FFT (CPU)",
])

plt.tight_layout()
plt.savefig(f"plots/onnx_spectremix_h{num_heads}.png")

# %% Onnx not power of 2
dims = 2 ** np.array(np.linspace(1, 12, 12), dtype=np.int32) + 1

# Convert models
for dim in dims:
    input_tensor = torch.rand((batch_size, num_patches, int(dim)), dtype=torch.float32)
    fft_approx = MHPermutMix(int(dim), num_patches, num_heads, int(dim))
    approx_model = f"export/fft_approx{int(dim)}.onnx"
    fft_rfft = FNetAttention()
    rfft_model = f"export/fft_rfft{int(dim)}.onnx"
    if not path.exists(approx_model):
        torch.onnx.export(
            fft_approx,
            (input_tensor,),
            approx_model,
            input_names=["input"],
            output_names=["output"],
            dynamo=True,
            external_data=False,
        )
    if not path.exists(rfft_model):
        torch.onnx.export(
            fft_rfft,
            (input_tensor,),
            rfft_model,
            input_names=["input"],
            output_names=["output"],
            dynamo=True,
            external_data=False,
        )


# Check approximation
with torch.no_grad():
    approx_cpu_time = []
    for dim in dims:
        input = np.ones((batch_size, num_patches, int(dim)), dtype=np.float32)
        approx_model = f"export/fft_approx{int(dim)}.onnx"
        fft_approx = ort.InferenceSession(approx_model)
        time = timeit.timeit(
            lambda: inference_approx(input, fft_approx), number=iters, timer=timeit.default_timer
        )
        approx_cpu_time.append(time / iters)

    rfft_cpu_time = []
    for dim in dims:
        input = np.ones((batch_size, num_patches, int(dim)), dtype=np.float32)
        rfft_model = f"export/fft_rfft{int(dim)}.onnx"
        fft_rfft = ort.InferenceSession(rfft_model)
        time = timeit.timeit(
            lambda: inference_approx(input, fft_rfft), number=iters, timer=timeit.default_timer
        )
        rfft_cpu_time.append(time / iters)


plt.plot(dims, approx_cpu_time, "g")
plt.plot(dims, rfft_cpu_time, "c")
plt.title("Performance comparation (ONNX)")
plt.xlabel("Dimension")
plt.ylabel("Time, us")
plt.legend(["Approx (CPU)", "RFFT (CPU)"])
