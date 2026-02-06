# %%
import math
import random
import timeit
from os import path

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

# %%


class FFTApproximator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, in_channels // 2 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# %%
# Training
batch_size = 160
epochs = 100000
learning_rate = 1e-3
embed_dim = 64
num_patches = 16
#
criterion = nn.MSELoss()
model = FFTApproximator(embed_dim).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    model.train()
    #
    inputs = torch.randn(batch_size, num_patches, embed_dim).cuda()
    targets = torch.fft.rfft2(inputs, dim=(-1)).real

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #
    # writer.add_scalar("Loss/train", loss.item(), epoch)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
# %% Check performance in comparation with torch.fft.rfft

batch_size = 16
num_patches = 16
# dims = range(2, 512)
dims = 2 ** np.array(np.linspace(8, 12, 4), dtype=np.int32)
iters = 5000

# Check approximation
with torch.no_grad():
    approx_gpu_time = []
    for dim in dims:
        input = torch.randn(batch_size, num_patches, int(dim)).cuda()
        fft_approx = FFTApproximator(input.shape[-2], -2).cuda().eval()
        time = timeit.timeit(lambda: fft_approx(input), number=iters)
        approx_gpu_time.append(time)

    approx_cpu_time = []
    for dim in dims:
        input = torch.randn(batch_size, num_patches, int(dim)).cpu()
        fft_approx = FFTApproximator(input.shape[-2], -2).cpu().eval()
        time = timeit.timeit(lambda: fft_approx(input), number=iters)
        approx_cpu_time.append(time)

    rfft_cpu_time = []
    for dim in dims:
        input = torch.randn(batch_size, num_patches, int(dim)).cpu()
        time = timeit.timeit(lambda: torch.fft.rfft(input, dim=-2), number=iters)
        rfft_cpu_time.append(time)

    rfft_gpu_time = []
    for dim in dims:
        input = torch.randn(batch_size, num_patches, int(dim)).cuda()
        time = timeit.timeit(lambda: torch.fft.rfft(input, dim=-2), number=iters)
        rfft_gpu_time.append(time)

    # fwht_cpu_time = []
    # for dim in dims:
    #     input = torch.randn(batch_size, num_patches, int(dim)).cpu()
    #     fwht = FFTApproximator(
    #         input.shape[-1], method='fwht').cpu().eval()
    #     time = timeit.timeit(lambda: fwht(input), number=iters)
    #     fwht_cpu_time.append(time)


plt.plot(dims, approx_gpu_time, "r")
plt.plot(dims, approx_cpu_time, "g")
plt.plot(dims, rfft_gpu_time, "b")
plt.plot(dims, rfft_cpu_time, "c")
# plt.plot(dims, fwht_cpu_time, 'k')
plt.yscale("log")
plt.title("Performance comparation (PyTorch)")
plt.xlabel("Dimension")
plt.ylabel("Time, us")
plt.legend(["Approx (GPU)", "Approx (CPU)", "RFFT (GPU)", "RFFT (CPU)"])

# %% ONNX performance benchmark

batch_size = 16
num_patches = 16
dims = 2 ** np.array(np.linspace(1, 12, 12), dtype=np.int32) + 1
iters = 500

# Convert models
for dim in dims:
    input_tensor = torch.rand((batch_size, num_patches, int(dim)), dtype=torch.float32)
    fft_approx = FFTApproximator(int(dim), -2)
    approx_model = f"export/fft_approx{int(dim)}.onnx"
    fft_rfft = FFTApproximator(int(dim), -2)
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


def inference_approx(x, ort_session):
    return ort_session.run(None, {"input": x})


# Check approximation
with torch.no_grad():
    approx_cpu_time = []
    for dim in dims:
        input = np.ones((batch_size, num_patches, int(dim)), dtype=np.float32)
        approx_model = f"export/fft_approx{int(dim)}.onnx"
        fft_approx = ort.InferenceSession(approx_model)
        time = timeit.timeit(lambda: inference_approx(input, fft_approx), number=iters)
        approx_cpu_time.append(time)

    rfft_cpu_time = []
    for dim in dims:
        input = np.ones((batch_size, num_patches, int(dim)), dtype=np.float32)
        rfft_model = f"export/fft_rfft{int(dim)}.onnx"
        fft_rfft = ort.InferenceSession(rfft_model)
        time = timeit.timeit(lambda: inference_approx(input, fft_rfft), number=iters)
        rfft_cpu_time.append(time)


plt.plot(dims, approx_cpu_time, "g")
plt.plot(dims, rfft_cpu_time, "c")
plt.title("Performance comparation (PyTorch)")
plt.xlabel("Dimension")
plt.ylabel("Time, us")
plt.legend(["Approx (CPU)", "RFFT (CPU)"])

# %% Onnx not power of 2

batch_size = 16
num_patches = 16
dims = 2 ** np.array(np.linspace(1, 12, 12), dtype=np.int32)
iters = 500

# Convert models
for dim in dims:
    input_tensor = torch.rand((batch_size, num_patches, int(dim)), dtype=torch.float32)
    fft_approx = FFTApproximator(int(dim), -2)
    approx_model = f"export/fft_approx{int(dim)}.onnx"
    fft_rfft = FFTApproximator(int(dim), -2)
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


def inference_approx(x, ort_session):
    return ort_session.run(None, {"input": x})


# Check approximation
with torch.no_grad():
    approx_cpu_time = []
    for dim in dims:
        input = np.ones((batch_size, num_patches, int(dim)), dtype=np.float32)
        approx_model = f"export/fft_approx{int(dim)}.onnx"
        fft_approx = ort.InferenceSession(approx_model)
        time = timeit.timeit(lambda: inference_approx(input, fft_approx), number=iters)
        approx_cpu_time.append(time)

    rfft_cpu_time = []
    for dim in dims:
        input = np.ones((batch_size, num_patches, int(dim)), dtype=np.float32)
        rfft_model = f"export/fft_rfft{int(dim)}.onnx"
        fft_rfft = ort.InferenceSession(rfft_model)
        time = timeit.timeit(lambda: inference_approx(input, fft_rfft), number=iters)
        rfft_cpu_time.append(time)


plt.plot(dims, approx_cpu_time, "g")
plt.plot(dims, rfft_cpu_time, "c")
plt.title("Performance comparation (ONNX)")
plt.xlabel("Dimension")
plt.ylabel("Time, us")
plt.legend(["Approx (CPU)", "RFFT (CPU)"])
