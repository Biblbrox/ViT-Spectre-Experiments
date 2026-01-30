import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def fwht(x, dim=-1, normalize=True):
    """Vectorized Fast Walsh–Hadamard Transform (no in-place ops)."""
    n = x.size(dim)
    x = x.transpose(dim, -1).contiguous()
    orig_shape = x.shape
    x = x.view(-1, n)

    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.cat((a + b, a - b), dim=2)
        h *= 2

    x = x.view(orig_shape).transpose(dim, -1)

    if normalize:
        x = x * (n**-0.5)

    return x


# def fwht_fast(x, stages, dim=-1, normalize=True):
#    n = x.size(dim)

#    x = x.transpose(dim, -1)
#    orig_shape = x.shape
#    x = x.reshape(-1, n)

#    h = 1
#    for _ in range(stages):
#        x = x.reshape(-1, n // (2*h), 2, h)
#        a = x[:, :, 0, :]
#        b = x[:, :, 1, :]
#        x = torch.stack((a + b, a - b), dim=2)
#        h *= 2

#    x = x.reshape(orig_shape).transpose(dim, -1)

#    if normalize:
#        x = x * (n ** -0.5)

#    return x


def fwht_fast(x):
    """
    x: [..., N] where N is power of 2
    """
    orig_shape = x.shape
    N = orig_shape[-1]

    x = x.reshape(-1, N)  # flatten batch dims

    h = 1
    while h < N:
        x = x.view(x.shape[0], -1, 2 * h)

        a = x[..., :h]
        b = x[..., h : 2 * h]

        # butterfly
        x = (a + b).repeat_interleave(2, dim=-1)
        x[..., 1::2] = a - b

        h *= 2

    return x.reshape(orig_shape)


def hadamard_transform(x: torch.Tensor):
    """Fast Walsh–Hadamard transform

    The hadamard transform is not numerically stable by nature (lots of subtractions),
    it is recommended to use with float64 when possible

    :param x: Either a vector or a batch of vectors where the first dimension is the batch dimension.
              Each vector's length is expected to be a power of 2! (or each row if it is batched)
    :return: The normalized Hadamard transform of each vector in x
    """
    original_shape = x.shape
    assert 1 <= len(original_shape) <= 2, "input's dimension must be either 1 or 2"
    if len(original_shape) == 1:
        # add fake 1 batch dimension
        # for making the code a follow a single (batched) path
        x = x.unsqueeze(0)
    batch_dim, d = x.shape

    h = 2
    while h <= d:
        hf = h // 2
        x = x.view(batch_dim, d // h, h)

        half_1, half_2 = x[:, :, :hf], x[:, :, hf:]

        x = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)

        h *= 2

    return (x / math.sqrt(d)).view(*original_shape)


class LearnableHadamard(nn.Module):
    def __init__(self, dim, num_blocks=2):
        super().__init__()
        self.orig_dim = dim
        self.dim = next_pow2(dim)  # internal power-of-2 dim
        self.pad = self.dim - dim

        self.params = nn.ParameterList([
            nn.Parameter(torch.ones(self.dim)) for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x: [..., D]
        residual = x

        # Pad if needed
        if self.pad > 0:
            x = F.pad(x, (0, self.pad))

        # Hadamard blocks
        for p in self.params:
            x = fwht_fast(x)  # * p

        # Crop back
        x = x[..., : self.orig_dim]

        return x + residual
