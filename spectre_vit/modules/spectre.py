import torch
import torch.nn as nn


class FFT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        # dim = x.shape[-1]
        #
        # assert (dim & (dim - 1) == 0) and dim != 0

        return torch.fft.rfft(x, dim=(-1)).real
