import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
from pytorch_wavelets import DWT1DForward, DWT1DInverse, DWTForward, DWTInverse


def transform(x: torch.Tensor, dims, method='fft', pad=False):
    if method == 'fft':
        if pad:
            # F.pad(x,)
            pass
        return torch.fft.fft(x, dims=dims)
    elif method == 'hadamar':
        pass


class SpectreLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, scale=2):
        super().__init__()
        self.win_size = in_channels // scale
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.scale = scale

        self.head = nn.Sequential(
            nn.Linear(in_channels // 2 + 1, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )
        self.fft_params = nn.Parameter(torch.ones(in_channels // 2 + 1))
        self.avg_pool = nn.AdaptiveAvgPool1d(out_channels)

    def forward(self, x):
        """
        x: [B,N,E]
        """

        residual = x
        x = torch.fft.rfft(x, dim=(-1))
        x = x.real * self.fft_params
        # x = x[..., :self.win_size]

        x = self.head(x) + self.avg_pool(residual)

        return x


class SpectreMix(nn.Module):
    def __init__(self, in_channels, num_heads, seq_length, method='fft'):
        super().__init__()

        self.num_heads = num_heads
        self.in_channels = in_channels
        self.method = method

        if method == 'fft_bare':
            pass
        elif method == 'fft_mh':
            self.head_linears = nn.ModuleList([
                nn.Linear(in_channels, in_channels) for _ in range(num_heads)
            ])
        elif method == 'fft_mh_spectrelayers':
            scale = 2
            self.head_linears = nn.ModuleList([
                SpectreLinear(in_channels, in_channels // 6, scale=scale) for _ in range(num_heads)
            ])
            self.common_linear = SpectreLinear(
                in_channels, in_channels // scale, scale=scale)
            self.token_heads = nn.ModuleList([
                nn.Sequential(
                    SpectreLinear(seq_length, seq_length * 2),
                    SpectreLinear(seq_length * 2, seq_length)

                ) for _ in range(num_heads)])
            self.proj_head = nn.Sequential(
                SpectreLinear(
                    in_channels // 6 * num_heads, in_channels // 6 * num_heads // 2, bias=True, scale=scale),
                SpectreLinear(
                    in_channels // 6 * num_heads // 2, in_channels, bias=True, scale=scale),
            )
        elif method == 'fft_param':
            self.param = [nn.Parameter(torch.ones(seq_length, 1))
                          for _ in range(num_heads)]
        elif method == 'dwt_embed':
            self.dwt = DWT1DForward(num_heads, wave='haar', mode='zero')
            self.head_linear = nn.ModuleList([
                nn.Linear(in_channels // 2**(i + 1), in_channels) for i in range(num_heads)
            ])
        elif method == 'dwt_token':
            self.dwt = DWT1DForward(num_heads, wave='haar', mode='zero')
            self.head_linear = nn.ModuleList([
                nn.Linear(seq_length // 2**(i + 1) + 1, seq_length) for i in range(num_heads)
            ])

    def forward(self, x):
        """
        x: [B, N, E]
        """
        if self.method == 'fft_bare':
            return torch.fft.fft2(x, dim=(-2, -1)).real
        elif self.method == 'fft_param':
            feat = None
            for i in range(self.num_heads):
                if feat is None:
                    feat = torch.fft.fft2(x, dim=(-2, -1)).real * self.param[i]
                else:
                    feat += torch.fft.fft2(x, dim=(-2, -1)
                                           ).real * self.param[i]
            return feat
        elif self.method == 'fft_mh' or self.method == 'fft_mh_spectrelayers':
            full_embed = []
            head_embed = self.common_linear(x)
            for i in range(self.num_heads):
                head_embed = self.head_linears[i](x)
                head_embed = self.token_heads[i](
                    head_embed.transpose(1, 2)).transpose(1, 2)
                full_embed.append(head_embed)

            full_embed = torch.cat(full_embed, dim=-1)
            projected = self.proj_head(full_embed)
            return projected
        elif self.method == 'dwt_embed':
            approx, detail = self.dwt(x)
            full_embed = None
            for i in range(self.num_heads):
                head_embed = self.head_linear[i](detail[i])
                if full_embed is None:
                    full_embed = head_embed
                else:
                    full_embed += head_embed
            return full_embed
        elif self.method == 'dwt_token':
            x = x.transpose(2, 1)
            approx, detail = self.dwt(x)
            full_embed = None
            for i in range(self.num_heads):
                head_embed = self.head_linear[i](detail[i])
                head_embed = head_embed.transpose(2, 1)
                if full_embed is None:
                    full_embed = head_embed
                else:
                    full_embed += head_embed
            return full_embed


class SpectreEncoderLayer(nn.Module):
    """Spectre encoding layer. The following configurations are supported:
    - fft_bare - FNet implementation
    - fft_mh - Multi-Head fft with individials linear layers for each head
    - dwt_embed - Wavelet transform in embedding dimension
    - dwt_token - Wavelet transform in token dimension
    - attention - Native ViT Self-Attention
    """

    def __init__(self,
                 seq_length,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout,
                 activation,
                 method='fft_mh_spectrelayers'):
        super().__init__()
        assert method in ['fft_bare', 'fft_mh', 'fft_param',
                          'dwt_embed', 'dwt_token', 'fft_mh_spectrelayers']
        self.method = method
        bias = True
        layer_norm_eps = 1e-5
        factory_kwargs = {"device": 'cuda', "dtype": None}
        self.mix_layer = SpectreMix(
            d_model, nhead, seq_length, method=method)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = SpectreLinear(d_model, dim_feedforward,
                                     bias=bias)
        self.linear2 = SpectreLinear(
            dim_feedforward, dim_feedforward, bias=bias)
        self.linear3 = SpectreLinear(
            dim_feedforward, d_model, bias)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps,
                                  bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps,
                                  bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation

    def forward(self, src):
        x = src
        old_x = x
        x = self.mix_layer(x)
        x = self.norm1(x) + old_x
        x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.linear1(x))
        x = self.linear2(x)
        # x = self.transform_norm(x)
        x = self.linear3(x)
        return self.dropout2(x)


class SpectreEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        norm=None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
    ):
        output = src
        for idx, mod in enumerate(self.layers):
            output = mod(
                output,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output + src


class OrthoLearnedHaar1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta = nn.Parameter(torch.zeros(channels))

    def _filters(self):
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)

        lp = torch.stack([c, s], dim=1)      # [C,2]
        hp = torch.stack([s, -c], dim=1)     # [C,2]

        w = torch.cat([lp, hp], dim=0)       # [2C,2]
        return w.unsqueeze(1)                # [2C,1,2]

    def forward(self, x):
        """
        x: [B,C,N]
        returns: low, high, pad
        """
        B, C, N = x.shape
        pad = (N % 2 == 1)
        if pad:
            x = F.pad(x, (0, 1))

        w = self._filters()
        y = F.conv1d(x, w, stride=2, groups=C)
        low, high = y.chunk(2, dim=1)
        return low, high, pad


class OrthoLearnedHaarInverse1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta = nn.Parameter(torch.zeros(channels))

    def _filters(self):
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)

        lp = torch.stack([c, s], dim=1)
        hp = torch.stack([s, -c], dim=1)

        w = torch.cat([lp, hp], dim=0)
        return w.unsqueeze(1)

    def forward(self, low, high, pad=False):
        x = torch.cat([low, high], dim=1)
        w = self._filters()

        out = F.conv_transpose1d(
            x, w, stride=2, groups=self.channels
        )

        if pad:
            out = out[..., :-1]
        return out


class LearnedHaarConv1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 2 filters per channel: low + high
        self.weight = nn.Parameter(
            torch.randn(2 * channels, 1, 2) * 0.1
        )

    def forward(self, x):
        """
        x: [B, C, N]
        returns: low, high, pad
        """
        B, C, N = x.shape
        pad = (N % 2 == 1)

        if pad:
            x = F.pad(x, (0, 1))

        y = F.conv1d(
            x, self.weight,
            stride=2, groups=self.channels
        )

        low, high = y.chunk(2, dim=1)
        return low, high, pad


class LearnedHaarInverseConv1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.weight = nn.Parameter(
            torch.randn(2 * channels, 1, 2) * 0.1
        )

    def forward(self, low, high, pad=False):
        """
        low, high: [B,C,L]
        """
        x = torch.cat([low, high], dim=1)

        out = F.conv_transpose1d(
            x, self.weight,
            stride=2, groups=self.channels
        )

        if pad:
            out = out[..., :-1]

        return out


class HaarInverseConv1D(nn.Module):
    def __init__(self, channels):
        super().__init__()

        lp = torch.tensor([1., 1.]) / 2**0.5
        hp = torch.tensor([1., -1.]) / 2**0.5

        weight = torch.stack([lp, hp]).unsqueeze(1)  # [2,1,2]
        self.register_buffer("weight", weight)
        self.channels = channels

    def forward(self, low, high, pad=False):
        """
        low, high: [B, C, L]
        returns:   [B, C, 2L]
        """
        x = torch.cat([low, high], dim=1)   # [B, 2C, L]
        w = self.weight.repeat(self.channels, 1, 1)

        out = F.conv_transpose1d(
            x, w, stride=2, groups=self.channels
        )

        if pad:
            out = out[..., :-1]

        return out


class MultiScaleHaar1D(nn.Module):
    def __init__(self, channels, J):
        super().__init__()
        self.J = J

        # ONE shared transform
        self.haar = LearnedHaarConv1D(channels)
        self.ihaar = LearnedHaarInverseConv1D(channels)

    def forward(self, x):
        """
        x: [B,C,N]
        returns:
          low_J: lowest resolution
          highs: list of detail coeffs [H1, H2, ... HJ]
          pads:  list of padding flags
        """
        lows = x
        highs = []
        pads = []

        for _ in range(self.J):
            low, high, pad = self.haar(lows)
            highs.append(high)
            pads.append(pad)
            lows = low

        return lows, highs, pads

    def inverse(self, low, highs, pads):
        x = low
        for high, pad in reversed(list(zip(highs, pads))):
            x = self.ihaar(x, high, pad)
        return x


class GlobalWindowPredictor(nn.Module):
    def __init__(self, embed_dim, num_bins):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_bins)
        )

    def forward(self, x):
        """
        x: [B,N,E]
        returns: [K] probability over positions
        """
        g = x.mean(dim=(0, 1))          # [E]
        logits = self.net(g)          # [K]
        return torch.softmax(logits, dim=-1)


class SpectralMask(nn.Module):
    def __init__(self, n_freq):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.zeros(n_freq))

    def forward(self, x):
        # x: [B,N,E]
        X = torch.fft.fft(x, dim=-1).real

        m = torch.sigmoid(self.mask_logits)      # [F]
        X = X * m[None, None, :]                 # broadcast

        return X
