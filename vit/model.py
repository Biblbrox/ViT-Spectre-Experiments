from sklearn.externals.array_api_compat.torch import layer_norm
import torch
from torch import nn
from torch.nn import functional as F
import warnings
import copy
from torch.nn.modules.transformer import _detect_is_causal_mask, _get_clones, _get_activation_fn, _get_seq_len
from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()

        self.embed_dim = embed_dim

        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Flatten(2)  # [B, E, N]
        )

        # FIXED: correct CLS token shape
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B = x.shape[0]

        x = self.patcher(x)           # [B, E, N]
        x = x.permute(0, 2, 1)        # [B, N, E]

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.position_embeddings
        x = self.dropout(x)

        return x


def transform(x, method='fft', threshold_percentage=None, dims=(-1, -2)):
    if method == 'fft':
        fft = torch.fft.fft2(x, dim=dims).real
        # torch.save(fft, 'fft_tensor.pt')
        if threshold_percentage is not None:
            # Shrink the FFT embedded dimensions
            embed_dim = fft.shape[-1]
            return fft[:, :, :int(embed_dim * threshold_percentage)]
        return fft
    elif method == 'ifft':
        return torch.fft.ifft2(x, dim=dims).real
    elif method == 'dwt':
        dwt = DWT1DForward(J=1, mode='zero', wave='db9').to(x.device)
        Yl, Yh = dwt(x)
        # Y = torch.cat((Yl, Yh[0]), dim=-1)
        if threshold_percentage is not None:
            embed_dim = Yl.shape[-1]
            return Yl[:, :, :int(embed_dim * threshold_percentage)]
        return Yl
    else:
        raise NotImplementedError(
            f"Transform method '{method}' is not implemented.")


class SpectreLayer(nn.Module):
    def __init__(self, seq_length, threshold_percentage, dims=(-2, -1)):
        super().__init__()
        self.seq_length = seq_length
        self.threshold_percentage = threshold_percentage
        self.transform_weights = torch.nn.Parameter(torch.randn(seq_length, 1))
        self.dims = dims

    def forward(self, x, method='fft'):
        return transform(x, method=method, threshold_percentage=self.threshold_percentage, dims=self.dims) * self.transform_weights


class FNetEncoderLayer(nn.Module):
    def __init__(self,
                 seq_length,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout,
                 activation,
                 batch_first,
                 norm_first,
                 use_spectre=False,
                 spectre_threshold=None):
        super().__init__()

        self.use_spectre = use_spectre
        self.spectre_threshold = spectre_threshold

        if use_spectre and spectre_threshold is not None:
            self.original_d_model = d_model
            d_model = int(d_model * spectre_threshold)

        bias = True
        layer_norm_eps = 1e-5
        self.transform_norm = nn.LayerNorm(dim_feedforward, eps=layer_norm_eps,
                                           bias=bias)

        factory_kwargs = {"device": 'cuda', "dtype": None}
        if not self.use_spectre:
            self.self_attn = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                bias=bias,
                batch_first=batch_first,
                **factory_kwargs,
            )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward,
                                 bias=bias, **factory_kwargs)
        self.linear = nn.Linear(d_model, d_model,
                                bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model,
                                 bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps,
                                  bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps,
                                  bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.spectre_layer = SpectreLayer(
            seq_length=seq_length, threshold_percentage=spectre_threshold, dims=(-2, -1))
        self.final_spectre_layer = SpectreLayer(
            seq_length=seq_length, threshold_percentage=spectre_threshold, dims=(-2, -1))

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.use_spectre:
            spectre_threshold = self.spectre_threshold if x.shape[-1] == self.original_d_model else None
            skip_x = x[:, :, :int(
                x.shape[-1] * spectre_threshold)] if spectre_threshold is not None else x
            x = self.norm1(skip_x + self.spectre_layer(x, method='fft'))
            x = self.norm2(x + self._ff_block(x))
        else:
            if self.norm_first:
                x += self._sa_block(
                    self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
                )
                x += self._ff_block(self.norm2(x))
            else:
                x = self.norm1(
                    x
                    + self._sa_block(x, src_mask,
                                     src_key_padding_mask, is_causal=is_causal)
                )
                x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask,
        key_padding_mask,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.transform_norm(self.final_spectre_layer(self.dropout(self.activation(
            self.linear1(x))))))
        return self.dropout2(x)


class FNetEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        norm=None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
        use_spectre: bool = False,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
        self.use_spectre = use_spectre

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ""
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first:
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.self_attn.batch_first was not True"
                + "(use batch_first for better inference performance)"
            )
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
            )
        elif encoder_layer.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn was passed bias=False"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.activation_relu_or_gelu was not True"
            )
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps):
            why_not_sparsity_fast_path = (
                f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
            )
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(
                f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}"
            )
            self.use_nested_tensor = False

    def forward(
        self,
        src: torch.Tensor,
        mask=None,
        src_key_padding_mask=None,
        is_causal=None,
    ) -> torch.Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer = "self.layers[0]"
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = (
                "self.use_nested_tensor (set in init) was not True"
            )
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (
            (not hasattr(self, "mask_check")) or self.mask_check
        ) and not torch._nested_tensor_from_mask_left_aligned(
            src, src_key_padding_mask.logical_not()
        ):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = (
                "src_key_padding_mask and mask were both supplied"
            )
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = (
                    f"src device is neither one of {_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(
                    output, src_key_padding_mask.logical_not(), mask_check=False
                )
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, True)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
            )

        if convert_to_nested:
            output = output.to_padded_tensor(0.0, src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output


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
        use_spectre=False,
        spectre_threshold=None
    ):
        super().__init__()

        num_patches = (img_size // patch_size) ** 2

        self.embeddings_block = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        )

        encoder_layer = FNetEncoderLayer(
            seq_length=num_patches + 1,
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
            use_spectre=use_spectre,
            spectre_threshold=spectre_threshold)

        # self.encoder_blocks = nn.TransformerEncoder(
        #     encoder_layer, num_layers=num_encoders
        # )
        self.encoder_blocks = FNetEncoder(
            encoder_layer, num_layers=num_encoders, use_spectre=use_spectre
        )

        if use_spectre and spectre_threshold is not None:
            embed_dim = int(embed_dim * spectre_threshold)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = x[:, 0, :]
        x = self.mlp_head(x)  # CLS token
        return x
