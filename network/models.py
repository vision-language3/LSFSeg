"""LSFSeg model definitions.

This module defines the label encoder/decoder, image encoder, MeanFlow
denoiser, and checkpoint helpers. Internal module attribute names are kept
stable so existing state_dict checkpoints remain loadable.
"""

import json
import math
import os
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.fourier import FFCResnetBlock

LABEL_ENCODER_CONTRACT_FILENAME = "label_encoder_contract.json"
LABEL_ENCODER_CONTRACT_VERSION = 1
LABEL_ENCODER_DOWNSAMPLE_FACTOR = 8
LABEL_LATENT_CHANNELS = 4
IMAGE_LATENT_CHANNELS = 4


# -----------------------
# Utilities / initializers
# -----------------------


def variance_scaling_init(tensor: torch.Tensor, scale: float = 1.0):
    """VarianceScaling with mode='fan_avg', distribution='uniform'"""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    fan_avg = max(1.0, (fan_in + fan_out) / 2.0)
    bound = math.sqrt(3.0 * max(scale, 1e-10) / fan_avg)
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    return tensor


def he_uniform_init(tensor: torch.Tensor, scale: float = 1.0):
    """HeUniform initializer (equivalent to Keras HeUniform)"""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    fan_avg = max(1.0, (fan_in + fan_out) / 2.0)
    bound = math.sqrt(6.0 * max(scale, 1e-10) / fan_avg)
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    return tensor


# -------------------------
# Conv2d with TensorFlow-style SAME padding and selectable initialization.
# -------------------------
class Conv2dSame(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        bias=True,
        scale=1.0,
        init_type="variance",
    ):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )

        # Padding is computed dynamically in forward, so Conv2d itself uses padding=0.
        self.conv = nn.Conv2d(
            in_ch, out_ch, self.kernel_size, stride=self.stride, padding=0, bias=bias
        )

        # Weight initialization.
        if init_type == "variance":
            variance_scaling_init(self.conv.weight, scale)
        elif init_type == "he":
            he_uniform_init(self.conv.weight, scale)
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")

        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        sh, sw = self.stride
        kh, kw = self.kernel_size

        # TensorFlow SAME padding formula.
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + kh - ih, 0)
        pad_w = max((ow - 1) * sw + kw - iw, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


# %% 1x1 conv
def conv2d_1x1(in_ch, out_ch, bias=True, scale=1.0):
    conv = Conv2dSame(in_ch, out_ch, kernel_size=1, bias=bias, init_type="variance")
    return conv


# Backward-compatible alias for older code.
conv2d_same = Conv2dSame

# -----------------------
# Activation helpers
# -----------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation(name):
    if name == "silu":
        return nn.SiLU()
    elif name == "swish":
        return Swish()
    elif name == "relu":
        return nn.ReLU(inplace=True)
    else:
        raise ValueError(f"Unsupported activation: {name}")


def scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False):
    """Let PyTorch select a stable, memory-efficient SDPA backend."""

    if q.device.type == "cpu" and not is_causal:
        # PyTorch's CPU math backend can become non-finite for the 64x64 MGU
        # feature map. Query chunking keeps the score matrix bounded in memory
        # and uses softmax's stable max subtraction.
        scale = q.shape[-1] ** -0.5
        key_transposed = k.transpose(-2, -1)
        output_chunks = []
        for query_chunk in q.split(256, dim=-2):
            scores = torch.matmul(query_chunk, key_transposed) * scale
            probabilities = torch.softmax(scores, dim=-1)
            if float(dropout_p) > 0.0:
                probabilities = F.dropout(
                    probabilities, p=float(dropout_p), training=True
                )
            output_chunks.append(torch.matmul(probabilities, v))
        return torch.cat(output_chunks, dim=-2)

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=float(dropout_p),
        is_causal=bool(is_causal),
    )


# -----------------------
# Up/downsampling layers
# -----------------------
# %% DownSample
class DownSample(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = Conv2dSame(
            width, width, kernel_size=3, stride=2, scale=1.0, init_type="variance"
        )

    def forward(self, x):
        return self.conv(x)


# %% UpSample
class UpSample(nn.Module):
    def __init__(self, width, interpolation="nearest"):
        super().__init__()
        self.interp = interpolation
        self.conv = Conv2dSame(
            width, width, kernel_size=3, stride=1, scale=1.0, init_type="variance"
        )

    def forward(self, x):
        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode=self.interp,
            align_corners=False
            if self.interp in ["linear", "bilinear", "bicubic", "trilinear"]
            else None,
        )
        x = self.conv(x)
        return x


# -------------------------
# Residual convolution block: (B, in_ch, H, W) -> (B, out_ch, H, W)
# -------------------------


class ResConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        filter_size=3,
        dropout=0.0,
        batch_norm=False,
        activation="relu",
        init_type="he",
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout_prob = dropout
        self.init_type = init_type

        # ----- Conv path -----
        self.conv1 = Conv2dSame(
            in_ch, out_ch, kernel_size=filter_size, stride=1, init_type=init_type
        )
        self.bn1 = nn.BatchNorm2d(out_ch) if batch_norm else nn.Identity()

        self.conv2 = Conv2dSame(
            out_ch, out_ch, kernel_size=filter_size, stride=1, init_type=init_type
        )
        self.bn2 = nn.BatchNorm2d(out_ch) if batch_norm else nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ----- Shortcut path -----
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=True)
        if init_type == "he":
            he_uniform_init(self.shortcut.weight)
        else:
            variance_scaling_init(self.shortcut.weight)
        nn.init.zeros_(self.shortcut.bias)

        self.bn_sc = nn.BatchNorm2d(out_ch) if batch_norm else nn.Identity()

    def forward(self, x):
        # Conv path
        out = self.conv1(x)
        out = self.bn1(out)
        if self.activation == "swish":
            out = F.silu(out)
        else:
            out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        # Shortcut path
        sc = self.shortcut(x)
        sc = self.bn_sc(sc)

        # Residual sum + activation
        res = out + sc
        if self.activation == "swish":
            return F.silu(res)
        else:
            return F.relu(res)


# -----------------------
# ConvBlock used by ImageEncoder; each block ends with one downsampling step.
# conv_block(x, filter_size, kernel_size, activation_fn, groups=4, dropout_rate=True)
# Note: original TF used some Functional activation_fn; here we accept nn.Module or function
# -----------------------
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        activation_fn: nn.Module,
        groups=4,
        dropout_rate=0.2,
    ):
        super().__init__()
        # 1x1 residual projection
        self.residual_conv = conv2d_1x1(in_ch, out_ch, scale=1.0)
        self.activation_fn = activation_fn
        self.conv1 = Conv2dSame(in_ch, out_ch, kernel_size=kernel_size, scale=1.0)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate else nn.Identity()
        self.conv2 = Conv2dSame(out_ch, out_ch, kernel_size=kernel_size, scale=0.0)
        self.post_activation = activation_fn
        self.down = DownSample(
            out_ch
        )  # original used DownSample(filter_size) (conv stride=2) but TF code called DownSample(filter_size) at the end; we use maxpool to match encoder pooling behavior
        self.gn = nn.GroupNorm(groups, out_ch)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x = x + residual
        x = self.activation_fn(x)
        x = self.down(x)
        x = self.gn(x)
        return x


# -----------------------
# MultiHeadAttentionBlock (channels-first)
# -----------------------
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, in_channels, units=None, num_heads=8, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.units = units or in_channels
        self.num_heads = num_heads
        self.groups = groups

        # normalization
        self.norm = nn.GroupNorm(groups, self.units)

        # Project channels when the attention width differs from input channels.
        if in_channels != self.units:
            self.channel_proj = conv2d_1x1(in_channels, self.units, scale=1.0)
        else:
            self.channel_proj = nn.Identity()

        # QKV and output projections.
        self.query = nn.Linear(self.units, self.units)
        self.key = nn.Linear(self.units, self.units)
        self.value = nn.Linear(self.units, self.units)
        self.proj = nn.Linear(self.units, self.units)

        # Project back to the original channel count when needed.
        if self.units != in_channels:
            self.out_proj = conv2d_1x1(self.units, in_channels, scale=0.0)
        else:
            self.out_proj = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x
        x = self.channel_proj(x)
        x_norm = self.norm(x)
        tokens = x_norm.view(B, self.units, H * W).permute(0, 2, 1)

        q = self.query(tokens)
        k = self.key(tokens)
        v = self.value(tokens)

        depth = self.units // self.num_heads
        q = q.view(B, -1, self.num_heads, depth).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.num_heads, depth).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.num_heads, depth).permute(0, 2, 1, 3)

        attn_out = scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, H * W, self.units)
        attn_out = self.proj(attn_out)
        attn_out = attn_out.permute(0, 2, 1).view(B, self.units, H, W)

        attn_out = self.out_proj(attn_out)
        return x_in + attn_out


# -----------------------
# Encoder (Label-Encoder)
# -----------------------
class LabelEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        elayers: Sequence[int] = (1, 2, 4, 8),
        dropout_rate=0.2,
        batch_norm=True,
        filter_size=3,
        filter_num=16,
        activation="swish",
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        self.input_channels = int(in_channels)
        self.output_channels = int(out_channels)
        self.downsample_factor = 2 ** max(0, len(tuple(elayers)) - 1)
        cur_in = in_channels
        act = activation
        for i, mul in enumerate(elayers):
            out_ch = mul * filter_num
            self.blocks.append(
                ResConvBlock(
                    cur_in,
                    out_ch,
                    filter_size,
                    dropout_rate,
                    batch_norm,
                    activation=act,
                )
            )
            cur_in = out_ch
        self.final_conv = Conv2dSame(
            cur_in,
            self.output_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            scale=1.0,
            init_type="he",
        )
        self.layer_norm = nn.GroupNorm(1, self.output_channels, affine=True)

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i != len(self.blocks) - 1:
                x = self.pool(x)
        x = self.final_conv(x)
        x = self.layer_norm(x)
        return x


def build_label_encoder_contract(
    label_encoder: LabelEncoder,
    num_classes: int,
    image_size,
    label_condition_mode: str,
):
    """Describe the exact Stage-1 encoder interface reused by Stage 2."""

    condition_mode = str(label_condition_mode).strip().lower()
    target_size = [int(value) for value in image_size]
    if len(target_size) != 2:
        raise ValueError(f"image_size must contain height and width, got {image_size}")

    expected = {
        "input_channels": int(num_classes),
        "latent_channels": LABEL_LATENT_CHANNELS,
        "downsample_factor": LABEL_ENCODER_DOWNSAMPLE_FACTOR,
        "label_condition_mode": "one_hot",
    }
    actual = {
        "input_channels": int(label_encoder.input_channels),
        "latent_channels": int(label_encoder.output_channels),
        "downsample_factor": int(label_encoder.downsample_factor),
        "label_condition_mode": condition_mode,
    }
    mismatches = [
        f"{key}: expected {expected[key]!r}, got {actual[key]!r}"
        for key in expected
        if actual[key] != expected[key]
    ]
    if mismatches:
        raise RuntimeError(
            "MGU Stage-1/Stage-2 label encoder interface is incompatible: "
            + "; ".join(mismatches)
        )

    return {
        "version": LABEL_ENCODER_CONTRACT_VERSION,
        "dataset": "MGU",
        "architecture": label_encoder.__class__.__name__,
        "num_classes": int(num_classes),
        "label_condition_mode": condition_mode,
        "input_channels": int(label_encoder.input_channels),
        "latent_channels": int(label_encoder.output_channels),
        "downsample_factor": int(label_encoder.downsample_factor),
        "image_size": target_size,
        "latent_size": [
            max(1, target_size[0] // int(label_encoder.downsample_factor)),
            max(1, target_size[1] // int(label_encoder.downsample_factor)),
        ],
        "state_shapes": {
            key: list(tensor.shape)
            for key, tensor in label_encoder.state_dict().items()
        },
    }


def save_label_encoder_contract(
    directory: str,
    label_encoder: LabelEncoder,
    num_classes: int,
    image_size,
    label_condition_mode: str,
):
    """Persist the encoder interface beside Stage-1 checkpoints."""

    contract = build_label_encoder_contract(
        label_encoder,
        num_classes=num_classes,
        image_size=image_size,
        label_condition_mode=label_condition_mode,
    )
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, LABEL_ENCODER_CONTRACT_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def validate_label_encoder_contract(
    directory: str,
    label_encoder: LabelEncoder,
    num_classes: int,
    image_size,
    label_condition_mode: str,
):
    """Require a Stage-1 contract that exactly matches the Stage-2 encoder."""

    path = os.path.join(directory, LABEL_ENCODER_CONTRACT_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Missing Stage-1 encoder contract: "
            f"{path}. Run mgu_stage1.py --mode train with the current MGU code first."
        )
    with open(path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    current = build_label_encoder_contract(
        label_encoder,
        num_classes=num_classes,
        image_size=image_size,
        label_condition_mode=label_condition_mode,
    )
    required_keys = (
        "version",
        "dataset",
        "architecture",
        "num_classes",
        "label_condition_mode",
        "input_channels",
        "latent_channels",
        "downsample_factor",
        "image_size",
        "latent_size",
        "state_shapes",
    )
    mismatches = [
        f"{key}: Stage 1 has {saved.get(key)!r}, Stage 2 expects {current[key]!r}"
        for key in required_keys
        if saved.get(key) != current[key]
    ]
    if mismatches:
        raise RuntimeError(
            "Stage-1 encoder checkpoint is not compatible with the Stage-2 encoder: "
            + "; ".join(mismatches)
        )
    return path


def load_label_encoder_state_dict(
    label_encoder: LabelEncoder,
    state_dict,
    checkpoint_name: str = "label encoder checkpoint",
):
    """Load with explicit key/shape diagnostics and strict compatibility."""

    expected_state = label_encoder.state_dict()
    missing = sorted(set(expected_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected_state))
    shape_mismatches = [
        f"{key}: checkpoint {tuple(state_dict[key].shape)} != model {tuple(expected_state[key].shape)}"
        for key in sorted(set(expected_state) & set(state_dict))
        if tuple(state_dict[key].shape) != tuple(expected_state[key].shape)
    ]
    if missing or unexpected or shape_mismatches:
        details = []
        if missing:
            details.append(f"missing keys={missing}")
        if unexpected:
            details.append(f"unexpected keys={unexpected}")
        if shape_mismatches:
            details.append(f"shape mismatches={shape_mismatches}")
        raise RuntimeError(
            f"{checkpoint_name} is incompatible with the current MGU label encoder: "
            + "; ".join(details)
        )
    label_encoder.load_state_dict(state_dict, strict=True)


def load_label_decoder_state_dict(
    label_decoder,
    state_dict,
    checkpoint_name: str = "label decoder checkpoint",
    allowed_missing_prefixes=("latent_fusion_mlp.",),
):
    """Load segmentation-decoder weights while dropping a legacy boundary head.

    Older Stage-2 checkpoints may contain two
    ``boundary_head.*`` tensors. They have no consumer in the current model and
    are removed before all remaining decoder keys and tensor shapes are checked.
    """

    legacy_boundary_keys = sorted(
        key for key in state_dict if key.startswith("boundary_head.")
    )
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("boundary_head.")
    }
    expected_state = label_decoder.state_dict()
    missing = sorted(set(expected_state) - set(filtered_state))
    unexpected = sorted(set(filtered_state) - set(expected_state))
    shape_mismatches = [
        f"{key}: checkpoint {tuple(filtered_state[key].shape)} != "
        f"model {tuple(expected_state[key].shape)}"
        for key in sorted(set(expected_state) & set(filtered_state))
        if tuple(filtered_state[key].shape) != tuple(expected_state[key].shape)
    ]
    disallowed_missing = [
        key for key in missing if not key.startswith(tuple(allowed_missing_prefixes))
    ]
    if disallowed_missing or unexpected or shape_mismatches:
        details = []
        if disallowed_missing:
            details.append(f"missing keys={disallowed_missing}")
        if unexpected:
            details.append(f"unexpected keys={unexpected}")
        if shape_mismatches:
            details.append(f"shape mismatches={shape_mismatches}")
        raise RuntimeError(
            f"{checkpoint_name} is incompatible with the segmentation decoder: "
            + "; ".join(details)
        )
    incompatible = label_decoder.load_state_dict(filtered_state, strict=False)
    if sorted(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{checkpoint_name} produced inconsistent load diagnostics: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if legacy_boundary_keys:
        print(
            f"Ignored {len(legacy_boundary_keys)} legacy boundary-head tensors from "
            f"{checkpoint_name}: {legacy_boundary_keys}",
            flush=True,
        )
    if missing:
        print(
            f"{checkpoint_name} is missing optional decoder keys: {missing}",
            flush=True,
        )


# -----------------------
# Decoder (Label-Decoder)
# -----------------------


class LabelDecoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        dlayers: Sequence[int] = (8, 4, 2, 1),
        filter_num=16,
        out_channels=8,
        image_skip_channels=(16, 32, 64, 128),
        skip_mode="concat",
        dropout_rate=0.2,
        batch_norm=True,
        filter_size=3,
        enable_latent_image_fusion=False,
        latent_image_channels=None,
        latent_fusion_hidden_channels=128,
    ):
        super().__init__()

        self.deconv_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        self.image_skip_channels = tuple(image_skip_channels)
        self.skip_mode = str(skip_mode).strip().lower().replace("-", "_")
        if self.skip_mode not in {"concat", "none"}:
            raise ValueError("skip_mode must be 'concat' or 'none'")
        self.enable_latent_image_fusion = bool(enable_latent_image_fusion)
        if latent_image_channels is None and self.image_skip_channels:
            latent_image_channels = self.image_skip_channels[-1]
        self.latent_image_channels = (
            None if latent_image_channels is None else int(latent_image_channels)
        )
        self.decoder_input_channels = int(in_channels)
        if self.enable_latent_image_fusion:
            if self.latent_image_channels is None or self.latent_image_channels <= 0:
                raise ValueError(
                    "latent_image_channels must be set when latent-image fusion is enabled"
                )
            fusion_hidden_channels = int(latent_fusion_hidden_channels)
            if fusion_hidden_channels <= 0:
                fusion_hidden_channels = max(in_channels, self.latent_image_channels)
            self.decoder_input_channels = fusion_hidden_channels
            self.latent_fusion_mlp = nn.Sequential(
                nn.Conv2d(
                    in_channels + self.latent_image_channels,
                    fusion_hidden_channels,
                    kernel_size=1,
                ),
                nn.SiLU(),
            )
        else:
            self.latent_fusion_mlp = None
        self.num_skips = len(dlayers)
        if self.skip_mode != "none" and len(self.image_skip_channels) != self.num_skips:
            raise ValueError(
                "image_skip_channels must contain one channel count per decoder level"
            )

        cur_in = self.decoder_input_channels
        for i, mul in enumerate(dlayers):
            out_ch = mul * filter_num
            if i == 0:
                self.deconv_blocks.append(
                    Conv2dSame(cur_in, out_ch, kernel_size=3, scale=1.0)
                )
            else:
                self.deconv_blocks.append(
                    nn.Sequential(
                        nn.Upsample(
                            scale_factor=2.0, mode="bilinear", align_corners=False
                        ),
                        Conv2dSame(cur_in, out_ch, kernel_size=3, scale=1.0),
                    )
                )

            skip_ch = (
                self.image_skip_channels[-(i + 1)] if self.image_skip_channels else 0
            )
            block_in = out_ch + skip_ch if self.skip_mode == "concat" else out_ch
            self.res_blocks.append(
                ResConvBlock(
                    block_in,
                    out_ch,
                    filter_size=filter_size,
                    dropout=dropout_rate,
                    batch_norm=batch_norm,
                )
            )
            cur_in = out_ch

        self.out_conv = nn.Conv2d(cur_in, out_channels, 1)
        self.swish = nn.SiLU()

    def _get_skip(self, skip_features, index, size):
        image_skip = skip_features[-(index + 1)]
        if image_skip.shape[2:] != size:
            image_skip = F.interpolate(
                image_skip,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return image_skip

    def _fuse_latent_with_image_feature(self, x, skip_features, image_latent_feature):
        if self.latent_fusion_mlp is None:
            return x
        if image_latent_feature is None:
            if skip_features is None or len(skip_features) == 0:
                raise ValueError(
                    "Latent-image fusion requires the 128-channel image encoder feature"
                )
            image_latent_feature = skip_features[-1]
        if image_latent_feature.shape[1] != self.latent_image_channels:
            raise ValueError(
                "Latent-image fusion expected "
                f"{self.latent_image_channels} image channels, "
                f"got {image_latent_feature.shape[1]}"
            )
        if image_latent_feature.shape[2:] != x.shape[2:]:
            image_latent_feature = F.interpolate(
                image_latent_feature,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([x, image_latent_feature], dim=1)
        return self.latent_fusion_mlp(x)

    def forward(self, x, skip_features=None, image_latent_feature=None):
        if self.skip_mode != "none" and skip_features is None:
            raise ValueError("LabelDecoder requires image encoder skip features")
        if self.skip_mode != "none" and len(skip_features) != self.num_skips:
            raise ValueError(
                f"Expected {self.num_skips} image skips, " f"got {len(skip_features)}"
            )
        x = self._fuse_latent_with_image_feature(x, skip_features, image_latent_feature)
        for i, (deconv, res) in enumerate(zip(self.deconv_blocks, self.res_blocks)):
            x = deconv(x)
            if self.skip_mode == "concat":
                x = torch.cat([x, self._get_skip(skip_features, i, x.shape[2:])], dim=1)
            x = res(x)

        x = self.swish(x)
        return self.out_conv(x)


# -----------------------
# Image Encoder (ImgEncoder)
# -----------------------
class ImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        elayers: Sequence[int] = (1, 2, 4, 8),
        filter_size=16,
        kernel_size=3,
        dropout=0.2,
        groups=4,
        activation=nn.SiLU(),
        use_ffc1=True,
        padding_type="reflect",
        ffc_activation=nn.ReLU,
        ffc_norm=nn.BatchNorm2d,
        **resnet_conv_kwargs,
    ):
        super().__init__()
        self.output_channels = out_channels
        self.conv_blocks = nn.ModuleList()
        self.mha_blocks = nn.ModuleList()
        self.mha_positions = []
        self.feature_channels = tuple(mul * filter_size for mul in elayers)
        self.skip_channels = self.feature_channels
        self.use_ffc1 = use_ffc1
        self.ffc_skip_blocks = nn.ModuleList()

        self.init_conv = Conv2dSame(
            in_channels, filter_size, kernel_size=3, stride=1, bias=True, scale=1.0
        )
        cur_in = in_channels

        for i, mul in enumerate(elayers):
            out_ch = mul * filter_size
            if i == 0:
                self.conv_blocks.append(
                    Conv2dSame(cur_in, out_ch, kernel_size, scale=1.0)
                )
            else:
                self.conv_blocks.append(
                    ConvBlock(
                        cur_in,
                        out_ch,
                        kernel_size,
                        activation,
                        groups=groups,
                        dropout_rate=dropout,
                    )
                )

                # At 512x512, attention one level earlier would operate on
                # 128x128 tokens and materialize an impractically large
                # attention matrix. Keep attention only at the deepest 64x64
                # feature level for stable MGU training.
                if len(elayers) > 3 and i == len(elayers) - 1:
                    self.mha_blocks.append(MultiHeadAttentionBlock(out_ch))
                    self.mha_positions.append(i)
            if self.use_ffc1 and i == 0:
                self.ffc_skip_blocks.append(
                    FFCResnetBlock(
                        dim=out_ch,
                        padding_type=padding_type,
                        activation_layer=ffc_activation,
                        norm_layer=ffc_norm,
                        ratio_gin=0.5,
                        ratio_gout=0.5,
                        gated=True,
                        inline=True,
                        use_se=True,
                        enable_lfu=False,
                        **resnet_conv_kwargs,
                    )
                )
            else:
                self.ffc_skip_blocks.append(nn.Identity())
            cur_in = out_ch

        self.final_conv = Conv2dSame(
            cur_in,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            scale=1.0,
            init_type="he",
        )
        self.final_mha = MultiHeadAttentionBlock(
            out_channels, num_heads=out_channels, groups=1
        )
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, return_skips=False):
        features = []
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            skip = self.ffc_skip_blocks[i](x)
            if i in self.mha_positions:
                mha_idx = self.mha_positions.index(i)
                x = self.mha_blocks[mha_idx](x)
            features.append(skip)

        x = self.final_conv(x)
        x = self.final_mha(x)
        x = self.activation(x)
        x = self.bn(x)

        # Return skip features only when the decoder path needs them.
        if return_skips:
            return x, features
        return x


# -------------------------
# Residual Block
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, activation_fn="swish", temb_dim=None):
        super().__init__()
        self.activation_fn = get_activation(activation_fn)
        self.need_proj = in_ch != out_ch
        if self.need_proj:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        self.gn1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # Time embedding projection
        if temb_dim is not None:
            self.temb_proj = nn.Linear(temb_dim, out_ch)
            nn.init.kaiming_uniform_(self.temb_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.temb_proj.bias)
        else:
            self.temb_proj = None

    def forward(self, x, t=None):
        residual = x
        if self.need_proj:
            residual = self.residual_conv(x)

        out = self.gn1(x)
        out = self.activation_fn(out)
        out = self.conv1(out)

        if self.temb_proj is not None and t is not None:
            temb = self.activation_fn(t)
            temb = self.temb_proj(temb)[:, :, None, None]
            out = out + temb

        out = self.gn2(out)
        out = self.activation_fn(out)
        out = self.conv2(out)

        out = out + residual
        return out


# -------------------------
# Time embedding
# -------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


# -------------------------
# Denoiser
# -------------------------
class Denoiser(nn.Module):
    def __init__(
        self,
        input_shape_lv,
        input_shape_ie,
        first_conv_channels=16,
        widths=(16, 32, 64),
        has_attention=(False, True, True),
        num_res_blocks=2,
        norm_groups=4,
        channels=1,
        with_time_emb=True,
        activation="swish",
        velocity_eps=0.05,
    ):
        super().__init__()
        self.widths = widths
        self.num_res_blocks = num_res_blocks
        self.has_attention = has_attention
        self.norm_groups = norm_groups
        self.activation = activation
        self.activation_fn = get_activation(activation)
        self.label_latent_channels = int(input_shape_lv[0])
        self.image_latent_channels = int(input_shape_ie[0])
        self.output_channels = int(channels)
        self.velocity_eps = float(velocity_eps)
        # -------------------------
        # First conv
        # -------------------------
        in_ch = input_shape_lv[0] + input_shape_ie[0]
        self.first_conv = nn.Conv2d(
            in_ch, first_conv_channels, kernel_size=3, padding=1
        )

        # -------------------------
        # Time embedding
        # -------------------------
        self.time_embed = TimeEmbedding(dim=first_conv_channels)
        # Old Stage-2 checkpoints were trained with three extra scalar
        # embeddings.  They are folded into fixed, non-persistent biases by
        # load_denoiser_state_dict so the archived MGU predictions remain
        # reproducible without keeping those conditions in the model API.
        for profile in ("evaluation", "visualization"):
            for index in range(3):
                self.register_buffer(
                    f"_checkpoint_{profile}_bias_{index}",
                    torch.zeros(first_conv_channels),
                    persistent=False,
                )
        if with_time_emb:
            time_dim = first_conv_channels
            self.time_mlp = nn.Sequential(
                nn.Linear(first_conv_channels, first_conv_channels * 4),
                Swish(),
                nn.Linear(first_conv_channels * 4, first_conv_channels),
            )
        else:
            time_dim = None
            self.time_mlp = None
        # -------------------------
        # CNN residual blocks
        # -------------------------
        self.down_blocks = nn.ModuleList()
        self.attn_down = nn.ModuleList()
        cur_ch = first_conv_channels
        for i, w in enumerate(widths):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResidualBlock(
                        cur_ch,
                        out_ch=w,
                        groups=norm_groups,
                        activation_fn=self.activation,
                        temb_dim=time_dim,
                    )
                )
                cur_ch = w
            self.down_blocks.append(blocks)
            if has_attention[i]:
                self.attn_down.append(MultiHeadAttentionBlock(cur_ch, num_heads=w))
            else:
                self.attn_down.append(nn.Identity())

        # -------------------------
        # Middle
        # -------------------------
        self.mid_res1 = ResidualBlock(
            cur_ch,
            cur_ch,
            groups=norm_groups,
            activation_fn=self.activation,
            temb_dim=time_dim,
        )
        self.mid_attn = MultiHeadAttentionBlock(cur_ch, num_heads=cur_ch)
        self.mid_res2 = ResidualBlock(
            cur_ch,
            cur_ch,
            groups=norm_groups,
            activation_fn=self.activation,
            temb_dim=time_dim,
        )

        # -------------------------
        # Up blocks
        # -------------------------
        self.up_blocks = nn.ModuleList()
        self.attn_up = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.skip_proj = nn.ModuleList()

        for i in reversed(range(len(widths))):
            w = widths[i]
            self.skip_proj.append(nn.Conv2d(w, cur_ch, kernel_size=1))
            in_ch_up = cur_ch + cur_ch
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(
                    ResidualBlock(
                        in_ch_up,
                        out_ch=w,
                        groups=norm_groups,
                        activation_fn=self.activation,
                        temb_dim=time_dim,
                    )
                )
                in_ch_up = w
            self.up_blocks.append(blocks)
            self.upsample_blocks.append(nn.Upsample(scale_factor=2.0, mode="nearest"))
            if has_attention[i]:
                self.attn_up.append(MultiHeadAttentionBlock(w, num_heads=w))
            else:
                self.attn_up.append(nn.Identity())
            cur_ch = w

        # -------------------------
        # Final conv
        # -------------------------
        self.final_gn = nn.GroupNorm(norm_groups, cur_ch)
        self.final_activation = self.activation_fn
        self.out_conv = nn.Conv2d(cur_ch, channels, kernel_size=3, padding=1)
        self.v_out_conv = nn.Conv2d(cur_ch, channels, kernel_size=3, padding=1)

    def _scalar_condition(self, value, batch_size, device, dtype, default):
        if value is None:
            value = torch.full(
                (batch_size,), float(default), device=device, dtype=dtype
            )
        elif not torch.is_tensor(value):
            value = torch.full((batch_size,), float(value), device=device, dtype=dtype)
        else:
            value = value.to(device=device, dtype=dtype).view(batch_size)
        return value

    def _mean_flow_time_embedding(
        self,
        time_input,
        h_input=None,
        checkpoint_profile="evaluation",
    ):
        batch_size = time_input.shape[0]
        device = time_input.device
        dtype = time_input.dtype
        h = self._scalar_condition(h_input, batch_size, device, dtype, default=0.0)
        temb = self.time_mlp(self.time_embed(h))
        if checkpoint_profile not in {"evaluation", "visualization"}:
            raise ValueError(
                "checkpoint_profile must be 'evaluation' or 'visualization', "
                f"got {checkpoint_profile!r}"
            )
        for index in range(3):
            bias = getattr(self, f"_checkpoint_{checkpoint_profile}_bias_{index}")
            temb = temb + bias.to(device=device, dtype=temb.dtype)
        return temb

    def forward(
        self,
        lv_input,
        img_input,
        time_input,
        h_input=None,
        checkpoint_profile="evaluation",
        return_velocity=False,
    ):
        H, W = lv_input.shape[2:]
        cur = self.first_conv(torch.cat([lv_input, img_input], dim=1))

        # -------------------------
        # Time embedding
        # -------------------------
        time_input = time_input.to(device=lv_input.device, dtype=lv_input.dtype).view(
            lv_input.shape[0]
        )
        temb = self._mean_flow_time_embedding(
            time_input,
            h_input=h_input,
            checkpoint_profile=checkpoint_profile,
        )

        # -------------------------
        # Down
        # -------------------------
        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for blk in blocks:
                cur = blk(cur, temb)
            cur = self.attn_down[i](cur)
            skips.append(cur)
            if i != len(self.down_blocks) - 1:
                cur = F.avg_pool2d(cur, kernel_size=2, stride=2)

        # -------------------------
        # Middle
        # -------------------------
        cur = self.mid_res1(cur, temb)
        cur = self.mid_attn(cur)
        cur = self.mid_res2(cur, temb)

        # -------------------------
        # Up
        # -------------------------
        for i, (blocks, upsample, skip_proj, attn) in enumerate(
            zip(self.up_blocks, self.upsample_blocks, self.skip_proj, self.attn_up)
        ):
            skip = skips.pop()
            if skip.shape[2:] != cur.shape[2:]:
                skip = F.interpolate(skip, size=cur.shape[2:], mode="nearest")
            skip = skip_proj(skip)
            cur = torch.cat([cur, skip], dim=1)
            for blk in blocks:
                cur = blk(cur, temb)
            cur = attn(cur)
            if i != len(self.up_blocks) - 1:
                cur = upsample(cur)

        # -------------------------
        # Final conv
        # -------------------------
        cur = self.final_gn(cur)
        cur = self.final_activation(cur)
        u_x = self.out_conv(cur)
        u_x = F.interpolate(u_x, size=(H, W), mode="bilinear", align_corners=False)
        if not return_velocity:
            t_view = time_input.view(-1, *([1] * (lv_input.ndim - 1)))
            return (lv_input - u_x) / t_view.clamp_min(self.velocity_eps)

        v_x = self.v_out_conv(cur)
        v_x = F.interpolate(v_x, size=(H, W), mode="bilinear", align_corners=False)
        t_view = time_input.view(-1, *([1] * (lv_input.ndim - 1)))
        denom = t_view.clamp_min(self.velocity_eps)
        u = (lv_input - u_x) / denom
        v = (lv_input - v_x) / denom
        return u, v


def load_image_encoder_state_dict(image_encoder, state_dict):
    """Load image encoder weights for the original MHA attention structure."""
    image_encoder.load_state_dict(state_dict)


def _time_embedding_from_state(state_dict, prefix, value):
    """Evaluate one removed scalar embedding directly from archived weights."""
    inv_freq = state_dict[f"{prefix}_embed.inv_freq"]
    scalar = inv_freq.new_tensor(float(value)).view(1)
    embedded = torch.cat(
        [
            torch.outer(scalar.float(), inv_freq).sin(),
            torch.outer(scalar.float(), inv_freq).cos(),
        ],
        dim=-1,
    )
    hidden = F.linear(
        embedded,
        state_dict[f"{prefix}_mlp.0.weight"],
        state_dict[f"{prefix}_mlp.0.bias"],
    )
    hidden = hidden * torch.sigmoid(hidden)
    return F.linear(
        hidden,
        state_dict[f"{prefix}_mlp.2.weight"],
        state_dict[f"{prefix}_mlp.2.bias"],
    ).squeeze(0)


def _load_archived_time_biases(denoiser, state_dict, preserve_archived_output):
    """Fold removed checkpoint-only scalar modules into fixed inference biases."""
    archived_prefixes = ("omega", "t_min", "t_max")
    required_keys = {
        f"{prefix}_{suffix}"
        for prefix in archived_prefixes
        for suffix in (
            "embed.inv_freq",
            "mlp.0.weight",
            "mlp.0.bias",
            "mlp.2.weight",
            "mlp.2.bias",
        )
    }
    available_keys = set(state_dict)
    present_keys = required_keys & available_keys
    if present_keys and present_keys != required_keys:
        missing = sorted(required_keys - available_keys)
        raise RuntimeError(
            "Archived denoiser scalar modules are incomplete; " f"missing={missing}"
        )

    for profile in ("evaluation", "visualization"):
        for index in range(3):
            getattr(denoiser, f"_checkpoint_{profile}_bias_{index}").zero_()

    if not present_keys or not preserve_archived_output:
        cleaned_state = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(tuple(f"{prefix}_" for prefix in archived_prefixes))
        }
        if present_keys and not preserve_archived_output:
            print(
                "Archived scalar modules were discarded while warm-starting "
                "the condition-free denoiser."
            )
        if present_keys:
            return cleaned_state, False
        return dict(state_dict), False

    # These are the exact constants used by the archived MGU evaluation and
    # denoising-visualization paths.  Precomputing their contributions removes
    # runtime scalar conditioning while preserving those historical results.
    profile_values = {
        "evaluation": (1.0 - 1.0 / 8.5, 0.1, 0.7),
        "visualization": (0.0, 0.0, 1.0),
    }
    with torch.no_grad():
        for profile, values in profile_values.items():
            for index, (prefix, value) in enumerate(zip(archived_prefixes, values)):
                bias = _time_embedding_from_state(state_dict, prefix, value)
                target = getattr(denoiser, f"_checkpoint_{profile}_bias_{index}")
                target.copy_(bias.to(device=target.device, dtype=target.dtype))

    cleaned_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(tuple(f"{prefix}_" for prefix in archived_prefixes))
    }
    return cleaned_state, True


def load_denoiser_state_dict(denoiser, state_dict, preserve_archived_output=True):
    """Load current or archived denoiser weights."""
    state_dict, used_archived_biases = _load_archived_time_biases(
        denoiser,
        state_dict,
        preserve_archived_output=preserve_archived_output,
    )
    try:
        denoiser.load_state_dict(state_dict)
        if used_archived_biases:
            print(
                "Archived denoiser scalar embeddings were folded into fixed "
                "checkpoint-compatibility biases."
            )
        return
    except RuntimeError:
        incompatible = denoiser.load_state_dict(state_dict, strict=False)

    allowed_missing_prefixes = (
        "mid_attn.",
        "v_out_conv.",
    )
    non_mid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    non_mid_unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith("mid_attn.")
    ]
    if non_mid_missing or non_mid_unexpected:
        raise RuntimeError(
            "Denoiser checkpoint mismatch outside allowed compatibility keys: "
            f"missing={non_mid_missing}, unexpected={non_mid_unexpected}"
        )
    print(
        "Denoiser checkpoint loaded with compatibility fallback. "
        "Missing keys are MeanFlow-only heads or mid_attn drift."
    )


# -----------------------
# Helper to instantiate models and load/save
# -----------------------
def _load_label_model(
    sample_batch,
    device,
    num_classes,
    decoder_skip_mode="concat",
    enable_latent_image_fusion=False,
    latent_fusion_hidden_channels=128,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_tensor = sample_batch["image"].to(device)
    image_channels = images_tensor.shape[1]
    label_condition_tensor = sample_batch["sdf"].to(device)
    label_condition_channels = label_condition_tensor.shape[1]
    if int(image_channels) != 1:
        raise RuntimeError(
            f"MGU image encoder expects one grayscale channel, got {image_channels}"
        )
    if int(label_condition_channels) != int(num_classes):
        raise RuntimeError(
            "MGU Stage-1/Stage-2 label encoder expects an 11-channel one-hot "
            f"condition matching num_classes, got {label_condition_channels} channels "
            f"for num_classes={num_classes}"
        )
    if images_tensor.shape[-2:] != label_condition_tensor.shape[-2:]:
        raise RuntimeError(
            "MGU image and label condition must have matching spatial sizes, got "
            f"{tuple(images_tensor.shape[-2:])} and {tuple(label_condition_tensor.shape[-2:])}"
        )
    # Use four-channel label/image latents for the denoiser input:
    # noisy label latent first, then image latent. MeanFlow predicts velocities
    # in the label latent space, so the u/v heads also have four channels.
    label_latent_channels = LABEL_LATENT_CHANNELS
    image_latent_channels = IMAGE_LATENT_CHANNELS

    image_encoder = ImageEncoder(
        in_channels=image_channels,
        out_channels=image_latent_channels,
    ).to(device)
    label_encoder = LabelEncoder(
        in_channels=label_condition_channels,
        out_channels=label_latent_channels,
    ).to(device)
    label_decoder = LabelDecoder(
        in_channels=label_latent_channels,
        out_channels=int(num_classes),
        image_skip_channels=image_encoder.skip_channels,
        skip_mode=decoder_skip_mode,
        enable_latent_image_fusion=enable_latent_image_fusion,
        latent_image_channels=image_encoder.skip_channels[-1],
        latent_fusion_hidden_channels=latent_fusion_hidden_channels,
    ).to(device)

    latent_height = max(1, label_condition_tensor.shape[2] // 8)
    latent_width = max(1, label_condition_tensor.shape[3] // 8)
    denoiser = Denoiser(
        (label_latent_channels, latent_height, latent_width),
        (image_latent_channels, latent_height, latent_width),
        first_conv_channels=16,
        widths=(16, 32, 64),
        has_attention=(False, True, True),
        channels=label_latent_channels,
    ).to(device)
    return label_encoder, label_decoder, image_encoder, denoiser


def load_model(
    sample_batch,
    device=None,
    num_classes=11,
    decoder_skip_mode="concat",
    enable_latent_image_fusion=False,
    latent_fusion_hidden_channels=128,
):
    """Build models from one batch.

    Expected batch keys include:
    - image: B x 1 x H x W
    - sdf: B x C x H x W label condition tensor.
      The historical key name is kept for trainer compatibility; it is not a
      distance map.
    """
    return _load_label_model(
        sample_batch,
        device,
        num_classes,
        decoder_skip_mode,
        enable_latent_image_fusion,
        latent_fusion_hidden_channels,
    )


def load_label_vae_model(sample_batch, device=None, num_classes=11):
    """Build only the Stage-1 label encoder/decoder used for reconstruction tests."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_condition = sample_batch["sdf"]
    condition_channels = int(label_condition.shape[1])
    if condition_channels != int(num_classes):
        raise RuntimeError(
            "MGU Stage-1 label encoder expects one-hot input with "
            f"{num_classes} channels, got {condition_channels}"
        )
    label_encoder = LabelEncoder(
        in_channels=condition_channels,
        out_channels=LABEL_LATENT_CHANNELS,
    ).to(device)
    label_decoder = LabelDecoder(
        in_channels=LABEL_LATENT_CHANNELS,
        out_channels=int(num_classes),
        skip_mode="none",
        enable_latent_image_fusion=False,
    ).to(device)
    return label_encoder, label_decoder
