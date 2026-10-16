from __future__ import absolute_import

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402  (stdlib-only, always importable)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "models.quant_yolov3_tiny requires torch==1.13.1 (PAPER FACT Table 2); "
        "not installed by design. models.arch_spec works without it."
    ) from exc

from models.quant_common import QuantConvBlock, QuantInputStem  # noqa: E402


class SamePadMaxPool(nn.Module):
    """2x2 / stride-1 max pool that preserves the spatial size.

    DECISION D2 (ARITHMETICALLY FORCED). Paper Table 1 row 12 prints stride 2
    with 13x13 in AND out, which is impossible. Stride 1 with right/bottom
    padding is forced by the printed shapes, by stock YOLOv3-Tiny's sixth pool,
    and decisively by Table 6's layer-13 MAC count
    (13^2 * 9 * 104 * 208 = 32,902,272 exactly; 6x6 would give 7,008,768).
    """

    def __init__(self, kernel_size=2):
        super(SamePadMaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1)

    def forward(self, x):
        pad = self.kernel_size - 1
        x = F.pad(x, (0, pad, 0, pad), mode="replicate")
        return self.pool(x)


class QuantYOLOv3TinyBackbone(nn.Module):
    """Everything that runs on the FPGA: Table 1 layers 1..16.

    The detection head is NOT part of this module -- PAPER FACT 3.2 detaches it
    at export because it is coordinate scaling, better done in PS floating point.

    Output: (N, 18, 13, 13) in NCHW. FINN works in NHWC; the two residual
    Transpose nodes that paper 3.4.3 describes are handled by the driver.
    """

    def __init__(self, quant_cfg=None, batchnorm=None):
        super(QuantYOLOv3TinyBackbone, self).__init__()
        cfg = quant_cfg or {}
        bn = batchnorm if batchnorm is not None else bool(
            cfg.get("structure", {}).get("batchnorm", False))   # A15

        divisor = cfg.get("input_normalisation", {}).get(
            "divisor", A.INPUT_NORMALISATION_DIVISOR)           # A3
        in_bits = cfg.get("bit_widths", {}).get("input_act_bits", 8)  # A14

        self.stem = QuantInputStem(bit_width=in_bits, divisor=divisor)

        blocks = []
        self.layer_names = []
        for layer in A.LAYERS:
            if layer.is_conv:
                pad = (layer.k - 1) // 2 if layer.pad == "same" else 0
                act = "hardtanh" if layer.act == "QuantHardTanh" else "relu"
                blocks.append(QuantConvBlock(
                    cin=layer.cin, cout=layer.cout, kernel_size=layer.k,
                    stride=layer.stride, padding=pad,
                    weight_bits=layer.w_bits, act_bits=layer.a_out_bits,
                    act=act, batchnorm=bn,
                ))
                self.layer_names.append("conv{}_L{}".format(layer.conv_index,
                                                            layer.idx))
            else:
                if layer.stride == 1:
                    blocks.append(SamePadMaxPool(kernel_size=layer.k))   # D2
                else:
                    blocks.append(nn.MaxPool2d(kernel_size=layer.k,
                                               stride=layer.stride))
                self.layer_names.append("pool_L{}".format(layer.idx))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    # -- introspection used by tests and by the equivalence harness ----------

    def conv_modules(self):
        return [m for m in self.blocks if isinstance(m, QuantConvBlock)]

    def weight_counts(self):
        """Per-conv parameter counts, for cross-checking against Table 6."""
        out = []
        for layer, blk in zip(A.CONV_LAYERS, self.conv_modules()):
            out.append((layer.mvu, blk.conv.weight.numel()))
        return out


def build(quant_cfg_path="configs/quant.yaml", batchnorm=None):
    """Build the backbone, reading the quantiser assumptions from YAML."""
    cfg = {}
    if quant_cfg_path and os.path.isfile(quant_cfg_path):
        try:
            import yaml
            with open(quant_cfg_path) as fh:
                cfg = yaml.safe_load(fh) or {}
        except ImportError:  # pragma: no cover
            cfg = {}
    return QuantYOLOv3TinyBackbone(quant_cfg=cfg, batchnorm=batchnorm)
