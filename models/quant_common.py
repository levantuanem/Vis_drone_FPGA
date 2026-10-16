from __future__ import absolute_import

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - dependency documented, not installed
    raise ImportError(
        "models.quant_common requires torch==1.13.1 (PAPER FACT Table 2). "
        "Not installed in this environment by design; see requirements/README.md."
    ) from exc

try:
    import brevitas.nn as qnn
    from brevitas.inject.enum import ScalingImplType
    from brevitas.quant import (
        Int8WeightPerTensorFloat,
        Uint8ActPerTensorFloat,
        Int8ActPerTensorFloatMinMaxInit,
    )
    from brevitas.core.restrict_val import RestrictValueType  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "models.quant_common requires brevitas==0.9.0 (PAPER FACT Table 2). "
        "Not installed in this environment by design; see requirements/README.md."
    ) from exc


# ---------------------------------------------------------------------------
# Quantiser families
# ---------------------------------------------------------------------------
# OUR ASSUMPTION A10/A11: per-tensor, learned float scale, narrow range.
# Brevitas' Int8WeightPerTensorFloat is the per-tensor float-scale family; the
# bit width is overridden per layer, which is how 4-bit and 8-bit weights share
# one quantiser definition.

class WeightQuant(Int8WeightPerTensorFloat):
    """Per-tensor weight quantiser; bit_width supplied per layer. A10/A11."""
    narrow_range = True


class ActQuantUnsigned(Uint8ActPerTensorFloat):
    """Unsigned activation quantiser, used after ReLU. A12."""
    pass


class ActQuantSigned(Int8ActPerTensorFloatMinMaxInit):
    """Signed activation quantiser, used after HardTanh. A12.

    CONSTANT scale = max_val / 127 (narrow range), i.e. exactly 1/127 at 8 bits,
    so HardTanh's [-1, 1] maps onto [-127, 127]. This is what makes the
    REPOSITORY FACT dequant scale (scale.npy == 1/127) true by construction; a
    learned scale would make the PS-side Mul constant whatever training chose.
    """
    scaling_impl_type = ScalingImplType.CONST
    narrow_range = True
    min_val = -1.0
    max_val = 1.0


class QuantConvBlock(nn.Module):
    """QuantConv2d (+ optional BN) + quantised activation.

    PAPER FACT Table 1: every convolution is followed by QuantReLU, except the
    final one which uses QuantHardTanh.

    OUR ASSUMPTION A15: Table 1 lists no BN layers, but configs/quant.yaml turns
    BN on because the BN-free network does not train (DECISIONS.md 2026-10-16).
    This is the project's highest-risk assumption -- FINN absorbs BN into
    MultiThreshold during streamlining so the *hardware* is unaffected either
    way, but training dynamics and accuracy are not. The flag exists precisely so
    the ablation can be run without editing code.
    """

    def __init__(self, cin, cout, kernel_size, stride=1, padding=0,
                 weight_bits=4, act_bits=4, act="relu", batchnorm=False,
                 bias=False):
        super(QuantConvBlock, self).__init__()
        self.conv = qnn.QuantConv2d(
            in_channels=cin, out_channels=cout, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias,
            weight_quant=WeightQuant, weight_bit_width=weight_bits,
        )
        self.bn = nn.BatchNorm2d(cout) if batchnorm else None
        if act == "relu":
            # PAPER FACT Table 1: QuantReLU
            self.act = qnn.QuantReLU(act_quant=ActQuantUnsigned,
                                     bit_width=act_bits, return_quant_tensor=False)
        elif act == "hardtanh":
            # PAPER FACT Table 1: QuantHardTanh on the final conv.
            # DERIVED FACT (spec P11): act_bits must be 8 here -- odt=INT8 with a
            # dequant scale of 1/127 maps HardTanh [-1,1] onto [-127,127].
            self.act = qnn.QuantHardTanh(act_quant=ActQuantSigned,
                                         bit_width=act_bits,
                                         return_quant_tensor=False)
        elif act is None:
            self.act = nn.Identity()
        else:
            raise ValueError("unknown activation {!r}".format(act))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)


class QuantInputStem(nn.Module):
    """Input quantiser + on-chip normalisation placeholder.

    PAPER FACT 3.4.1: the accelerator receives UINT8 and a `Div` node is inserted
    before the first convolution so the network sees [0,1]. At training time the
    equivalent division happens here. OUR ASSUMPTION A3: divisor = 255.
    """

    def __init__(self, bit_width=8, divisor=255):
        super(QuantInputStem, self).__init__()
        self.divisor = float(divisor)
        self.quant = qnn.QuantIdentity(act_quant=ActQuantUnsigned,
                                       bit_width=bit_width,
                                       return_quant_tensor=False)

    def forward(self, x):
        # x arrives as uint8-valued float in [0,255]
        return self.quant(x / self.divisor)


def maxpool(kernel_size=2, stride=2, padding=0, ceil_mode=False):
    """Plain MaxPool2d -- FINN maps this to StreamingMaxPool.

    Layer 12 uses stride 1 with asymmetric 'same' padding (DECISION D2); the
    model builder handles that with an explicit pad, not with this helper's args.
    """
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding,
                        ceil_mode=ceil_mode)
