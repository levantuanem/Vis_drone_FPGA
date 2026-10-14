"""Canonical architecture specification — the single source of truth.

This module is **stdlib-only on purpose**. It carries no torch / brevitas / numpy
dependency so that the Table 6 arithmetic gate, the cycle model and the folding
search can all run before any heavy toolchain exists.

Provenance of every field is tagged:

  PAPER FACT      - stated in the paper; section/table cited
  REPOSITORY FACT - recovered from sn0wst0rm/FINN-VisDrone-YOLO; file cited
  DERIVED FACT    - only one value is self-consistent; the constraint is given
  OUR ASSUMPTION  - undeterminable from either source; A-id from the spec

See final_implementation_spec.md §1 and implementation_decisions.md.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

INPUT_RESOLUTION = 416          # PAPER FACT  Table 1 / Table 3
INPUT_CHANNELS = 3              # PAPER FACT  Table 1
NUM_CLASSES = 1                 # PAPER FACT  §3.1.2 ("car objects")
                                # REPOSITORY FACT infer_client_v3.py:24 (nc = 1)
NUM_ANCHORS = 3                 # DERIVED: 18 out-channels / (nc + 5) = 3
OUTPUT_CHANNELS = NUM_ANCHORS * (NUM_CLASSES + 5)   # = 18

# REPOSITORY FACT  client/infer_client_v3.py:25
#   anchors = np.array([[10,14, 23,27, 37,58]]) / np.array([32])
# The paper never states the anchors. These are YOLOv3-Tiny's *small object*
# anchor set, which in stock YOLOv3-Tiny belongs to the stride-16 branch; here
# it is reused at stride 32.
ANCHORS_PX: Tuple[Tuple[int, int], ...] = ((10, 14), (23, 27), (37, 58))
DETECT_STRIDE = 32              # REPOSITORY FACT  infer_client_v3.py:25

# REPOSITORY FACT  finn_streamer/scale.npy == 0.00787402 == 1/127
# This is the trailing `Mul` node that §3.4.4 says stays on the PS.
OUTPUT_DEQUANT_SCALE = 1.0 / 127.0

# DERIVED FACT (spec P11): odt == INT8 (REPOSITORY FACT driver.py) together with
# a dequant scale of 1/127 forces an 8-bit output activation. A 4-bit output
# would require a scale of 1/7. §3.1.1's blanket "all remaining ... INT4" is
# therefore not literal for the final activation, and the paper's own rationale
# ("to produce more detailed outputs at the end") agrees with 8 bits.
OUTPUT_ACT_BITS = 8

# OUR ASSUMPTION A3 — §3.4.1 says only "normalized to values between 0 and 1".
INPUT_NORMALISATION_DIVISOR = 255

# ---------------------------------------------------------------------------
# Device budget  -- PAPER FACT §3
# ---------------------------------------------------------------------------

DEVICE_LUT = 53_200
DEVICE_DSP48E1 = 220
DEVICE_BRAM18K = 140

# DECISION D1 (REPOSITORY-DETERMINED): visdrone.hwh declares SPEEDGRADE="-1"
# while paper §3 prose says XC7Z020CLG400-2. We build for -1.
FPGA_PART = "xc7z020clg400-1"
FPGA_BOARD = "myir.com:mys-7z020:part0:2.1"   # REPOSITORY FACT visdrone.hwh
VIVADO_VERSION = "2022.1"                     # REPOSITORY FACT visdrone.hwh
FINN_VERSION = "0.10"                         # PAPER FACT §3
TARGET_CLOCK_MHZ = 200.0                      # REPOSITORY FACT model_info.json
CLOCK_MODE = "aggressive"                     # REPOSITORY FACT model_info.json

# PAPER FACT Table 4 — DSP packing on DSP48E1.
#   activations [4,4]b & weights [4,4]b -> 4 MAC/DSP
#   activations (4,8]b & weights (4,8]b -> 2 MAC/DSP
#   HLS DSP style                       -> 1 MAC/DSP
DSP48E1_PACK_4B = 4
DSP48E1_PACK_8B = 2
HLS_DSP_PACK = 1


# ---------------------------------------------------------------------------
# Layer model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Layer:
    """One entry of paper Table 1."""

    idx: int                    # 1-based row in Table 1
    kind: str                   # 'conv' | 'maxpool'
    in_hw: int                  # square spatial input size
    out_hw: int                 # square spatial output size
    cin: int
    cout: int
    k: int
    stride: int
    pad: str                    # 'same' | 'valid'
    act: Optional[str]          # 'QuantReLU' | 'QuantHardTanh' | None
    w_bits: Optional[int]       # weight precision (conv only)
    a_in_bits: int              # input activation precision
    a_out_bits: int             # output activation precision
    impl: Optional[str]         # 'RTL' | 'HLS' (FINN backend)
    mvu: Optional[str]          # FINN instance name from paper Table 6
    conv_index: Optional[int]   # 1..10 among convolutions only
    note: str = ""

    # -- derived quantities -------------------------------------------------

    @property
    def is_conv(self) -> bool:
        return self.kind == "conv"

    @property
    def weights(self) -> int:
        """k^2 * Cin * Cout. Folding-independent."""
        if not self.is_conv:
            return 0
        return self.k * self.k * self.cin * self.cout

    @property
    def macs(self) -> int:
        """H_out * W_out * k^2 * Cin * Cout. Folding-independent."""
        if not self.is_conv:
            return 0
        return self.out_hw * self.out_hw * self.weights

    @property
    def mw(self) -> int:
        """MVU inner dimension after Conv lowering: k^2 * Cin (PAPER FACT §3.4.2)."""
        return self.k * self.k * self.cin if self.is_conv else 0

    @property
    def mh(self) -> int:
        """MVU outer dimension: Cout."""
        return self.cout if self.is_conv else 0

    @property
    def dtype_pair(self) -> str:
        """Table 6 'activation b - weight b' notation."""
        if not self.is_conv:
            return ""
        return "{}b-{}b".format(self.a_in_bits, self.w_bits)

    @property
    def dsp_pack_factor(self) -> int:
        """MAC per DSP48E1 for this layer's data types -- PAPER FACT Table 4."""
        if not self.is_conv:
            return 0
        if self.impl == "HLS":
            return HLS_DSP_PACK
        if self.a_in_bits <= 4 and (self.w_bits or 0) <= 4:
            return DSP48E1_PACK_4B
        return DSP48E1_PACK_8B


def _conv(idx, in_hw, out_hw, cin, cout, k, act, w_bits, a_in, a_out,
          impl, mvu, conv_index, stride=1, pad="same", note=""):
    return Layer(idx=idx, kind="conv", in_hw=in_hw, out_hw=out_hw, cin=cin,
                 cout=cout, k=k, stride=stride, pad=pad, act=act,
                 w_bits=w_bits, a_in_bits=a_in, a_out_bits=a_out, impl=impl,
                 mvu=mvu, conv_index=conv_index, note=note)


def _pool(idx, in_hw, out_hw, ch, stride, pad="valid", note=""):
    return Layer(idx=idx, kind="maxpool", in_hw=in_hw, out_hw=out_hw, cin=ch,
                 cout=ch, k=2, stride=stride, pad=pad, act=None, w_bits=None,
                 a_in_bits=4, a_out_bits=4, impl="HLS", mvu=None,
                 conv_index=None, note=note)


# ---------------------------------------------------------------------------
# THE ARCHITECTURE  -- PAPER FACT Table 1, with two documented corrections
# ---------------------------------------------------------------------------
#
# Correction 1 (spec P1 / D-note): §3.1 prose says channel widths were "reduced
#   by a factor of 5"; Table 1 shows factors of 2..4.92. Table 1 is
#   authoritative -- it is corroborated exactly by Table 6's weight counts.
#
# Correction 2 (spec P2 / DECISION D2, ARITHMETICALLY FORCED): Table 1 row 12
#   prints kernel 2x2 / stride 2 with input AND output 13x13x104, which is
#   impossible. Stride 1 with 'same' padding is forced by (a) the printed
#   shapes, (b) stock YOLOv3-Tiny's sixth pool, and (c) Table 6's layer-13 MAC
#   count: 13^2*9*104*208 = 32,902,272 exactly, versus 7,008,768 at 6x6.
#
# Correction 3 (spec P3): row 16 is 3x3, not stock YOLOv3-Tiny's 1x1. Tables 1
#   and 6 agree (16,848 = 3*3*104*18), so this is a deliberate deviation.
#
# RTL/HLS assignment is PAPER FACT, decoded from Table 6's MVU instance names
# and verified against Table 1 by weight count (see paper_vs_repository.md 2.7).

LAYERS: List[Layer] = [
    _conv(1,  416, 416,   3,   8, 3, "QuantReLU",    8, 8, 4, "RTL", "MVAU_rtl_0",  1,
          note="first conv: INT8 weights (PAPER §3.1.1); input act is the UINT8 image"),
    _pool(2,  416, 208,   8, 2),
    _conv(3,  208, 208,   8,   8, 3, "QuantReLU",    4, 4, 4, "RTL", "MVAU_rtl_1",  2),
    _pool(4,  208, 104,   8, 2),
    _conv(5,  104, 104,   8,  16, 3, "QuantReLU",    4, 4, 4, "HLS", "MVAU_hls_0",  3),
    _pool(6,  104,  52,  16, 2),
    _conv(7,   52,  52,  16,  32, 3, "QuantReLU",    4, 4, 4, "RTL", "MVAU_rtl_2",  4),
    _pool(8,   52,  26,  32, 2),
    _conv(9,   26,  26,  32,  56, 3, "QuantReLU",    4, 4, 4, "HLS", "MVAU_hls_1",  5),
    _pool(10,  26,  13,  56, 2),
    _conv(11,  13,  13,  56, 104, 3, "QuantReLU",    4, 4, 4, "HLS", "MVAU_hls_2",  6),
    _pool(12,  13,  13, 104, 1, pad="same",
          note="DECISION D2 / spec P2: stride 1 (Table 1 prints 2 -- impossible)"),
    _conv(13,  13,  13, 104, 208, 3, "QuantReLU",    4, 4, 4, "HLS", "MVAU_hls_3",  7),
    _conv(14,  13,  13, 208,  56, 1, "QuantReLU",    4, 4, 4, "HLS", "MVAU_hls_4",  8,
          pad="valid"),
    _conv(15,  13,  13,  56, 104, 3, "QuantReLU",    4, 4, 4, "RTL", "MVAU_rtl_3",  9),
    _conv(16,  13,  13, 104,  18, 3, "QuantHardTanh", 8, 4, OUTPUT_ACT_BITS,
          "RTL", "MVAU_rtl_4", 10,
          note="last conv: INT8 weights (PAPER §3.1.1); INT8 output (DERIVED, spec P11)"),
]

CONV_LAYERS: List[Layer] = [l for l in LAYERS if l.is_conv]
POOL_LAYERS: List[Layer] = [l for l in LAYERS if not l.is_conv]


# ---------------------------------------------------------------------------
# PAPER FACT  Table 6 -- the reference the architecture must reproduce exactly
# ---------------------------------------------------------------------------

TABLE6_REFERENCE: Dict[str, Dict[str, object]] = {
    "MVAU_rtl_0": {"macs": 37_380_096, "weights":     216, "dtype": "8b-8b"},
    "MVAU_rtl_1": {"macs": 24_920_064, "weights":     576, "dtype": "4b-4b"},
    "MVAU_hls_0": {"macs": 12_460_032, "weights":   1_152, "dtype": "4b-4b"},
    "MVAU_rtl_2": {"macs": 12_460_032, "weights":   4_608, "dtype": "4b-4b"},
    "MVAU_hls_1": {"macs": 10_902_528, "weights":  16_128, "dtype": "4b-4b"},
    "MVAU_hls_2": {"macs":  8_858_304, "weights":  52_416, "dtype": "4b-4b"},
    "MVAU_hls_3": {"macs": 32_902_272, "weights": 194_688, "dtype": "4b-4b"},
    "MVAU_hls_4": {"macs":  1_968_512, "weights":  11_648, "dtype": "4b-4b"},
    "MVAU_rtl_3": {"macs":  8_858_304, "weights":  52_416, "dtype": "4b-4b"},
    "MVAU_rtl_4": {"macs":  2_847_312, "weights":  16_848, "dtype": "4b-8b"},
}

# PAPER FACT  Table 6 totals row
TABLE6_TOTALS = {
    "mac_4b_4b": 113_330_048,
    "mac_8b_8b":  37_380_096,
    "mac_4b_8b":   2_847_312,
    "weights_total": 350_696,
    "weights_8b":     17_064,
    "weights_4b":    333_632,
}

# PAPER FACT  Table 8, "Estimated" column
TABLE8_ESTIMATED = {
    "total_cycles": 4_586_895,
    "max_node_cycles": 216_320,
    "max_node_name": "StreamingMaxPool_hls_0",
    "latency_ms_at_100mhz": 45.87,
    "throughput_fps_at_100mhz": 462.28,
}

# PAPER FACT  Table 7, "Actual" column -- context for T1.9, not a target
TABLE7_ACTUAL = {
    "bram18k": 138, "lut": 41_605, "dsp": 204,
    "bram18k_estimated": 200, "lut_estimated": 26_694, "dsp_estimated": 202,
}

# REPOSITORY FACT  driver.py io_shape_dict / model.pkl
IO_CONTRACT = {
    "idt": "UINT8",
    "odt": "INT8",
    "ishape_normal": (1, 416, 416, 3),
    "oshape_normal": (1, 13, 13, 18),
    "ishape_folded": (1, 416, 416, 1, 3),
    "oshape_folded": (1, 13, 13, 18, 1),
    "input_dma_name": "idma0",
    "output_dma_name": "odma0",
    "number_of_external_weights": 0,   # confirms PAPER §3.8 all-BRAM weights
}


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def layer_by_mvu(name: str) -> Layer:
    for l in CONV_LAYERS:
        if l.mvu == name:
            return l
    raise KeyError("no layer with MVU instance {!r}".format(name))


def totals() -> Dict[str, int]:
    """Recompute Table 6's totals row from the architecture alone."""
    out = {"mac_4b_4b": 0, "mac_8b_8b": 0, "mac_4b_8b": 0,
           "weights_total": 0, "weights_8b": 0, "weights_4b": 0}
    for l in CONV_LAYERS:
        key = "mac_{}b_{}b".format(l.a_in_bits, l.w_bits)
        out[key] = out.get(key, 0) + l.macs
        out["weights_total"] += l.weights
        if l.w_bits == 8:
            out["weights_8b"] += l.weights
        else:
            out["weights_4b"] += l.weights
    return out


def shape_chain_is_consistent() -> List[str]:
    """Verify every layer's output feeds the next layer's input. Returns errors."""
    errors = []
    prev = None
    for l in LAYERS:
        if prev is not None:
            if l.in_hw != prev.out_hw:
                errors.append(
                    "L{}: in_hw={} but L{} produced out_hw={}".format(
                        l.idx, l.in_hw, prev.idx, prev.out_hw))
            if l.cin != prev.cout:
                errors.append(
                    "L{}: cin={} but L{} produced cout={}".format(
                        l.idx, l.cin, prev.idx, prev.cout))
        # spatial arithmetic must match the declared stride/padding
        if l.pad == "same":
            expect = (l.in_hw + l.stride - 1) // l.stride
        else:
            expect = (l.in_hw - l.k) // l.stride + 1
        if expect != l.out_hw:
            errors.append(
                "L{} ({} k={} s={} pad={}): declared out_hw={} but arithmetic "
                "gives {}".format(l.idx, l.kind, l.k, l.stride, l.pad,
                                  l.out_hw, expect))
        prev = l
    if LAYERS[0].in_hw != INPUT_RESOLUTION or LAYERS[0].cin != INPUT_CHANNELS:
        errors.append("first layer input does not match 416x416x3")
    if LAYERS[-1].out_hw != 13 or LAYERS[-1].cout != OUTPUT_CHANNELS:
        errors.append("last layer output does not match 13x13x18")
    return errors
