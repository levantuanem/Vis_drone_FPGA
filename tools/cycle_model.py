#!/usr/bin/env python3
"""Analytical cycle / resource model for the FINN dataflow pipeline (T1.8).

Stdlib-only. No FINN, no Vivado, no dataset.

CYCLE MODEL -- DERIVED FACT, verified against the paper's own Table 8
--------------------------------------------------------------------
    MVU cycles = (C_out / PE) * (k^2 * C_in / SIMD) * (H_out * W_out)
    latency    = total_cycles / f_clk
    throughput = f_clk / max_node_cycles

Verification performed before adopting the model (all exact):
    4,586,895 cycles / 100 MHz            = 45.87 ms   == Table 8 latency
    100 MHz / 216,320 cycles              = 462.3 FPS  == Table 8 throughput
    416^2 + 208^2 = 173,056 + 43,264      = 216,320    == Table 8 max node
                                                          (StreamingMaxPool_hls_0)

The maxpool term (one cycle per input pixel plus one per output pixel) is a
DERIVED FACT: it is the only simple form that reproduces 216,320 exactly for the
first pool, and the paper names that very node as the bottleneck.

RESOURCE MODEL -- OUR ASSUMPTION, provisional
---------------------------------------------
DSP counting follows paper Table 4 packing exactly and is trustworthy.
LUT and BRAM estimates are COARSE and must be calibrated against the real
Vivado report from T1.9 (paper Table 7: 138 BRAM / 41,605 LUT / 204 DSP).
Until then every resource number emitted here carries
`"calibration_required": true`. See configs/resource_model.yaml.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402

# --- resource-model parameters (OUR ASSUMPTION; calibrate against T1.9) -----
# LUT cost of one MAC/cycle in an HLS MVU using the LUT implementation style.
# Provisional placeholder: the paper gives no per-MAC figure. Calibration target
# is Table 7's 41,605 actual LUT.
LUT_PER_MAC_HLS_LUT = 14.0
# LUT overhead per instantiated hardware node (control, address generation).
LUT_PER_NODE_OVERHEAD = 180.0
# BRAM18K capacity and port limits (Xilinx 7-series): 18,432 bits, selectable
# aspect ratio from 16384x1 to 512x36. A single port is at most 36 bits wide.
BRAM18K_BITS = 18_432
BRAM18K_MAX_WIDTH = 36
BRAM18K_MAX_DEPTH = 16_384
# Memories at or below this many bits are mapped to distributed RAM (LUTRAM)
# rather than BRAM. This is the effect PAPER FACT §4.2.4 describes when Vivado
# converted part of an estimated 200 BRAM into LUTRAM, landing at 138 actual.
LUTRAM_BIT_THRESHOLD = 1_024
# LUT cost per bit of distributed RAM (OUR ASSUMPTION; 64 bits per LUT in SLICEM).
LUT_PER_LUTRAM_BIT = 1.0 / 64.0


def divisors(n):
    """Ascending divisors of n."""
    out = []
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            out.append(d)
            if d != n // d:
                out.append(n // d)
    return sorted(out)


# ---------------------------------------------------------------------------
# cycle counts
# ---------------------------------------------------------------------------

def mvu_cycles(layer, pe, simd):
    """DERIVED FACT -- see module docstring."""
    assert layer.is_conv
    if layer.mh % pe or layer.mw % simd:
        raise ValueError("PE must divide C_out and SIMD must divide k^2*C_in")
    return (layer.mh // pe) * (layer.mw // simd) * (layer.out_hw ** 2)


def maxpool_cycles(layer):
    """DERIVED FACT -- in-pixels + out-pixels; reproduces 216,320 for L2 exactly.

    Note PAPER FACT §3.7.2: the *actual* HLS implementation was measured at
    ~570,000 cycles for this node (~2.6x the prediction) because Vitis fails to
    unroll loops on large feature maps, and Table 8's RTLSim column reports
    741,670. Those two paper figures disagree (spec P8). This model deliberately
    reproduces the *prediction*, and T1.7 measures reality.
    """
    assert not layer.is_conv
    return layer.in_hw ** 2 + layer.out_hw ** 2


def swg_cycles(layer, simd):
    """Sliding-window generator.

    PAPER FACT §3.5.4: the RTL SWG's parallel-window mode emits one full window
    per cycle but requires full SIMD unfolding (SIMD == k^2*C_in). Modelled as
    one cycle per output pixel in that mode, otherwise folded by SIMD.
    """
    assert layer.is_conv
    px = layer.out_hw ** 2
    if simd == layer.mw:
        return px
    return px * (layer.mw // simd)


def threshold_cycles(layer, pe):
    """MultiThreshold activation: one output element group per cycle."""
    assert layer.is_conv
    return (layer.mh // pe) * (layer.out_hw ** 2)


# ---------------------------------------------------------------------------
# resource estimates  (DSP exact; LUT/BRAM provisional)
# ---------------------------------------------------------------------------

def mvu_dsp(layer, pe, simd, style=None):
    """DSP48E1 count. PAPER FACT Table 4 packing."""
    style = style or layer.impl
    macs_per_cycle = pe * simd
    if style == "HLS_LUT":
        return 0
    pack = layer.dsp_pack_factor if style != "HLS_LUT" else 0
    if style == "HLS":
        pack = A.HLS_DSP_PACK
    return int(math.ceil(macs_per_cycle / float(pack)))


def mvu_lut(layer, pe, simd, style=None):
    """OUR ASSUMPTION -- provisional; calibrate against Table 7."""
    style = style or layer.impl
    if style == "HLS_LUT":
        bitscale = max(layer.a_in_bits, layer.w_bits or 4) / 4.0
        return int(LUT_PER_MAC_HLS_LUT * pe * simd * bitscale
                   + LUT_PER_NODE_OVERHEAD)
    return int(LUT_PER_NODE_OVERHEAD + 0.5 * pe * simd)


def bram18k_for(width, depth):
    """BRAM18K tiles needed for a width x depth memory.

    Aspect-ratio aware: one BRAM18K holds BRAM18K_BITS bits and a single port is
    at most BRAM18K_MAX_WIDTH bits wide, so a narrow deep memory packs far better
    than a fixed 512x36 geometry would suggest.
    """
    if width <= 0 or depth <= 0:
        return 0
    cascade_w = int(math.ceil(width / float(BRAM18K_MAX_WIDTH)))
    eff_width = BRAM18K_MAX_WIDTH if cascade_w > 1 else width
    per_bram_depth = min(BRAM18K_MAX_DEPTH, BRAM18K_BITS // max(1, eff_width))
    cascade_d = int(math.ceil(depth / float(max(1, per_bram_depth))))
    return cascade_w * cascade_d


def mvu_weight_memory(layer, pe, simd):
    """Weight-memory estimate for one MVU. Returns (bram18k, lutram_lut).

    OUR ASSUMPTION -- provisional; calibrate against paper Table 7 via T1.9.

    FINN partitions the lowered weight matrix (MW x MH) into PE memories, each
    SIMD*w_bits wide and (MW/SIMD)*(MH/PE) deep. Unfolding therefore fragments
    the weight store, which is why FINN's own estimate (200 BRAM, Table 7) can
    exceed the device's 140 while the placed design fits at 138.

    PAPER FACT §3.8 / REPOSITORY FACT number_of_external_weights == 0: every
    weight lives on-chip, so this memory is never traded for DRAM bandwidth.
    """
    w_bits = layer.w_bits or 4
    width = simd * w_bits
    depth = (layer.mw // simd) * (layer.mh // pe)
    bits_per_mem = width * depth
    if bits_per_mem <= LUTRAM_BIT_THRESHOLD:
        return 0, int(pe * bits_per_mem * LUT_PER_LUTRAM_BIT)
    return pe * bram18k_for(width, depth), 0


def mvu_weight_bram18k(layer, pe, simd):
    """BRAM18K only (back-compat helper)."""
    return mvu_weight_memory(layer, pe, simd)[0]


# ---------------------------------------------------------------------------
# whole-network evaluation
# ---------------------------------------------------------------------------

def evaluate(folding, styles=None):
    """Evaluate a folding dict {mvu_name: {'PE':p,'SIMD':s}}.

    Returns a dict with per-node cycles/resources plus network totals.
    """
    styles = styles or {}
    nodes = []
    for layer in A.LAYERS:
        if layer.is_conv:
            f = folding[layer.mvu]
            pe, simd = f["PE"], f["SIMD"]
            style = styles.get(layer.mvu, layer.impl)
            bram, lutram = mvu_weight_memory(layer, pe, simd)
            nodes.append({
                "name": layer.mvu, "kind": "MVU", "layer": layer.idx,
                "style": style, "PE": pe, "SIMD": simd,
                "cycles": mvu_cycles(layer, pe, simd),
                "dsp": mvu_dsp(layer, pe, simd, style),
                "lut": mvu_lut(layer, pe, simd, style) + lutram,
                "bram18k": bram,
            })
            nodes.append({
                "name": "SWG_L{}".format(layer.idx), "kind": "SWG",
                "layer": layer.idx, "style": layer.impl, "PE": None, "SIMD": simd,
                "cycles": swg_cycles(layer, simd),
                "dsp": 0, "lut": int(LUT_PER_NODE_OVERHEAD), "bram18k": 0,
            })
            nodes.append({
                "name": "Thresholding_L{}".format(layer.idx), "kind": "MultiThreshold",
                "layer": layer.idx, "style": "HLS", "PE": pe, "SIMD": None,
                "cycles": threshold_cycles(layer, pe),
                "dsp": 0, "lut": int(LUT_PER_NODE_OVERHEAD), "bram18k": 0,
            })
        else:
            nodes.append({
                "name": "StreamingMaxPool_L{}".format(layer.idx), "kind": "MaxPool",
                "layer": layer.idx, "style": "HLS", "PE": None, "SIMD": None,
                "cycles": maxpool_cycles(layer),
                "dsp": 0, "lut": int(LUT_PER_NODE_OVERHEAD), "bram18k": 0,
            })

    total_cycles = sum(n["cycles"] for n in nodes)
    slowest = max(nodes, key=lambda n: n["cycles"])
    return {
        "nodes": nodes,
        "total_cycles": total_cycles,
        "max_node_cycles": slowest["cycles"],
        "max_node_name": slowest["name"],
        "dsp": sum(n["dsp"] for n in nodes),
        "lut": sum(n["lut"] for n in nodes),
        "bram18k": sum(n["bram18k"] for n in nodes),
        "calibration_required": True,
    }


def throughput_fps(max_node_cycles, f_clk_mhz):
    return f_clk_mhz * 1e6 / float(max_node_cycles)


def latency_ms(total_cycles, f_clk_mhz):
    return total_cycles / (f_clk_mhz * 1e6) * 1e3


def fully_folded():
    """PAPER FACT §3.6.1 starting point: every PE and SIMD = 1."""
    return {l.mvu: {"PE": 1, "SIMD": 1} for l in A.CONV_LAYERS}


def verify_against_table8():
    """Confirm the model reproduces Table 8's published arithmetic."""
    ref = A.TABLE8_ESTIMATED
    out = []
    lat = latency_ms(ref["total_cycles"], 100.0)
    out.append(("Table 8 latency from total cycles", round(lat, 2),
                ref["latency_ms_at_100mhz"]))
    fps = throughput_fps(ref["max_node_cycles"], 100.0)
    out.append(("Table 8 throughput from max node", round(fps, 2),
                ref["throughput_fps_at_100mhz"]))
    pool2 = A.LAYERS[1]
    out.append(("max node == first maxpool cycles", maxpool_cycles(pool2),
                ref["max_node_cycles"]))
    conv1 = A.CONV_LAYERS[0]
    out.append(("conv1 at SIMD=k^2Cin, PE=Cout fits under target",
                mvu_cycles(conv1, conv1.mh, conv1.mw) <= ref["max_node_cycles"],
                True))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folding", help="folding JSON; default = fully folded")
    ap.add_argument("--clock", type=float, default=100.0, help="MHz (default 100)")
    ap.add_argument("--verify", action="store_true",
                    help="check the model against paper Table 8 and exit")
    ap.add_argument("--json", help="write the evaluation to this path")
    args = ap.parse_args()

    if args.verify:
        rc = 0
        print("CYCLE MODEL VERIFICATION AGAINST PAPER TABLE 8")
        print("-" * 56)
        for name, got, want in verify_against_table8():
            ok = (got == want)
            rc |= (0 if ok else 1)
            print("[{}] {:<44} {} (want {})".format(
                "PASS" if ok else "FAIL", name, got, want))
        return rc

    if args.folding:
        with open(args.folding) as fh:
            doc = json.load(fh)
        folding = doc.get("folding", doc)
        styles = doc.get("styles", {})
    else:
        folding, styles = fully_folded(), {}

    res = evaluate(folding, styles)
    print("nodes: {}".format(len(res["nodes"])))
    print("total cycles   : {:,}".format(res["total_cycles"]))
    print("max node       : {} @ {:,} cycles".format(res["max_node_name"],
                                                     res["max_node_cycles"]))
    print("latency        : {:.2f} ms @ {:.0f} MHz".format(
        latency_ms(res["total_cycles"], args.clock), args.clock))
    print("throughput     : {:.2f} FPS @ {:.0f} MHz".format(
        throughput_fps(res["max_node_cycles"], args.clock), args.clock))
    print("DSP (exact)    : {} / {}".format(res["dsp"], A.DEVICE_DSP48E1))
    print("LUT (provisional)   : {} / {}".format(res["lut"], A.DEVICE_LUT))
    print("BRAM18K (provisional): {} / {}".format(res["bram18k"], A.DEVICE_BRAM18K))
    print("NOTE: LUT/BRAM are provisional; calibrate against T1.9 (Table 7).")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
