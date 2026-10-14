"""Cycle model tests, including verification against the paper's own Table 8."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A
from tools import cycle_model as CM


def test_divisors():
    assert CM.divisors(27) == [1, 3, 9, 27]
    assert CM.divisors(18) == [1, 2, 3, 6, 9, 18]
    assert CM.divisors(1) == [1]


def test_table8_latency_identity():
    """4,586,895 cycles / 100 MHz == 45.87 ms, exactly as Table 8 prints."""
    e = A.TABLE8_ESTIMATED
    assert round(CM.latency_ms(e["total_cycles"], 100.0), 2) == \
        e["latency_ms_at_100mhz"]


def test_table8_throughput_identity():
    """100 MHz / 216,320 == 462.28 FPS, exactly as Table 8 prints."""
    e = A.TABLE8_ESTIMATED
    assert round(CM.throughput_fps(e["max_node_cycles"], 100.0), 2) == \
        e["throughput_fps_at_100mhz"]


def test_bottleneck_decomposition():
    """416^2 + 208^2 == 216,320 == Table 8's max node (StreamingMaxPool_hls_0)."""
    pool2 = A.LAYERS[1]
    assert CM.maxpool_cycles(pool2) == 416 ** 2 + 208 ** 2 == 216_320
    assert CM.maxpool_cycles(pool2) == A.TABLE8_ESTIMATED["max_node_cycles"]


def test_first_pool_is_the_global_fixed_bottleneck():
    cycles = [CM.maxpool_cycles(p) for p in A.POOL_LAYERS]
    assert max(cycles) == 216_320
    assert cycles.index(max(cycles)) == 0


def test_conv1_full_unfold_fits_under_target():
    """SIMD = k^2*Cin = 27 (RTL parallel window), PE = Cout = 8."""
    c1 = A.CONV_LAYERS[0]
    assert c1.mw == 27 and c1.mh == 8
    assert CM.mvu_cycles(c1, 8, 27) == 416 ** 2 == 173_056
    assert CM.mvu_cycles(c1, 8, 27) < A.TABLE8_ESTIMATED["max_node_cycles"]


def test_conv1_without_parallel_window_would_miss_target():
    """At SIMD=3 conv1 costs 9x more and blows the balance target."""
    c1 = A.CONV_LAYERS[0]
    assert CM.mvu_cycles(c1, 8, 3) == 9 * 173_056
    assert CM.mvu_cycles(c1, 8, 3) > A.TABLE8_ESTIMATED["max_node_cycles"]


def test_mvu_cycles_rejects_illegal_folding():
    c = A.CONV_LAYERS[0]
    try:
        CM.mvu_cycles(c, 3, 27)        # 3 does not divide Cout=8
    except ValueError:
        pass
    else:
        raise AssertionError("illegal PE should raise")


def test_fully_folded_cycles_equal_mac_count():
    """At PE=SIMD=1 an MVU takes one cycle per MAC."""
    for l in A.CONV_LAYERS:
        assert CM.mvu_cycles(l, 1, 1) == l.macs


def test_verify_against_table8_all_pass():
    for name, got, want in CM.verify_against_table8():
        assert got == want, name


def test_bram_model_is_aspect_ratio_aware():
    """A 4-bit wide, very deep memory must not be charged a 36-bit geometry."""
    deep_narrow = CM.bram18k_for(width=4, depth=194_688)
    assert deep_narrow <= 50, deep_narrow          # ~43 tiles, not ~381
    bits = 4 * 194_688
    assert deep_narrow >= bits // CM.BRAM18K_BITS


def test_small_memories_go_to_lutram():
    bram, lutram = CM.mvu_weight_memory(A.CONV_LAYERS[0], 8, 27)
    assert bram == 0 and lutram >= 0        # 216 weights x 8b is tiny


def test_evaluate_fully_folded_is_slow_and_cheap():
    res = CM.evaluate(CM.fully_folded())
    assert res["max_node_cycles"] >= A.CONV_LAYERS[6].macs
    assert res["dsp"] <= A.DEVICE_DSP48E1
    assert res["calibration_required"] is True
