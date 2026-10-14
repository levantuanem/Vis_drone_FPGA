"""GATE 1 as a unit test: the Table 6 arithmetic reconstruction.

Every number the task specification demands is asserted explicitly.
Stdlib only -- runs with no dataset, GPU, FINN, Vivado or board.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A
from tools import table6_selfcheck


def test_ten_of_ten_weight_counts():
    hits = 0
    for mvu, ref in A.TABLE6_REFERENCE.items():
        assert A.layer_by_mvu(mvu).weights == ref["weights"], mvu
        hits += 1
    assert hits == 10


def test_ten_of_ten_mac_counts():
    hits = 0
    for mvu, ref in A.TABLE6_REFERENCE.items():
        assert A.layer_by_mvu(mvu).macs == ref["macs"], mvu
        hits += 1
    assert hits == 10


def test_mac_4b_4b_total():
    assert A.totals()["mac_4b_4b"] == 113_330_048


def test_mac_8b_8b_total():
    assert A.totals()["mac_8b_8b"] == 37_380_096


def test_mac_4b_8b_total():
    assert A.totals()["mac_4b_8b"] == 2_847_312


def test_total_weights():
    assert A.totals()["weights_total"] == 350_696


def test_weights_8b():
    assert A.totals()["weights_8b"] == 17_064


def test_weights_4b():
    assert A.totals()["weights_4b"] == 333_632


def test_layer15_mac_count_is_8858304():
    """Explicitly required by the implementation brief."""
    assert A.layer_by_mvu("MVAU_rtl_3").macs == 8_858_304
    assert A.LAYERS[14].macs == 8_858_304


def test_grand_total_mac():
    t = A.totals()
    assert (t["mac_4b_4b"] + t["mac_8b_8b"] + t["mac_4b_8b"]) == 153_557_456


def test_weight_formula_is_k2_cin_cout():
    for l in A.CONV_LAYERS:
        assert l.weights == l.k * l.k * l.cin * l.cout


def test_mac_formula_is_outpx_times_weights():
    for l in A.CONV_LAYERS:
        assert l.macs == l.out_hw * l.out_hw * l.weights


def test_selfcheck_script_reports_zero_failures():
    """The GATE 1 CLI must pass end to end."""
    c = table6_selfcheck.run()
    assert c.failures == 0, [r for r in c.rows if not r["ok"]]
    assert len(c.rows) >= 55
