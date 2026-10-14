"""Folding derivation tests (DECISION D7)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A
from tools import cycle_model as CM
from tools import folding_search as FS

FOLDING_JSON = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "configs", "folding.json")


def test_initial_state_is_fully_folded_except_pinned():
    folding, styles = FS.initial_state()
    assert folding["MVAU_rtl_0"]["SIMD"] == 27     # REPOSITORY FACT pin
    assert folding["MVAU_rtl_4"]["PE"] == 1        # REPOSITORY FACT pin
    for mvu, f in folding.items():
        if mvu == "MVAU_rtl_0":
            assert f["PE"] == 1
        elif mvu == "MVAU_rtl_4":
            assert f["SIMD"] == 1
        else:
            assert f["PE"] == 1 and f["SIMD"] == 1


def test_pinned_endpoints_are_never_changed_by_the_search():
    folding, styles, res, log, floor, sat = FS.search()
    assert folding["MVAU_rtl_0"]["SIMD"] == 27
    assert folding["MVAU_rtl_4"]["PE"] == 1


def test_search_converges_on_the_papers_bottleneck():
    folding, styles, res, log, floor, sat = FS.search()
    floor_cycles, floor_name = floor
    assert floor_cycles == A.TABLE8_ESTIMATED["max_node_cycles"] == 216_320
    assert "MaxPool" in floor_name
    # every MVU must be at or below the fixed floor
    for n in res["nodes"]:
        if n["kind"] == "MVU":
            assert n["cycles"] <= floor_cycles, n


def test_search_respects_device_budgets():
    folding, styles, res, log, floor, sat = FS.search()
    assert res["dsp"] <= A.DEVICE_DSP48E1
    assert res["lut"] <= A.DEVICE_LUT
    assert res["bram18k"] <= A.DEVICE_BRAM18K


def test_folding_is_legal_everywhere():
    folding, styles, res, log, floor, sat = FS.search()
    for mvu, f in folding.items():
        l = A.layer_by_mvu(mvu)
        assert l.mh % f["PE"] == 0, mvu
        assert l.mw % f["SIMD"] == 0, mvu


def test_hls_lut_escape_hatch_is_used():
    """PAPER FACT 3.6.2 -- switch HLS MVUs to LUT style once DSPs run out."""
    folding, styles, res, log, floor, sat = FS.search()
    lut_style = [m for m, s in styles.items() if s == "HLS_LUT"]
    assert lut_style, "the 3.6.2 LUT escape hatch was never exercised"
    for mvu in lut_style:
        assert A.layer_by_mvu(mvu).impl == "HLS", \
            "only HLS layers may switch to LUT style (RTL MVU has no LUT style)"


def test_rtl_layers_never_switch_style():
    folding, styles, res, log, floor, sat = FS.search()
    for l in A.CONV_LAYERS:
        if l.impl == "RTL":
            assert styles[l.mvu] == "RTL"


def test_committed_folding_json_is_valid_and_marked_provisional():
    assert os.path.isfile(FOLDING_JSON), "run tools/folding_search.py"
    with open(FOLDING_JSON) as fh:
        doc = json.load(fh)
    assert set(doc["folding"]) == {l.mvu for l in A.CONV_LAYERS}
    assert "PROVISIONAL" in doc["_provenance"]["status"]
    assert doc["_provenance"]["decision"].startswith("D7")
    assert doc["fpga_part"] == A.FPGA_PART
    assert doc["estimated"]["calibration_required"] is True


def test_committed_folding_reproduces_its_recorded_estimate():
    with open(FOLDING_JSON) as fh:
        doc = json.load(fh)
    res = CM.evaluate(doc["folding"], doc["styles"])
    assert res["max_node_cycles"] == doc["estimated"]["max_node_cycles"]
    assert res["total_cycles"] == doc["estimated"]["total_cycles"]
    assert res["dsp"] == doc["estimated"]["dsp"]


def test_specialize_json_matches_table6():
    path = os.path.join(os.path.dirname(FOLDING_JSON), "specialize.json")
    with open(path) as fh:
        doc = json.load(fh)
    assert len(doc["layers"]) == 10
    for mvu, meta in doc["layers"].items():
        l = A.layer_by_mvu(mvu)
        assert meta["preferred_impl_style"] == ("hls" if l.impl == "HLS" else "rtl")
        assert meta["weights"] == A.TABLE6_REFERENCE[mvu]["weights"]
        assert meta["macs"] == A.TABLE6_REFERENCE[mvu]["macs"]
