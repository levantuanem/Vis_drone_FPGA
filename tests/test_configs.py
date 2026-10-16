"""Configuration-file consistency tests -- the configs must not drift from the spec."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _yaml(rel):
    import yaml
    with open(os.path.join(ROOT, rel)) as fh:
        return yaml.safe_load(fh)


def test_data_yaml_single_class():
    d = _yaml("data/visdrone_car.yaml")
    assert d["nc"] == 1
    assert d["names"] == ["car"]
    assert d["val"].endswith("val")          # DECISION D3 reporting split exists
    assert not d["path"].startswith("/home")  # must live on /mnt/data10tb


def test_quant_yaml_bit_widths_match_arch_spec():
    q = _yaml("configs/quant.yaml")
    bw = q["bit_widths"]
    assert bw["first_conv_weight_bits"] == A.CONV_LAYERS[0].w_bits == 8
    assert bw["last_conv_weight_bits"] == A.CONV_LAYERS[-1].w_bits == 8
    assert bw["hidden_weight_bits"] == 4
    assert bw["hidden_act_bits"] == 4
    assert bw["output_act_bits"] == A.OUTPUT_ACT_BITS == 8     # spec P11
    assert q["input_normalisation"]["divisor"] == \
        A.INPUT_NORMALISATION_DIVISOR == 255                   # A3


def test_quant_yaml_records_the_a15_batchnorm_assumption():
    q = _yaml("configs/quant.yaml")
    assert q["structure"]["batchnorm"] is True                 # A15 (2026-10-16)
    assert q["structure"]["qat_from_scratch"] is True          # A17


def test_hyp_yaml_has_the_fields_the_loss_consumes():
    h = _yaml("configs/hyp.scratch-low.yaml")
    for k in ("lr0", "lrf", "momentum", "weight_decay", "box", "obj", "cls",
              "anchor_t", "fliplr"):
        assert k in h, k


def test_resource_model_matches_device_budget():
    r = _yaml("configs/resource_model.yaml")
    assert r["device_budget"]["lut"] == A.DEVICE_LUT
    assert r["device_budget"]["dsp48e1"] == A.DEVICE_DSP48E1
    assert r["device_budget"]["bram18k"] == A.DEVICE_BRAM18K
    assert r["dsp_packing"]["act_4b_weight_4b"] == A.DSP48E1_PACK_4B == 4
    assert r["dsp_packing"]["act_or_weight_gt_4b"] == A.DSP48E1_PACK_8B == 2
    assert r["reference_targets"]["bram18k"] == A.TABLE7_ACTUAL["bram18k"] == 138
    assert r["reference_targets"]["lut"] == A.TABLE7_ACTUAL["lut"] == 41605
    assert r["reference_targets"]["dsp"] == A.TABLE7_ACTUAL["dsp"] == 204


def test_model_yaml_rows_match_arch_spec():
    """models/yolov3-tiny-pruned-4w4a.yaml documents what arch_spec defines."""
    m = _yaml("models/yolov3-tiny-pruned-4w4a.yaml")
    assert m["nc"] == 1
    assert m["img_size"] == A.INPUT_RESOLUTION
    assert m["detect_stride"] == A.DETECT_STRIDE
    assert [tuple(a) for a in m["anchors"]] == list(A.ANCHORS_PX)
    rows = m["layers"]
    assert len(rows) == len(A.LAYERS) == 16
    for row, layer in zip(rows, A.LAYERS):
        idx, kind, cin, cout, k, s, pad, act, wbits, impl, mvu = row
        assert idx == layer.idx
        assert kind == layer.kind
        assert (cin, cout, k, s) == (layer.cin, layer.cout, layer.k, layer.stride)
        assert pad == layer.pad
        assert (act or None) == layer.act
        assert (wbits if wbits != "null" else None) == layer.w_bits
        assert impl == layer.impl
        assert (mvu if mvu != "null" else None) == layer.mvu


def test_specialize_json_is_paper_derived():
    with open(os.path.join(ROOT, "configs/specialize.json")) as fh:
        d = json.load(fh)
    assert d["_provenance"]["classification"] == "PAPER FACT"
    assert d["_provenance"]["result"] == "5 RTL + 5 HLS"


def test_folding_json_is_marked_derived_and_provisional():
    with open(os.path.join(ROOT, "configs/folding.json")) as fh:
        d = json.load(fh)
    p = d["_provenance"]
    assert "re-derived" in p["decision"]
    assert "PROVISIONAL" in p["status"]
    assert p["resource_model"].startswith("OUR ASSUMPTION")
