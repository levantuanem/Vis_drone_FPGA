#!/usr/bin/env python3
"""T1.6 -- layer specialisation + folding application.

Specialisation is PAPER FACT, decoded from Table 6's MVU instance names and
stored in configs/specialize.json (conv 1,2,4,9,10 -> RTL; 3,5,6,7,8 -> HLS).
PAPER FACT 3.5: SpecializeLayers defaults to RTL where available, so HLS choices
MUST be applied explicitly via ApplyConfig.

Folding (PE/SIMD) is NOT taken from anywhere -- it is DERIVED by
tools/folding_search.py (DECISION D7) because paper 3.7.2 delegates these values
to a repository that does not contain them (spec P7).

DEPENDENCY NOT INSTALLED: FINN 0.10. NOT RUN in this phase.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402


def build_finn_config(folding_doc, specialize_doc):
    """Translate our derived folding + paper specialisation into FINN's format."""
    cfg = {"Defaults": {}}
    styles = folding_doc.get("styles", {})
    for mvu, f in folding_doc["folding"].items():
        layer = A.layer_by_mvu(mvu)
        style = styles.get(mvu, layer.impl)
        entry = {
            "PE": int(f["PE"]),
            "SIMD": int(f["SIMD"]),
            # PAPER FACT 3.8 / REPOSITORY FACT number_of_external_weights == 0
            "mem_mode": "internal_decoupled",
        }
        if style == "HLS_LUT":
            # PAPER FACT 3.6.2 -- the LUT escape hatch once DSPs are exhausted
            entry["resType"] = "lut"
        elif style == "HLS":
            entry["resType"] = "dsp"
        cfg[mvu] = entry
    cfg["_provenance"] = {
        "PE_SIMD": "DERIVED (DECISION D7) by tools/folding_search.py",
        "resType": "PAPER FACT 3.6.2 HLS-LUT escape hatch",
        "mem_mode": "PAPER FACT 3.8 all weights on-chip",
        "specialisation": "PAPER FACT Table 6 instance names",
        "status": folding_doc.get("_provenance", {}).get("status", "unknown"),
    }
    return cfg


def run(child_onnx, out_dir, folding_path, specialize_path):
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.general import GiveUniqueNodeNames
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.transformation.fpgadataflow.set_folding import SetFolding  # noqa: F401
    from qonnx.custom_op.registry import getCustomOp  # noqa: F401
    from finn.util.basic import pynq_part_map  # noqa: F401
    from finn.transformation.fpgadataflow.create_dataflow_partition import (  # noqa: F401
        CreateDataflowPartition)
    from qonnx.transformation.general import ApplyConfig

    with open(folding_path) as fh:
        folding_doc = json.load(fh)
    with open(specialize_path) as fh:
        specialize_doc = json.load(fh)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    model = ModelWrapper(child_onnx)

    # ---- step A: force the per-layer backend BEFORE specialisation --------
    pre = {"Defaults": {}}
    for mvu, meta in specialize_doc["layers"].items():
        pre[mvu] = {"preferred_impl_style": meta["preferred_impl_style"]}
    pre_path = os.path.join(out_dir, "step02_preferred_impl.json")
    with open(pre_path, "w") as fh:
        json.dump(pre, fh, indent=2)
    model = model.transform(ApplyConfig(pre_path))
    model = model.transform(SpecializeLayers(A.FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    # ---- step B: apply our DERIVED folding -------------------------------
    finn_cfg = build_finn_config(folding_doc, specialize_doc)
    cfg_path = os.path.join(out_dir, "step02_folding_applied.json")
    with open(cfg_path, "w") as fh:
        json.dump(finn_cfg, fh, indent=2)
    model = model.transform(ApplyConfig(cfg_path))

    out = os.path.join(out_dir, "step02_folded.onnx")
    model.save(out)
    return out, finn_cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", help="child dataflow graph from step 01 "
                                    "(not needed with --print-config)")
    ap.add_argument("--out-dir", default="finn_build_out")
    ap.add_argument("--folding", default="configs/folding.json")
    ap.add_argument("--specialize", default="configs/specialize.json")
    ap.add_argument("--print-config", action="store_true",
                    help="translate and print the FINN config without FINN")
    args = ap.parse_args(argv)

    if not args.print_config and not args.onnx:
        ap.error("--onnx is required unless --print-config is given")

    if args.print_config:
        with open(args.folding) as fh:
            fd = json.load(fh)
        with open(args.specialize) as fh:
            sd = json.load(fh)
        print(json.dumps(build_finn_config(fd, sd), indent=2))
        return 0

    out, cfg = run(args.onnx, args.out_dir, args.folding, args.specialize)
    print(json.dumps(cfg, indent=2))
    print("\nfolded model: {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
