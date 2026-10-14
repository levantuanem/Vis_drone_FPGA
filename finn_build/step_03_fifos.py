#!/usr/bin/env python3
"""T1.7 -- FIFO depth characterisation.

PAPER FACT 3.7: FIFOs absorb per-layer rate mismatch; "excessively short FIFOs
result in a high percentage of layer stalling, while overly long FIFOs
unnecessarily waste resources".
PAPER FACT 3.7.1: depths are determined by RTL simulation with `pyverilator`.

OUR ASSUMPTION A24: the depths themselves are DERIVED here, because paper 3.7.2
delegates them to a repository that does not contain them (spec P7).
OUR ASSUMPTION A25: FINN's automated characterisation first, then hand-tune only
nodes that still stall.
OUR ASSUMPTION A26: the stall-acceptance threshold is recorded in the output.
OUR ASSUMPTION A27: waveform tracing is DISABLED -- VCD over millions of cycles
is infeasible on this host's 15 GB RAM.

ALSO TESTS A PAPER CLAIM (spec P8): paper 3.7.2 reports the HLS StreamingMaxPool
at ~570,000 cycles against a predicted 216,320 (~2.6x), while Table 8 reports the
RTLSim max node at 741,670. Those two figures disagree. This step records what we
measure and flags the discrepancy rather than reconciling it.

DEPENDENCY NOT INSTALLED: FINN 0.10, Verilator, pyverilator. NOT RUN in this phase.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A        # noqa: E402
from tools import cycle_model as CM      # noqa: E402

STALL_THRESHOLD = 0.05        # A26: accept <=5% stall per node
MAXPOOL_PREDICTED = 216_320   # DERIVED, equals PAPER Table 8's max node
PAPER_MAXPOOL_MEASURED = 570_000      # PAPER FACT 3.7.2
PAPER_RTLSIM_MAX_NODE = 741_670       # PAPER FACT Table 8 -- disagrees with above


def run(folded_onnx, out_dir, clk_ns=5.0):
    from qonnx.core.modelwrapper import ModelWrapper
    from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
    from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
    from finn.transformation.fpgadataflow.set_fifo_depths import (
        InsertAndSetFIFODepths)
    from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
    from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
    from qonnx.transformation.general import GiveUniqueNodeNames
    from qonnx.custom_op.registry import getCustomOp

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    model = ModelWrapper(folded_onnx)
    model = model.transform(InsertDWC())
    model = model.transform(GiveUniqueNodeNames())
    # A25: automated characterisation. A27: no waveform tracing.
    model = model.transform(InsertAndSetFIFODepths(A.FPGA_PART, clk_ns))
    model = model.transform(GiveUniqueNodeNames())

    depths = {}
    for node in model.graph.node:
        if "FIFO" in node.op_type:
            op = getCustomOp(node)
            try:
                depths[node.name] = int(op.get_nodeattr("depth"))
            except Exception:
                pass

    out = os.path.join(out_dir, "step03_fifos.onnx")
    model.save(out)

    doc = {
        "_provenance": {
            "method": "PAPER FACT 3.7.1 -- pyverilator RTL characterisation",
            "values": "DERIVED (A24) -- absent from paper and repository (spec P7)",
            "approach": "A25 automated then hand-tune",
            "stall_threshold": STALL_THRESHOLD,
            "waveform_tracing": "disabled (A27, 15 GB RAM host)",
            "status": "PROVISIONAL until T1.7 is actually executed",
        },
        "fpga_part": A.FPGA_PART,
        "target_clk_ns": clk_ns,
        "fifo_depths": depths,
        "maxpool_anomaly_check": {
            "our_prediction": MAXPOOL_PREDICTED,
            "paper_3_7_2_measured": PAPER_MAXPOOL_MEASURED,
            "paper_table8_rtlsim_max_node": PAPER_RTLSIM_MAX_NODE,
            "note": "spec P8: the paper gives two different figures for the same "
                    "node. We report what we measure and flag the ambiguity.",
            "our_measured": None,
        },
    }
    with open(os.path.join(out_dir, "fifo_depths.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
    return out, doc


def predicted_profile():
    """The cycle profile the FIFOs must smooth, from our derived folding."""
    with open("configs/folding.json") as fh:
        d = json.load(fh)
    return CM.evaluate(d["folding"], d.get("styles", {}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", help="folded model from step 02")
    ap.add_argument("--out-dir", default="finn_build_out")
    ap.add_argument("--clk-ns", type=float, default=5.0, help="200 MHz default")
    ap.add_argument("--show-target", action="store_true",
                    help="print the predicted cycle profile without FINN")
    args = ap.parse_args(argv)

    if args.show_target:
        res = predicted_profile()
        print("predicted per-node cycle profile (FIFOs must smooth this):")
        for n in sorted(res["nodes"], key=lambda x: -x["cycles"])[:14]:
            print("  {:<28} {:>12,}".format(n["name"], n["cycles"]))
        print("\nPAPER FACT 3.7.2 anomaly to test: HLS StreamingMaxPool measured")
        print("  ~{:,} cycles vs {:,} predicted (~2.6x);".format(
            PAPER_MAXPOOL_MEASURED, MAXPOOL_PREDICTED))
        print("  but PAPER Table 8 says the RTLSim max node is {:,} -- spec P8.".format(
            PAPER_RTLSIM_MAX_NODE))
        return 0

    if not args.onnx:
        ap.error("--onnx is required unless --show-target is given")
    out, doc = run(args.onnx, args.out_dir, args.clk_ns)
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
