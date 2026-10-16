#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402

WARMUP_FRAMES = 2       # A28
MEASURE_FRAMES = 8      # A28


def run(stitched_onnx, out_dir, clk_mhz=100.0, warmup=WARMUP_FRAMES,
        frames=MEASURE_FRAMES):
    import numpy as np
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    from finn.core.rtlsim_exec import rtlsim_exec
    from finn.util.basic import gen_finn_dt_tensor

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    model = ModelWrapper(stitched_onnx)
    model.set_metadata_prop("exec_mode", "rtlsim")
    model.set_metadata_prop("rtlsim_trace", "")      # A27 -- tracing OFF

    iname = model.graph.input[0].name
    ishape = A.IO_CONTRACT["ishape_normal"]
    idt = DataType["UINT8"]

    per_frame_cycles = []
    first_frame_cycles = None
    for i in range(warmup + frames):
        x = gen_finn_dt_tensor(idt, ishape)
        ctx = {iname: x}
        rtlsim_exec(model, ctx)
        cyc = int(model.get_metadata_prop("cycles_rtlsim") or 0)
        if i == 0:
            first_frame_cycles = cyc
        if i >= warmup:
            per_frame_cycles.append(cyc)

    steady = min(per_frame_cycles) if per_frame_cycles else None
    doc = {
        "_provenance": {
            "method": "PAPER FACT 4.2.2/4.2.5 -- Verilator RTL simulation",
            "tracing": "disabled (A27)",
            "warmup_frames": warmup, "measure_frames": frames, "_id": "A28",
            "reporting": "throughput cycles and first-frame latency reported "
                         "SEPARATELY (spec P9)",
        },
        "clock_mhz": clk_mhz,
        "first_frame_cycles": first_frame_cycles,
        "steady_state_cycles_per_frame": steady,
        "all_measured_cycles": per_frame_cycles,
        "throughput_fps": (clk_mhz * 1e6 / steady) if steady else None,
        "first_frame_latency_ms": ((first_frame_cycles / (clk_mhz * 1e6)) * 1e3
                                  if first_frame_cycles else None),
        "paper_table8_rtlsim_for_context": {
            "max_node_cycles": 741_670, "latency_ms": 7.41, "fps": 134.83,
            "caveat": "spec P9 -- latency equals max-node count in the paper",
        },
    }
    with open(os.path.join(out_dir, "step05_rtlsim.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="stitched IP model from step 04")
    ap.add_argument("--out-dir", default="finn_build_out")
    ap.add_argument("--clock", type=float, default=100.0, help="MHz (Table 8 uses 100)")
    ap.add_argument("--warmup", type=int, default=WARMUP_FRAMES)
    ap.add_argument("--frames", type=int, default=MEASURE_FRAMES)
    args = ap.parse_args(argv)
    print(json.dumps(run(args.onnx, args.out_dir, args.clock,
                         args.warmup, args.frames), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
