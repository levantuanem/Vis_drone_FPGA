#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402

STEPS = [
    ("01_frontend", "T1.5", "QONNX -> single dataflow partition (Figs. 4-7)",
     ["FINN 0.10"]),
    ("02_folding", "T1.6", "specialisation (PAPER Table 6) + DERIVED folding",
     ["FINN 0.10"]),
    ("03_fifos", "T1.7", "FIFO depth characterisation (DERIVED, A24)",
     ["FINN 0.10", "Verilator", "pyverilator"]),
    ("04_synth", "T1.9", "HLS synth -> stitch -> ZynqBuild (Table 7, Figs. 12/13/16/17)",
     ["FINN 0.10", "Vivado 2022.2", "Vitis HLS 2022.2", "MYIR board files"]),
    ("05_rtlsim", "T1.10", "RTLSim throughput (Table 8 RTLSim column)",
     ["FINN 0.10", "Verilator"]),
]


def config_hash(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def plan():
    print("=" * 78)
    print("FINN BUILD PLAN  (part {}, Vivado {}, FINN {})".format(
        A.FPGA_PART, A.VIVADO_VERSION, A.FINN_VERSION))
    print("=" * 78)
    for name, exp, desc, deps in STEPS:
        print("\nstep {:<12} [{}]".format(name, exp))
        print("  {}".format(desc))
        print("  requires: {}".format(", ".join(deps)))
    print("\nconfig hash (folding + specialize + quant): {}".format(
        config_hash(["configs/folding.json", "configs/specialize.json",
                     "configs/quant.yaml"])))
    print("\nBLOCKERS (nothing installed by design):")
    print("  - FINN 0.10 + Docker image            -> steps 01-05")
    print("  - Vivado + Vitis HLS 2022.2           -> step 04")
    print("  - MYIR mys-7z020 board files          -> step 04")
    print("  - Verilator / pyverilator (in image)  -> steps 03, 05")
    print("  - NUM_DEFAULT_WORKERS=1 (15 GB RAM)   -> steps 03-05")
    print("\nDECISION D10: FINN 0.10 + Vivado 2022.2 (FINN 0.10's documented pairing;")
    print("the reference bitstream used {}). Phase 0 spike checks it on this host."
          .format(A.REFERENCE_VIVADO_VERSION))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true",
                    help="print the chain and blockers; imports no heavy deps")
    ap.add_argument("--onnx", help="QONNX input for a full chain run")
    ap.add_argument("--out-dir", default="finn_build_out")
    ap.add_argument("--from-step", default="01_frontend")
    ap.add_argument("--to-step", default="05_rtlsim")
    args = ap.parse_args(argv)

    if args.plan or not args.onnx:
        plan()
        if not args.plan:
            print("\n(no --onnx given: printed the plan instead of running)")
        return 0

    names = [s[0] for s in STEPS]
    lo, hi = names.index(args.from_step), names.index(args.to_step)
    from finn_build import (step_01_frontend, step_02_folding, step_03_fifos,
                            step_04_synth, step_05_rtlsim)
    current = args.onnx
    for name in names[lo:hi + 1]:
        print("\n>>> {}".format(name))
        if name == "01_frontend":
            current, _ = step_01_frontend.run(current, args.out_dir)
        elif name == "02_folding":
            current, _ = step_02_folding.run(current, args.out_dir,
                                             "configs/folding.json",
                                             "configs/specialize.json")
        elif name == "03_fifos":
            current, _ = step_03_fifos.run(current, args.out_dir)
        elif name == "04_synth":
            current, _ = step_04_synth.run(current, args.out_dir)
        elif name == "05_rtlsim":
            step_05_rtlsim.run(current, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
