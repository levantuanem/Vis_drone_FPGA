#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402

PLATFORM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "platform", "mys-7z020")


def preflight():
    """Fail early and legibly on missing toolchain / board files."""
    problems = []
    xilinx = os.environ.get("FINN_XILINX_PATH")
    if not xilinx or not os.path.isdir(xilinx):
        problems.append(
            "FINN_XILINX_PATH is unset or missing. Install Vivado/Vitis "
            "{} on /mnt/data10tb (NOT / -- only 74 GB free) and export it."
            .format(A.VIVADO_VERSION))
    ver = os.environ.get("FINN_XILINX_VERSION")
    if ver and ver != A.VIVADO_VERSION:
        problems.append(
            "FINN_XILINX_VERSION={} but DECISION D10 fixes Vivado {} (the "
            "version FINN 0.10 documents; the reference build used {}). "
            "Escalate rather than silently switch versions.".format(
                ver, A.VIVADO_VERSION, A.REFERENCE_VIVADO_VERSION))
    board_files = [f for f in os.listdir(PLATFORM_DIR)
                   if f.endswith((".xml", ".tcl"))] if os.path.isdir(PLATFORM_DIR) else []
    if not board_files:
        problems.append(
            "MYIR board files absent from {}. The reference build used board "
            "{!r}; obtain it from MYIR or author a minimal custom platform. "
            "See that directory's README.".format(PLATFORM_DIR, A.FPGA_BOARD))
    workers = os.environ.get("NUM_DEFAULT_WORKERS")
    if workers and int(workers) > 2:
        problems.append(
            "NUM_DEFAULT_WORKERS={} will likely OOM: this host has 15 GB RAM. "
            "Use 1.".format(workers))
    return problems


def run(fifo_onnx, out_dir, clk_ns=5.0, strategy=None):
    from qonnx.core.modelwrapper import ModelWrapper
    from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
    from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
    from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
    from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
    from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
    from qonnx.transformation.general import GiveUniqueNodeNames

    problems = preflight()
    if problems:
        raise SystemExit("PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems))

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    model = ModelWrapper(fifo_onnx)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(A.FPGA_PART, clk_ns))     # 3.9
    model = model.transform(HLSSynthIP())                       # 3.9
    model = model.transform(CreateStitchedIP(A.FPGA_PART, clk_ns))  # 3.10
    stitched = os.path.join(out_dir, "step04_stitched.onnx")
    model.save(stitched)

    model = model.transform(ZynqBuild(platform=A.FPGA_BOARD,
                                      period_ns=clk_ns,
                                      enable_build_pdb_debug=False))  # 3.10.1
    model = model.transform(MakePYNQDriver(platform="zynq-iodma"))     # 3.10.2
    final = os.path.join(out_dir, "step04_synth.onnx")
    model.save(final)

    manifest = {
        "step": "04_synth",
        "fpga_part": A.FPGA_PART, "board": A.FPGA_BOARD,
        "vivado_version_expected": A.VIVADO_VERSION,
        "vivado_version_reference": A.REFERENCE_VIVADO_VERSION,
        "target_clock_mhz": 1000.0 / clk_ns,
        "clock_mode": A.CLOCK_MODE,
        "strategy": strategy or "Vivado defaults (A29)",
        "decisions": ["D1 speed grade -1", "D10 FINN 0.10 + Vivado 2022.2"],
        "paper_table7_actual_for_context": A.TABLE7_ACTUAL,
        "artifacts": {"stitched": stitched, "final": final},
    }
    with open(os.path.join(out_dir, "step04_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return final, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", help="model with FIFOs from step 03")
    ap.add_argument("--out-dir", default="finn_build_out")
    ap.add_argument("--clk-ns", type=float, default=5.0, help="5.0 ns = 200 MHz")
    ap.add_argument("--preflight", action="store_true",
                    help="check toolchain/board prerequisites and exit")
    args = ap.parse_args(argv)

    if args.preflight:
        problems = preflight()
        print("PREFLIGHT for T1.9 (part {}, board {}, Vivado {})".format(
            A.FPGA_PART, A.FPGA_BOARD, A.VIVADO_VERSION))
        if not problems:
            print("  all prerequisites present")
            return 0
        for p in problems:
            print("  [BLOCKED] {}".format(p))
        return 1

    if not args.onnx:
        ap.error("--onnx is required unless --preflight is given")
    final, manifest = run(args.onnx, args.out_dir, args.clk_ns)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
