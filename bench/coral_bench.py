#!/usr/bin/env python3
"""T3.1 -- Coral Edge TPU benchmark. HARDWARE BLOCKER: D9 PENDING (no Coral).

PAPER FACT 5.1: two driver configurations, `libedgetpu` standard mode and max
mode; internal frequencies 250 MHz and 500 MHz; batch size 1 on BOTH sides "to
allow for a fair comparison, at the cost of degraded performance for this work".
PAPER FACT Table 15: 9.69 ms / 103.2 FPS at 250 MHz; 6.73 ms / 148.7 FPS at
500 MHz; power 1.2 W / 2.4 W.

TWO THINGS THE PAPER ITSELF FLAGS OR IMPLIES, CARRIED FORWARD HERE:
  - The Coral power figures are DATASHEET ESTIMATES, not measurements; the 1.2 W
    value is obtained by halving 2.4 W for half frequency (the paper's own
    footnote says so). OUR ADDITION I3: measure Coral power on the same 5 V bench
    used in T2.3, making the comparison symmetric.
  - On the paper's own numbers the Coral at 250 MHz is MORE efficient
    (86.0 vs 64.5 FPS/W) than this work. The paper's efficiency claim is against
    FPGA literature (Table 16), not against the ASIC.

OUR ASSUMPTION A9 / DECISION D5: the Coral runs the PRUNED Table-1 topology, so
both platforms execute the identical network. The paper never states which
topology its FP32 run used.
OUR ADDITION I8: report the Edge TPU compiler's op-mapping fraction -- ops falling
back to the CPU would materially change this result and the paper never reports it.

DEPENDENCIES NOT INSTALLED: tflite-runtime / pycoral, libedgetpu (feranick fork,
PAPER FACT ref 68). NOT RUN.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.throughput_sweep import mean_sd    # noqa: E402

PAPER_TABLE15_CORAL = {
    250: {"latency_ms": 9.69, "fps": 103.2, "watts_datasheet_estimate": 1.2,
          "fps_per_watt": 86.0},
    500: {"latency_ms": 6.73, "fps": 148.7, "watts_datasheet_estimate": 2.4,
          "fps_per_watt": 62.0},
}
PAPER_TABLE15_OURS_REFERENCE = {   # the paper's Zynq column, batch 1
    50: {"latency_ms": 22.77, "fps": 43.91, "watts": 2.12, "fps_per_watt": 20.7},
    100: {"latency_ms": 11.56, "fps": 86.47, "watts": 2.38, "fps_per_watt": 36.3},
    200: {"latency_ms": 6.07, "fps": 164.8, "watts": 2.55, "fps_per_watt": 64.5},
}


def run(model_path, reps=100, warmup=10, out_json=None, measured_watts=None):
    import numpy as np
    from pycoral.utils.edgetpu import make_interpreter

    interp = make_interpreter(model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    shape = tuple(inp["shape"])
    dtype = inp["dtype"]

    x = np.zeros(shape, dtype=dtype)
    for _ in range(warmup):
        interp.set_tensor(inp["index"], x)
        interp.invoke()

    lat = []
    for _ in range(reps):          # PAPER FACT 5.1: batch size 1
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        lat.append((time.perf_counter() - t0) * 1e3)

    m, s = mean_sd(lat)
    doc = {
        "_provenance": {
            "batch": "1 (PAPER FACT 5.1 fairness rule)",
            "topology": "pruned Table-1 (A9 / DECISION D5)",
            "power": ("MEASURED (I3)" if measured_watts
                      else "NOT MEASURED -- paper used a datasheet estimate"),
            "op_mapping": "I8 -- record the compiler's TPU/CPU split",
        },
        "model": model_path, "reps": reps,
        "latency_ms_mean": m, "latency_ms_sd": s,
        "fps_mean": 1000.0 / m if m else None,
        "measured_watts": measured_watts,
        "fps_per_watt": (1000.0 / m / measured_watts) if (m and measured_watts)
                        else None,
        "paper_table15_coral_for_context": PAPER_TABLE15_CORAL,
        "paper_table15_zynq_for_context": PAPER_TABLE15_OURS_REFERENCE,
    }
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(doc, fh, indent=2)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="*_edgetpu.tflite from tools/export_tflite.py")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measured-watts", type=float, default=None, help="I3")
    ap.add_argument("--json", default="results/coral_bench.json")
    ap.add_argument("--show-reference", action="store_true")
    args = ap.parse_args(argv)

    if args.show_reference:
        print("PAPER FACT Table 15 -- Coral (CONTEXT ONLY):")
        for f, r in sorted(PAPER_TABLE15_CORAL.items()):
            print("  {:>4} MHz  {:>6.2f} ms  {:>6.1f} FPS  {:>4.1f} W* "
                  "{:>6.1f} FPS/W".format(f, r["latency_ms"], r["fps"],
                                          r["watts_datasheet_estimate"],
                                          r["fps_per_watt"]))
        print("  * datasheet ESTIMATE, not measured (the paper's own footnote);")
        print("    1.2 W is 2.4 W halved for half frequency.")
        print("\nNote on the paper's own numbers: Coral at 250 MHz reaches")
        print("86.0 FPS/W vs this work's 64.5 -- the paper's efficiency claim is")
        print("against FPGA literature (Table 16), not against the ASIC.")
        print("\nHARDWARE BLOCKER: D9 PENDING -- no Coral Edge TPU present.")
        print("T1.13 (TFLite + edgetpu_compiler) needs NO Coral and is not blocked.")
        return 0

    if not args.model:
        print("HARDWARE BLOCKER: D9 PENDING -- no Coral Edge TPU present.")
        return 1
    print(json.dumps(run(args.model, args.reps, args.warmup, args.json,
                         args.measured_watts), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
