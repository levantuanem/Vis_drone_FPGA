#!/usr/bin/env python
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.throughput_sweep import mean_sd    # noqa: E402

STAGES = ("preprocessing", "driver_exec", "rescale", "detect", "nms", "box_process")
PAPER_TABLE10 = {   # PAPER FACT -- ms, mean +/- sd. Context only.
    50:  {"preprocessing": (32.907, 1.040), "driver_exec": (28.913, 0.700),
          "rescale": (0.511, 0.213), "detect": (4.988, 1.137),
          "nms": (10.399, 2.644), "box_process": (11.373, 2.778),
          "total_postprocess": (27.980, 5.046)},
    100: {"preprocessing": (34.043, 1.212), "driver_exec": (17.891, 0.788),
          "rescale": (0.500, 0.176), "detect": (4.670, 0.237),
          "nms": (9.335, 1.051), "box_process": (11.211, 3.170),
          "total_postprocess": (26.362, 3.643)},
    200: {"preprocessing": (32.191, 1.490), "driver_exec": (12.121, 0.326),
          "rescale": (0.481, 0.024), "detect": (4.746, 4.502),
          "nms": (9.232, 2.383), "box_process": (10.749, 2.950),
          "total_postprocess": (25.819, 8.754)},
}
PAPER_DRIVER_PENALTY = {50: 0.21, 100: 0.35, 200: 0.50}   # PAPER FACT 4.2.6
DEFAULT_FRAMES = 200          # A33


class Timer(object):
    def __init__(self):
        self.acc = {k: [] for k in STAGES}

    def record(self, stage, seconds):
        self.acc[stage].append(seconds * 1e3)

    def summary(self):
        out = {}
        for k, v in self.acc.items():
            m, s = mean_sd(v)
            out[k] = {"mean_ms": m, "sd_ms": s, "n": len(v)}
        post = [sum(x) for x in zip(self.acc["detect"], self.acc["nms"],
                                    self.acc["box_process"])]
        m, s = mean_sd(post)
        out["total_postprocess"] = {"mean_ms": m, "sd_ms": s, "n": len(post)}
        return out


def run(bitfile, frames, clocks, source, driver_dir=".", out_json=None):
    import numpy as np
    sys.path.insert(0, driver_dir)
    from driver_base import FINNExampleOverlay
    from driver import io_shape_dict
    from utils.preprocess import preprocess_frame
    from utils.postprocess import dequantise, nhwc_to_nchw, decode, \
        non_max_suppression
    from utils.boxes import scale_coords

    rows = {}
    for clk in clocks:
        accel = FINNExampleOverlay(bitfile_name=bitfile, platform="zynq-iodma",
                                   io_shape_dict=io_shape_dict, batch_size=1,
                                   fclk_mhz=float(clk))   # PAPER FACT batch 1
        t = Timer()
        for frame in source(frames):
            t0 = time.time()
            tensor, gain, pad = preprocess_frame(frame, 416)
            t.record("preprocessing", time.time() - t0)

            t0 = time.time()
            out = accel.execute(tensor)
            t.record("driver_exec", time.time() - t0)

            t0 = time.time()
            deq = dequantise(out)
            t.record("rescale", time.time() - t0)

            t0 = time.time()
            pred = decode(nhwc_to_nchw(deq))
            t.record("detect", time.time() - t0)

            t0 = time.time()
            dets = non_max_suppression(pred)
            t.record("nms", time.time() - t0)

            t0 = time.time()
            for d in dets:
                if len(d):
                    d[:, :4] = scale_coords((416, 416), d[:, :4], frame.shape[:2])
            t.record("box_process", time.time() - t0)
        rows[clk] = t.summary()
        print("--- {} MHz ---".format(clk))
        for k in list(STAGES) + ["total_postprocess"]:
            r = rows[clk][k]
            print("  {:<20} {:>8.3f} +/- {:>6.3f} ms (n={})".format(
                k, r["mean_ms"], r["sd_ms"], r["n"]))

    doc = {
        "_provenance": {
            "stages": "PAPER FACT Table 10",
            "batch": "1 (PAPER FACT 4.2.6)",
            "frames_per_point": frames, "_frames_id": "A33",
            "driver_penalty_claim": PAPER_DRIVER_PENALTY,
        },
        "rows": rows, "paper_table10_for_context": PAPER_TABLE10,
    }
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(doc, fh, indent=2)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bitfile")
    ap.add_argument("--driver-dir", default=".")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--clocks", default="50,100,200")
    ap.add_argument("--json", default="results/latency_breakdown.json")
    ap.add_argument("--show-reference", action="store_true")
    args = ap.parse_args(argv)

    if args.show_reference:
        print("PAPER FACT Table 10 (ms, mean +/- sd) -- CONTEXT ONLY")
        hdr = "{:<20}" + "{:>20}" * 3
        print(hdr.format("stage", "50 MHz", "100 MHz", "200 MHz"))
        for k in list(STAGES) + ["total_postprocess"]:
            vals = ["{:.3f} +/- {:.3f}".format(*PAPER_TABLE10[c][k])
                    for c in (50, 100, 200)]
            print(hdr.format(k, *vals))
        print("\nPAPER FACT 4.2.6 driver penalty: {}".format(PAPER_DRIVER_PENALTY))
        print("Only `driver_exec` scales with clock -- every other stage is")
        print("PS-bound, which is the paper's own explanation for the 30 FPS")
        print("end-to-end ceiling in 4.4.")
        return 0

    if not args.bitfile:
        print("HARDWARE BLOCKER: D9 PENDING -- no Z-turn Board V2 present.")
        return 1

    import numpy as np

    def source(n):
        for _ in range(n):
            yield (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)

    run(args.bitfile, args.frames, [int(c) for c in args.clocks.split(",")],
        source, args.driver_dir, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
