#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.throughput_sweep import mean_sd    # noqa: E402

PAPER_TABLE13 = {"idle_no_bitstream": 1.9, 50: 2.12, 100: 2.38, 200: 2.55}
PAPER_TABLE14 = {
    50:  {1: 20.71, 100: 24.63},
    100: {1: 36.32, 100: 43.86},
    200: {1: 64.63, 100: 81.88},
}
PAPER_THERMAL = {"die_measured_c": 53.1, "vivado_max_ambient_c": 46.3}
PAPER_PL_ONLY_W = (0.22, 0.65)     # PAPER FACT 4.3.2, DERIVED by the authors
SAMPLE_HZ = 2.0                     # A34
AVERAGE_WINDOW_S = 30.0             # A34


class Instrument(object):
    def read_volts_amps(self):      # pragma: no cover
        raise NotImplementedError


class ManualInstrument(Instrument):
    def __init__(self, volts=5.0):
        self.volts = float(volts)

    def read_volts_amps(self):      # pragma: no cover (interactive)
        a = float(input("  enter measured current (A) @ {:.2f} V: ".format(self.volts)))
        return self.volts, a


def average_power(inst, seconds=AVERAGE_WINDOW_S, hz=SAMPLE_HZ):
    samples = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        v, a = inst.read_volts_amps()
        samples.append(v * a)
        time.sleep(1.0 / hz)
    m, s = mean_sd(samples)
    return {"watts_mean": m, "watts_sd": s, "n": len(samples),
            "window_s": seconds, "sample_hz": hz}


def run(inst, bitfile, clocks, throughputs=None, driver_dir=".", out_json=None,
        die_temp_c=None):
    sys.path.insert(0, driver_dir)
    from driver_base import FINNExampleOverlay
    from driver import io_shape_dict

    rows = {}
    print("STEP 1: idle, NO bitstream loaded (I4 -- enables direct PL-only measurement)")
    rows["idle_no_bitstream"] = average_power(inst)

    for clk in clocks:
        accel = FINNExampleOverlay(bitfile_name=bitfile, platform="zynq-iodma",
                                   io_shape_dict=io_shape_dict, batch_size=100,
                                   fclk_mhz=float(clk))
        print("STEP 2: {} MHz under continuous inference load".format(clk))
        import threading
        stop = threading.Event()

        def load():
            while not stop.is_set():
                accel.throughput_test()

        th = threading.Thread(target=load, daemon=True)
        th.start()
        rows[clk] = average_power(inst)
        stop.set()
        th.join(timeout=5.0)
        # OUR ADDITION I4: PL-only power measured, not derived
        rows[clk]["pl_only_watts_measured"] = (
            rows[clk]["watts_mean"] - rows["idle_no_bitstream"]["watts_mean"])
        if throughputs and clk in throughputs:
            for bs, fps in throughputs[clk].items():
                rows[clk].setdefault("fps_per_watt", {})[str(bs)] = \
                    fps / rows[clk]["watts_mean"]

    doc = {
        "_provenance": {
            "setup": "PAPER FACT 4.3 / Fig. 18 -- 5 V bench supply + IR camera, "
                     "no heatsink",
            "sampling": "A34 -- {} Hz over {} s windows".format(SAMPLE_HZ,
                                                               AVERAGE_WINDOW_S),
            "pl_only": "I4 -- MEASURED by differencing, not derived (the paper "
                       "derives 0.22-0.65 W arithmetically)",
            "instrument_uncertainty": "REQUIRED -- record the supply's stated "
                                      "accuracy here (A34)",
        },
        "rows": rows,
        "die_temp_c_measured": die_temp_c,
        "paper_table13_for_context": PAPER_TABLE13,
        "paper_table14_for_context": PAPER_TABLE14,
        "paper_thermal_for_context": PAPER_THERMAL,
        "paper_pl_only_derived_for_context": PAPER_PL_ONLY_W,
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
    ap.add_argument("--instrument", default="manual", choices=["manual"])
    ap.add_argument("--clocks", default="50,100,200")
    ap.add_argument("--die-temp-c", type=float, default=None,
                    help="IR camera reading (instrument absent on this host)")
    ap.add_argument("--json", default="results/power_measured.json")
    ap.add_argument("--show-reference", action="store_true")
    args = ap.parse_args(argv)

    if args.show_reference:
        print("PAPER FACT Table 13 (W): {}".format(PAPER_TABLE13))
        print("PAPER FACT Table 14 (FPS/W, batch 1 / 100):")
        for c in (50, 100, 200):
            print("  {:>4} MHz  {:>6.2f} / {:>6.2f}".format(
                c, PAPER_TABLE14[c][1], PAPER_TABLE14[c][100]))
        print("PAPER FACT 4.3.2 thermal: {}".format(PAPER_THERMAL))
        print("PAPER FACT 4.3.2 PL-only (DERIVED by the authors): {} W".format(
            PAPER_PL_ONLY_W))
        print("\nMISSING INSTRUMENTS (D9 PENDING): 5 V bench supply with current")
        print("readout; IR thermal camera. Without these, every FPS/W figure in")
        print("the paper is unreproducible even with the board.")
        return 0

    if not args.bitfile:
        print("HARDWARE BLOCKER: D9 PENDING -- board AND 5 V bench supply AND IR")
        print("camera all absent. Use --show-reference for context.")
        return 1
    run(ManualInstrument(), args.bitfile,
        [int(c) for c in args.clocks.split(",")], None, args.driver_dir,
        args.json, args.die_temp_c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
