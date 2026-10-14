#!/usr/bin/env python3
"""T2.3 -- measured power and thermal. HARDWARE BLOCKER: D9 PENDING (3 items).

PAPER FACT 4.3 / Fig. 18: the whole board is powered from a 5 V DC bench supply
with current readout; die temperature is read with an IR thermal camera; no
heatsink, static air only.

PAPER FACT Table 13: idle (no bitstream) 1.9 W; 50 MHz 2.12 W; 100 MHz 2.38 W;
200 MHz 2.55 W. PAPER FACT 4.3.2 derives PL-only draw of 0.22-0.65 W by
subtracting the 1.9 W PS+DDR figure.
PAPER FACT Table 14: FPS/W 20.71/24.63, 36.32/43.86, 64.63/81.88 (batch 1/100).
PAPER FACT 4.3.2: die 53.1 C measured vs Vivado's 46.3 C max-ambient estimate.

OUR ADDITION I4: measure PL-only power directly, by differencing
bitstream-loaded against not-loaded, instead of deriving it arithmetically.
OUR ASSUMPTION A34: sampling rate / averaging window fixed here and recorded,
with instrument uncertainty reported. The paper states none of this.

MISSING INSTRUMENTS (verified absent on this host):
  - 5 V DC bench supply with current measurement
  - IR thermal camera
Without them, every FPS/W figure in the paper is unreproducible EVEN WITH the
board, because power cannot be inferred from throughput.

DEPENDENCIES NOT INSTALLED: pynq; plus an instrument driver whose API depends on
the supply model. `--instrument manual` lets the numbers be entered by hand.
"""

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
    """Abstract 5 V supply with current readout.

    Subclass per supply model. `ManualInstrument` is the fallback that keeps the
    measurement protocol reproducible without a programmable instrument.
    """

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
