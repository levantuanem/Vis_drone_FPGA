#!/usr/bin/env python3
"""Folding derivation (T1.6) -- DECISION D7.

Re-derives the per-layer PE/SIMD configuration that paper §3.7.2 delegates to a
repository which does not contain it (spec P7). Nothing is copied or invented:
the procedure below is paper §3.6.1-3.6.2 and the cost function is the cycle
model verified against paper Table 8 (see tools/cycle_model.py).

PAPER FACT §3.6.1 -- the procedure, implemented verbatim:
  1. start fully folded (every PE and SIMD = 1)
  2. iteratively unfold the slowest layer
  3. balance the cycles-per-layer metric within "reasonable resource limits"
  4. keep small tail layers fully folded ("no advantage in accelerating an
     already fast layer")
  5. stop when the slowest layer is fully unfolded (the bottleneck)

PAPER FACT §3.6.2 -- the key move: when DSPs are exhausted, switch MVUs to the
  HLS LUT implementation style to keep unfolding using the abundant LUT fabric.

REPOSITORY FACT -- two endpoints are pinned and are enforced as hard constraints:
  conv 1  input stream folding 3  => MVU SIMD = k^2*C_in = 27  (ishape_folded)
  conv 10 output PE = 1                                        (oshape_folded)

OUTPUT: configs/folding.json, marked provisional until the resource model is
calibrated against T1.9's real Vivado report (paper Table 7).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A            # noqa: E402
from tools import cycle_model as CM          # noqa: E402

# OUR ASSUMPTION A23 -- §3.6.1's "reasonable resource limits" is unquantified.
BUDGET_FRACTION = 0.95

# REPOSITORY FACT constraints (driver.py / model.pkl folded shapes)
PINNED = {
    "MVAU_rtl_0": {"SIMD": 27},   # ishape_folded=(1,416,416,1,3) -> full window
    "MVAU_rtl_4": {"PE": 1},      # oshape_folded=(1,13,13,18,1)
}


def budgets():
    return {
        "dsp": int(A.DEVICE_DSP48E1 * BUDGET_FRACTION),
        "lut": int(A.DEVICE_LUT * BUDGET_FRACTION),
        "bram18k": int(A.DEVICE_BRAM18K * BUDGET_FRACTION),
    }


def initial_state():
    """Fully folded, honouring the two pinned REPOSITORY FACT endpoints."""
    folding = CM.fully_folded()
    styles = {l.mvu: l.impl for l in A.CONV_LAYERS}
    for mvu, pins in PINNED.items():
        for key, val in pins.items():
            folding[mvu][key] = val
    return folding, styles


def fixed_floor(res):
    """Slowest non-MVU node: the floor the MVUs are balanced against.

    PAPER FACT §3.6.1 keeps MaxPool/SWG/Thresholding folded, so the global
    bottleneck is a fixed node. For this architecture it is
    StreamingMaxPool_L2 at 416^2 + 208^2 = 216,320 cycles, which is exactly the
    node and the number paper Table 8 reports.
    """
    fixed = [n for n in res["nodes"] if n["kind"] == "MaxPool"]
    worst = max(fixed, key=lambda n: n["cycles"])
    return worst["cycles"], worst["name"]


def unfold_options(layer, pe, simd):
    """Next-larger legal (PE, SIMD) steps. PE | C_out, SIMD | k^2*C_in."""
    opts = []
    pe_divs = CM.divisors(layer.mh)
    simd_divs = CM.divisors(layer.mw)
    pins = PINNED.get(layer.mvu, {})
    if "PE" not in pins:
        nxt = [d for d in pe_divs if d > pe]
        if nxt:
            opts.append((nxt[0], simd))
    if "SIMD" not in pins:
        nxt = [d for d in simd_divs if d > simd]
        if nxt:
            opts.append((pe, nxt[0]))
    return opts


def resource_of(layer, pe, simd, style):
    bram, lutram = CM.mvu_weight_memory(layer, pe, simd)
    return {
        "dsp": CM.mvu_dsp(layer, pe, simd, style),
        "lut": CM.mvu_lut(layer, pe, simd, style) + lutram,
        "bram18k": bram,
    }


def fits(totals, delta, bud):
    for k in ("dsp", "lut", "bram18k"):
        if totals[k] + delta[k] > bud[k]:
            return False
    return True


def search(max_iters=2000, verbose=False):
    folding, styles = initial_state()
    bud = budgets()
    saturated = set()
    log = []

    for it in range(max_iters):
        res = CM.evaluate(folding, styles)
        floor_cycles, floor_name = fixed_floor(res)

        mvu_nodes = [n for n in res["nodes"]
                     if n["kind"] == "MVU" and n["name"] not in saturated]
        over = [n for n in mvu_nodes if n["cycles"] > floor_cycles]
        if not over:
            log.append({"iter": it, "action": "balanced",
                        "floor": floor_cycles, "floor_node": floor_name})
            break

        node = max(over, key=lambda n: n["cycles"])
        layer = A.layer_by_mvu(node["name"])
        pe, simd = node["PE"], node["SIMD"]
        style = styles[node["name"]]
        cur_res = resource_of(layer, pe, simd, style)
        totals = {k: res[k] for k in ("dsp", "lut", "bram18k")}

        best = None
        for cand_style in (style, "HLS_LUT" if layer.impl == "HLS" else style):
            for (npe, nsimd) in unfold_options(layer, pe, simd):
                new_res = resource_of(layer, npe, nsimd, cand_style)
                delta = {k: new_res[k] - cur_res[k] for k in new_res}
                trial_totals = {k: totals[k] - cur_res[k] for k in totals}
                if not fits(trial_totals, new_res, bud):
                    continue
                new_cycles = CM.mvu_cycles(layer, npe, nsimd)
                saved = node["cycles"] - new_cycles
                if saved <= 0:
                    continue
                cost = max(1.0, delta["dsp"] * 40.0 + delta["lut"] * 1.0
                           + delta["bram18k"] * 120.0)
                score = saved / cost
                if best is None or score > best["score"]:
                    best = {"score": score, "PE": npe, "SIMD": nsimd,
                            "style": cand_style, "cycles": new_cycles}

        if best is None:
            saturated.add(node["name"])
            log.append({"iter": it, "action": "saturated", "node": node["name"],
                        "cycles": node["cycles"],
                        "reason": "no legal unfold fits the remaining budget"})
            continue

        folding[node["name"]] = {"PE": best["PE"], "SIMD": best["SIMD"]}
        styles[node["name"]] = best["style"]
        log.append({"iter": it, "action": "unfold", "node": node["name"],
                    "PE": best["PE"], "SIMD": best["SIMD"], "style": best["style"],
                    "cycles_before": node["cycles"], "cycles_after": best["cycles"]})
        if verbose:
            print("[{:>3}] {:<12} PE={:<4} SIMD={:<4} {:<8} {:>12,} -> {:>12,}".format(
                it, node["name"], best["PE"], best["SIMD"], best["style"],
                node["cycles"], best["cycles"]))

    res = CM.evaluate(folding, styles)
    floor_cycles, floor_name = fixed_floor(res)
    return folding, styles, res, log, (floor_cycles, floor_name), sorted(saturated)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="configs/folding.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the result but do not write the file")
    args = ap.parse_args()

    folding, styles, res, log, floor, saturated = search(verbose=args.verbose)
    floor_cycles, floor_name = floor

    print()
    print("=" * 78)
    print("FOLDING DERIVATION (T1.6)  --  DECISION D7, re-derived not copied")
    print("=" * 78)
    print("balance floor      : {} @ {:,} cycles  (PAPER Table 8 max node: {:,})".format(
        floor_name, floor_cycles, A.TABLE8_ESTIMATED["max_node_cycles"]))
    print("resulting max node : {} @ {:,} cycles".format(
        res["max_node_name"], res["max_node_cycles"]))
    print("total cycles       : {:,}".format(res["total_cycles"]))
    print("throughput @200MHz : {:.1f} FPS".format(
        CM.throughput_fps(res["max_node_cycles"], 200.0)))
    print("throughput @100MHz : {:.1f} FPS".format(
        CM.throughput_fps(res["max_node_cycles"], 100.0)))
    print("DSP   (exact)      : {} / {}".format(res["dsp"], A.DEVICE_DSP48E1))
    print("LUT   (provisional): {} / {}".format(res["lut"], A.DEVICE_LUT))
    print("BRAM18K (provis.)  : {} / {}".format(res["bram18k"], A.DEVICE_BRAM18K))
    if saturated:
        print("saturated nodes    : {}".format(", ".join(saturated)))
    print()
    print("{:<12} {:>5} {:>6} {:>9} {:>14} {:>7} {:>8}".format(
        "MVU", "PE", "SIMD", "style", "cycles", "DSP", "LUT"))
    print("-" * 78)
    for n in res["nodes"]:
        if n["kind"] != "MVU":
            continue
        print("{:<12} {:>5} {:>6} {:>9} {:>14,} {:>7} {:>8}".format(
            n["name"], n["PE"], n["SIMD"], n["style"], n["cycles"],
            n["dsp"], n["lut"]))
    print("-" * 78)
    print("LUT/BRAM figures are PROVISIONAL (OUR ASSUMPTION resource model);")
    print("calibrate against T1.9's Vivado report, then re-run this search.")

    doc = {
        "_provenance": {
            "decision": "D7 (PROJECT MANDATE) -- re-derived, never copied",
            "procedure": "PAPER FACT paper sections 3.6.1 and 3.6.2",
            "cost_model": "DERIVED FACT -- verified against paper Table 8",
            "pinned_constraints": "REPOSITORY FACT driver.py folded shapes",
            "resource_model": "OUR ASSUMPTION -- provisional, see configs/resource_model.yaml",
            "status": "PROVISIONAL -- not validated by synthesis (T1.9 not run)",
            "budget_fraction": BUDGET_FRACTION,
        },
        "fpga_part": A.FPGA_PART,
        "target_clock_mhz": A.TARGET_CLOCK_MHZ,
        "balance_floor": {"node": floor_name, "cycles": floor_cycles},
        "folding": folding,
        "styles": styles,
        "estimated": {
            "total_cycles": res["total_cycles"],
            "max_node_cycles": res["max_node_cycles"],
            "max_node_name": res["max_node_name"],
            "dsp": res["dsp"], "lut": res["lut"], "bram18k": res["bram18k"],
            "calibration_required": True,
        },
        "search_log": log,
    }

    if args.dry_run:
        return 0
    d = os.path.dirname(args.out)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
    print("\nwrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
