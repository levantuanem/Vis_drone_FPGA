#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402

ROW = re.compile(r"\|\s*([A-Za-z0-9_ .\-/()]+?)\s*\|\s*([\d,]+)\s*\|")


def parse_utilisation(path):
    """Extract LUT / FF / BRAM / DSP from a `report_utilization` text file."""
    wanted = {
        "lut": ("Slice LUTs", "CLB LUTs", "LUT as Logic"),
        "ff": ("Slice Registers", "CLB Registers", "Register as Flip Flop"),
        "bram36": ("Block RAM Tile", "RAMB36/FIFO", "RAMB36E1"),
        "bram18": ("RAMB18", "RAMB18E1"),
        "dsp": ("DSPs", "DSP48E1", "DSP Blocks"),
    }
    found = {}
    with open(path) as fh:
        for line in fh:
            m = ROW.match(line)
            if not m:
                continue
            name = m.group(1).strip()
            try:
                val = int(m.group(2).replace(",", ""))
            except ValueError:
                continue
            for key, labels in wanted.items():
                if name in labels and key not in found:
                    found[key] = val
    # Vivado reports 36K tiles; the paper counts 18K blocks
    if "bram18" not in found and "bram36" in found:
        found["bram18"] = found["bram36"] * 2
    return found


def parse_timing(path):
    """Extract WNS / TNS / WHS from a `report_timing_summary` text file."""
    out = {}
    pats = {"wns": r"WNS\(ns\)", "tns": r"TNS\(ns\)", "whs": r"WHS\(ns\)"}
    with open(path) as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        for key, pat in pats.items():
            if re.search(pat, line) and key not in out:
                for nxt in lines[i + 1:i + 4]:
                    nums = re.findall(r"-?\d+\.\d+", nxt)
                    if nums:
                        out[key] = float(nums[0])
                        break
    out["timing_met"] = (out.get("wns", -1) >= 0 and out.get("whs", -1) >= 0)
    return out


def table7(actual, estimated=None):
    est = estimated or {}
    rows = []
    for key, dev, label in (("bram18", A.DEVICE_BRAM18K, "BRAM_18K"),
                            ("lut", A.DEVICE_LUT, "LUT"),
                            ("dsp", A.DEVICE_DSP48E1, "DSP")):
        a = actual.get(key)
        rows.append({
            "resource": label, "estimated": est.get(key), "actual": a,
            "available": dev,
            "util_pct": round(100.0 * a / dev, 2) if a else None,
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--utilisation", help="report_utilization text file")
    ap.add_argument("--timing", help="report_timing_summary text file")
    ap.add_argument("--folding", default="configs/folding.json",
                    help="our estimates, for the calibration delta")
    ap.add_argument("--json", default="results/resources.json")
    ap.add_argument("--show-reference", action="store_true")
    args = ap.parse_args(argv)

    if args.show_reference or not args.utilisation:
        print("PAPER FACT Table 7 (CONTEXT ONLY):")
        print("  {:<10} {:>10} {:>8} {:>10} {:>8}".format(
            "resource", "estimated", "actual", "available", "util%"))
        t = A.TABLE7_ACTUAL
        print("  {:<10} {:>10} {:>8} {:>10} {:>8}".format(
            "BRAM_18K", t["bram18k_estimated"], t["bram18k"], 140, 98.57))
        print("  {:<10} {:>10} {:>8} {:>10} {:>8}".format(
            "LUT", t["lut_estimated"], t["lut"], 53200, 78.20))
        print("  {:<10} {:>10} {:>8} {:>10} {:>8}".format(
            "DSP", t["dsp_estimated"], t["dsp"], 220, 92.73))
        print("\nPAPER FACT 4.2.4: estimated BRAM (200) exceeded the 140 available,")
        print("so Vivado converted some to LUTRAM; Vitis HLS inflates LUT further.")
        if not args.utilisation:
            print("\nNo --utilisation given (T1.9 not run; Vivado not installed).")
        return 0

    actual = parse_utilisation(args.utilisation)
    timing = parse_timing(args.timing) if args.timing else {}
    est = None
    if os.path.isfile(args.folding):
        with open(args.folding) as fh:
            d = json.load(fh)
        e = d.get("estimated", {})
        est = {"bram18": e.get("bram18k"), "lut": e.get("lut"), "dsp": e.get("dsp")}

    rows = table7(actual, est)
    print("{:<10} {:>10} {:>8} {:>10} {:>8}".format(
        "resource", "estimated", "actual", "available", "util%"))
    for r in rows:
        print("{:<10} {:>10} {:>8} {:>10} {:>8}".format(
            r["resource"], r["estimated"] if r["estimated"] is not None else "-",
            r["actual"] if r["actual"] is not None else "-",
            r["available"], r["util_pct"] if r["util_pct"] is not None else "-"))
    if timing:
        print("\ntiming: WNS={} TNS={} WHS={} met={}".format(
            timing.get("wns"), timing.get("tns"), timing.get("whs"),
            timing.get("timing_met")))

    calib = {}
    for r in rows:
        if r["estimated"] and r["actual"]:
            calib[r["resource"]] = round(r["actual"] / float(r["estimated"]), 4)
    if calib:
        print("\nCALIBRATION FACTORS (actual / our estimate): {}".format(calib))
        print("Apply to configs/resource_model.yaml, then RE-RUN")
        print("tools/folding_search.py and re-synthesise (the folding.json")
        print("currently on disk is marked PROVISIONAL for exactly this reason).")

    doc = {"table7": rows, "timing": timing, "calibration_factors": calib,
           "paper_table7_for_context": A.TABLE7_ACTUAL}
    d = os.path.dirname(args.json)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(args.json, "w") as fh:
        json.dump(doc, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
