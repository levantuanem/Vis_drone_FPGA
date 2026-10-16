#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys

PAPER_TABLE11 = {
    "top_wrapper": 3.129, "top_i": 3.129,
    "StreamingDataflowPartition_1": 1.554, "zynq_ps": 1.497,
    "smartconnect_0": 0.042, "idma0": 0.019, "odma0": 0.012,
    "axi_interconnect_0": 0.004,
}
PAPER_TABLE12 = {
    "Clocks": 0.124, "Slice Logic": 0.428, "LUT as Logic": 0.384,
    "CARRY4": 0.025, "Register": 0.014, "F7/F8 Muxes": 0.002,
    "LUT as Distributed RAM": 0.002, "Signals": 0.612, "Block RAM": 0.262,
    "DSPs": 0.207, "PS7": 1.496, "Static Power": 0.229, "Total": 3.358,
}

PWR_ROW = re.compile(r"\|\s*([A-Za-z0-9_ .\-/()\[\]]+?)\s*\|\s*([\d.]+)\s*\|")
CONF = re.compile(r"Confidence\s+level\s*\|?\s*([A-Za-z]+)", re.I)


def parse(path):
    modules, resources, confidence = {}, {}, None
    with open(path) as fh:
        text = fh.read()
    m = CONF.search(text)
    if m:
        confidence = m.group(1)
    for line in text.splitlines():
        hit = PWR_ROW.match(line)
        if not hit:
            continue
        name = hit.group(1).strip()
        try:
            val = float(hit.group(2))
        except ValueError:
            continue
        if name in PAPER_TABLE12 or name in ("Total", "Static Power",
                                             "Dynamic", "Device Static"):
            resources.setdefault(name, val)
        else:
            modules.setdefault(name, val)
    return {"modules": modules, "resources": resources, "confidence": confidence}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", help="report_power text file")
    ap.add_argument("--json", default="results/power_estimated.json")
    ap.add_argument("--show-reference", action="store_true")
    args = ap.parse_args(argv)

    if args.show_reference or not args.report:
        print("PAPER FACT Table 11 -- per-module power (W), CONTEXT ONLY:")
        for k, v in sorted(PAPER_TABLE11.items(), key=lambda kv: -kv[1]):
            print("  {:<32} {:>7.3f}".format(k, v))
        print("\nPAPER FACT Table 12 -- per-resource power (W), CONTEXT ONLY:")
        for k, v in sorted(PAPER_TABLE12.items(), key=lambda kv: -kv[1]):
            print("  {:<32} {:>7.3f}".format(k, v))
        print("\nPAPER FACT 4.3.1: measured power is considerably LOWER than this;")
        print("the estimates are worst-case/synthetic. A30: vectorless activity.")
        if not args.report:
            print("\nNo --report given (T1.11 not run; Vivado not installed).")
        return 0

    res = parse(args.report)
    res["_provenance"] = {"switching_activity": "vectorless (A30)",
                          "confidence_level": res.get("confidence")}
    res["paper_table11_for_context"] = PAPER_TABLE11
    res["paper_table12_for_context"] = PAPER_TABLE12
    print(json.dumps(res, indent=2))
    d = os.path.dirname(args.json)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(args.json, "w") as fh:
        json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
