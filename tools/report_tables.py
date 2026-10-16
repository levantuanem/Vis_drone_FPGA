#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A        # noqa: E402
from tools import cycle_model as CM      # noqa: E402

BLOCKERS = {
    "T1.1": "VisDrone not downloaded", "T1.2": "training not run",
    "T1.3": "training not run", "T1.9": "Vivado 2022.1 not installed",
    "T1.10": "FINN/Verilator not installed", "T1.11": "Vivado not installed",
    "T2.1": "D9 PENDING (board)", "T2.2": "D9 PENDING (board)",
    "T2.3": "D9 PENDING (board + 5 V bench + IR camera)",
    "T3.1": "D9 PENDING (Coral)",
}


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def table6():
    rows = []
    for mvu, ref in A.TABLE6_REFERENCE.items():
        l = A.layer_by_mvu(mvu)
        rows.append([mvu, "{:,}".format(l.macs), "{:,}".format(l.weights),
                     l.dtype_pair, "{:,}".format(ref["macs"]),
                     "{:,}".format(ref["weights"]),
                     "OK" if (l.macs == ref["macs"] and l.weights == ref["weights"])
                     else "MISMATCH"])
    t = A.totals()
    body = md_table(["MVU", "our MAC", "our weights", "dtype",
                     "paper MAC", "paper weights", "match"], rows)
    totals = md_table(
        ["total", "ours", "paper", "match"],
        [[k, "{:,}".format(t[k]), "{:,}".format(A.TABLE6_TOTALS[k]),
          "OK" if t[k] == A.TABLE6_TOTALS[k] else "MISMATCH"]
         for k in sorted(A.TABLE6_TOTALS)])
    return ("## Table 6 equivalent — per-MVU MAC and weights (T1.8)\n\n"
            "Status: **PRODUCED** (arithmetic only; no data, GPU or tools needed).\n\n"
            + body + "\n\n" + totals)


def table8():
    path = "configs/folding.json"
    if not os.path.isfile(path):
        return ("## Table 8 equivalent (T1.8 / T1.10 / T2.1)\n\n"
                "Status: configs/folding.json absent — run tools/folding_search.py.")
    with open(path) as fh:
        doc = json.load(fh)
    res = CM.evaluate(doc["folding"], doc.get("styles", {}))
    e = A.TABLE8_ESTIMATED
    rows = [
        ["total cycles", "{:,}".format(res["total_cycles"]),
         "{:,}".format(e["total_cycles"]), "—", "—"],
        ["max node cycles", "{:,}".format(res["max_node_cycles"]),
         "{:,}".format(e["max_node_cycles"]), "741,670", "—"],
        ["max node name", res["max_node_name"], e["max_node_name"], "—", "—"],
        ["latency @100 MHz (ms)",
         "{:.2f}".format(CM.latency_ms(res["total_cycles"], 100.0)),
         "{:.2f}".format(e["latency_ms_at_100mhz"]), "7.41", "11.56"],
        ["throughput @100 MHz (FPS)",
         "{:.2f}".format(CM.throughput_fps(res["max_node_cycles"], 100.0)),
         "{:.2f}".format(e["throughput_fps_at_100mhz"]), "134.83", "104.39"],
    ]
    return ("## Table 8 equivalent — estimated / RTLSim / actual\n\n"
            "Status: **estimated column PRODUCED**; RTLSim blocked ({}); "
            "actual blocked ({}).\n\n".format(BLOCKERS["T1.10"], BLOCKERS["T2.1"])
            + md_table(["metric", "OURS (estimated)", "paper estimated",
                        "paper RTLSim", "paper actual"], rows)
            + "\n\nNote spec P9: the paper's RTLSim latency equals its max-node "
              "cycle count, which is dimensionally odd. We report throughput "
              "cycles and first-frame latency separately instead.\n"
              "\nNote: our total-cycle figure differs from the paper's because our "
              "folding differs in detail, and because Table 8 prints "
              "\"4,586,895 -> critical path\", which is ambiguous between a sum "
              "over nodes and a critical path.")


def folding_table():
    path = "configs/folding.json"
    if not os.path.isfile(path):
        return ""
    with open(path) as fh:
        doc = json.load(fh)
    res = CM.evaluate(doc["folding"], doc.get("styles", {}))
    rows = [[n["name"], n["PE"], n["SIMD"], n["style"], "{:,}".format(n["cycles"]),
             n["dsp"], n["lut"], n["bram18k"]]
            for n in res["nodes"] if n["kind"] == "MVU"]
    prov = doc.get("_provenance", {})
    return ("## Derived folding (T1.6) — DECISION D7\n\n"
            "Status: **{}**\n\nBalance floor: {} @ {:,} cycles "
            "(paper Table 8 max node: {:,}).\n\n".format(
                prov.get("status", "unknown"),
                doc["balance_floor"]["node"], doc["balance_floor"]["cycles"],
                A.TABLE8_ESTIMATED["max_node_cycles"])
            + md_table(["MVU", "PE", "SIMD", "style", "cycles", "DSP",
                        "LUT (prov.)", "BRAM18K (prov.)"], rows)
            + "\n\nPE/SIMD are DERIVED, never copied — paper §3.7.2 delegates them "
              "to a repository that does not contain them (spec P7). LUT/BRAM are "
              "provisional until calibrated against T1.9.")


def blocked_tables():
    items = [
        ("Table 5", "accuracy P/R/mAP@0.5", "T1.2 + T1.3", BLOCKERS["T1.3"]),
        ("Table 7", "resource estimated vs actual", "T1.9", BLOCKERS["T1.9"]),
        ("Tables 11, 12", "Vivado power estimate", "T1.11", BLOCKERS["T1.11"]),
        ("Table 9", "throughput sweep", "T2.1", BLOCKERS["T2.1"]),
        ("Table 10", "per-stage latency", "T2.2", BLOCKERS["T2.2"]),
        ("Tables 13, 14", "measured power / FPS per W", "T2.3", BLOCKERS["T2.3"]),
        ("Table 15", "Coral comparison", "T1.13 + T3.1", BLOCKERS["T3.1"]),
        ("Table 16", "literature comparison", "T3.2", "needs our T2.1/T2.3 row"),
        ("Fig. 15", "training curves + PR", "T1.3", BLOCKERS["T1.3"]),
        ("Figs. 3-7", "graph stages", "T1.4 + T1.5", "FINN not installed"),
        ("Figs. 12/13/16/17", "Vivado views", "T1.9", BLOCKERS["T1.9"]),
        ("Fig. 19", "field frames", "T2.5", "D8 deferred + D9 PENDING"),
    ]
    return ("## Not yet produced\n\n"
            + md_table(["paper artifact", "content", "experiment", "blocker"],
                       [[a, b, c, d] for a, b, c, d in items])
            + "\n\nNo paper result is claimed as reproduced while its blocker "
              "stands.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="results/TABLES.md")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args(argv)

    parts = ["# Paper-equivalent tables",
             "",
             "Generated by `tools/report_tables.py`. PAPER FACT values appear as "
             "CONTEXT only — per the reconstruction mandate we do not optimise "
             "toward them.",
             "", table6(), "", folding_table(), "", table8(), "", blocked_tables()]
    text = "\n".join(p for p in parts if p is not None)

    if args.stdout:
        print(text)
    else:
        d = os.path.dirname(args.out)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
