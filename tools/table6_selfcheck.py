#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402


class Check(object):
    def __init__(self):
        self.rows = []
        self.failures = 0

    def expect(self, name, got, want):
        ok = (got == want)
        if not ok:
            self.failures += 1
        self.rows.append({"check": name, "got": got, "want": want, "ok": ok})
        return ok

    def note(self, name, value):
        self.rows.append({"check": name, "got": value, "want": value, "ok": True})


def run():
    c = Check()

    # -- 0. structural sanity ------------------------------------------------
    c.expect("layer count (Table 1 rows)", len(A.LAYERS), 16)
    c.expect("convolution count", len(A.CONV_LAYERS), 10)
    c.expect("maxpool count", len(A.POOL_LAYERS), 6)
    errs = A.shape_chain_is_consistent()
    c.expect("shape chain consistent (no errors)", errs, [])

    # -- 1. the two architecture corrections are actually in place ----------
    l12 = A.LAYERS[11]
    c.expect("L12 is maxpool", l12.kind, "maxpool")
    c.expect("L12 stride == 1 (DECISION D2)", l12.stride, 1)
    c.expect("L12 preserves 13x13", (l12.in_hw, l12.out_hw), (13, 13))
    l16 = A.LAYERS[15]
    c.expect("L16 kernel == 3 (spec P3)", l16.k, 3)
    c.expect("L16 Cin == 104", l16.cin, 104)
    c.expect("L16 Cout == 18", l16.cout, 18)
    c.expect("L16 output activation == INT8 (spec P11)", l16.a_out_bits, 8)
    c.expect("L16 activation is QuantHardTanh", l16.act, "QuantHardTanh")

    # -- 2. 10/10 weight counts and 10/10 MAC counts vs Table 6 -------------
    weight_hits = 0
    mac_hits = 0
    for mvu, ref in A.TABLE6_REFERENCE.items():
        layer = A.layer_by_mvu(mvu)
        w_ok = c.expect("{} weights".format(mvu), layer.weights, ref["weights"])
        m_ok = c.expect("{} MAC".format(mvu), layer.macs, ref["macs"])
        c.expect("{} dtype pair".format(mvu), layer.dtype_pair, ref["dtype"])
        weight_hits += 1 if w_ok else 0
        mac_hits += 1 if m_ok else 0
    c.expect("weight counts matched (of 10)", weight_hits, 10)
    c.expect("MAC counts matched (of 10)", mac_hits, 10)

    # -- 3. Table 6 totals row ---------------------------------------------
    t = A.totals()
    c.expect("MAC @ 4b-4b", t["mac_4b_4b"], A.TABLE6_TOTALS["mac_4b_4b"])
    c.expect("MAC @ 8b-8b", t["mac_8b_8b"], A.TABLE6_TOTALS["mac_8b_8b"])
    c.expect("MAC @ 4b-8b", t["mac_4b_8b"], A.TABLE6_TOTALS["mac_4b_8b"])
    c.expect("weights total", t["weights_total"], A.TABLE6_TOTALS["weights_total"])
    c.expect("weights @ 8b", t["weights_8b"], A.TABLE6_TOTALS["weights_8b"])
    c.expect("weights @ 4b", t["weights_4b"], A.TABLE6_TOTALS["weights_4b"])

    # -- 4. the explicitly required layer-15 value -------------------------
    c.expect("L15 (MVAU_rtl_3) MAC == 8,858,304", A.layer_by_mvu("MVAU_rtl_3").macs,
             8_858_304)

    # -- 5. RTL/HLS specialisation decoded from Table 6 names --------------
    rtl = [l.idx for l in A.CONV_LAYERS if l.impl == "RTL"]
    hls = [l.idx for l in A.CONV_LAYERS if l.impl == "HLS"]
    c.expect("RTL conv rows (Table 1 numbering)", rtl, [1, 3, 7, 15, 16])
    c.expect("HLS conv rows (Table 1 numbering)", hls, [5, 9, 11, 13, 14])
    c.expect("RTL count", len(rtl), 5)
    c.expect("HLS count", len(hls), 5)

    # -- 6. I/O contract (REPOSITORY FACT) ---------------------------------
    c.expect("accelerator output shape", (1, A.LAYERS[-1].out_hw,
                                          A.LAYERS[-1].out_hw, A.LAYERS[-1].cout),
             A.IO_CONTRACT["oshape_normal"])
    c.expect("accelerator input shape", (1, A.INPUT_RESOLUTION, A.INPUT_RESOLUTION,
                                         A.INPUT_CHANNELS),
             A.IO_CONTRACT["ishape_normal"])
    c.expect("out channels == na*(nc+5)", A.NUM_ANCHORS * (A.NUM_CLASSES + 5), 18)

    # -- 7. derived facts that must stay self-consistent -------------------
    c.expect("dequant scale is 1/127", round(A.OUTPUT_DEQUANT_SCALE, 8), 0.00787402)
    c.note("grand total MAC", sum(l.macs for l in A.CONV_LAYERS))
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also write the result table to this path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    c = run()

    if not args.quiet:
        width = max(len(r["check"]) for r in c.rows) + 2
        print("=" * (width + 44))
        print("TABLE 6 ARITHMETIC SELF-CHECK  (GATE 1)")
        print("=" * (width + 44))
        for r in c.rows:
            mark = "PASS" if r["ok"] else "FAIL"
            if r["got"] == r["want"]:
                detail = "{}".format(r["got"])
            else:
                detail = "got {!r} want {!r}".format(r["got"], r["want"])
            print("[{}] {:<{w}} {}".format(mark, r["check"], detail, w=width))
        print("=" * (width + 44))
        print("{} checks, {} failures".format(len(c.rows), c.failures))

    if args.json:
        d = os.path.dirname(args.json)
        if d:
            try:
                os.makedirs(d)
            except OSError:
                pass
        with open(args.json, "w") as fh:
            json.dump({"checks": c.rows, "failures": c.failures}, fh, indent=2)

    return 1 if c.failures else 0


if __name__ == "__main__":
    sys.exit(main())
