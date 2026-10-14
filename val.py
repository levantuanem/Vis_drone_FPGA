#!/usr/bin/env python3
"""Evaluation entry point -- produces the paper Table 5 equivalent.

DECISION D3 / A2: the reporting split is `val`, labelled explicitly. `--split
train` is also offered because the paper contradicts itself (spec P6: Table 5's
caption says "Validation results", 4.1's body says "training set"), and computing
both is how we tell a reconstruction failure apart from a labelling difference.

DEPENDENCY NOT INSTALLED: torch==1.13.1, brevitas==0.9.0. NOT RUN in this phase.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAPER_TABLE5 = {   # PAPER FACT -- context for interpretation, never a target
    "pruned":   {"precision": 0.4816, "recall": 0.1716, "map50": 0.1768,
                 "box": 0.1531, "obj": 0.3049, "cls": 0.0},
    "original": {"precision": 0.6706, "recall": 0.3844, "map50": 0.4176,
                 "box": 0.1220, "obj": 0.1653, "cls": 0.0},
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/visdrone_car.yaml")
    ap.add_argument("--quant", default="configs/quant.yaml")
    ap.add_argument("--split", default="val", choices=["train", "val", "test-dev"])
    ap.add_argument("--img", type=int, default=416)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", default=None)
    ap.add_argument("--compare-paper", default=None, choices=["pruned", "original"],
                    help="print the paper's values alongside ours, as CONTEXT")
    args = ap.parse_args(argv)

    import torch
    import yaml
    from models.quant_yolov3_tiny import build as build_quant
    from models.detect import Detect
    from utils.datasets import build_loader
    from train import evaluate_loader

    with open(args.data) as fh:
        data = yaml.safe_load(fh)
    root = data.get("path")
    if not root or not os.path.isdir(root):
        raise SystemExit("dataset root {!r} not present".format(root))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_quant(args.quant).to(device)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    head = Detect(nc=data.get("nc", 1)).to(device)

    loader = build_loader(root, args.split, args.img, args.batch_size,
                          augment=False, workers=4, shuffle=False)
    res = evaluate_loader(model, head, loader, device, head.na, head.no)
    res["split"] = args.split            # always labelled -- DECISION D3

    print("=" * 60)
    print("RESULTS  split={}  (DECISION D3: val is the reporting split)".format(
        args.split))
    print("=" * 60)
    for k in ("precision", "recall", "map50", "tp", "fp", "n_gt"):
        print("  {:<12} {}".format(k, res[k]))

    if args.compare_paper:
        ref = PAPER_TABLE5[args.compare_paper]
        print("\nPAPER FACT Table 5 ({}) -- CONTEXT ONLY, not a target:".format(
            args.compare_paper))
        for k in ("precision", "recall", "map50"):
            print("  {:<12} ours {:.4f}   paper {:.4f}".format(k, res[k], ref[k]))
        print("\nIf these diverge, check in this order: A15 BatchNorm, A5 "
              "hyper-parameters, A2/D3 split, A4 letterbox, A6 ignored regions.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
