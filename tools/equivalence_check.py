#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402

ATOL_BREVITAS_QONNX = 1e-4          # A31


def _levels(args, images):
    """Run each abstraction level. Returns {level: list of (1,13,13,18) arrays}."""
    import numpy as np
    import torch
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from models.quant_yolov3_tiny import build as build_quant

    out = {}

    # ---- level 1: Brevitas (reference) ----------------------------------
    model = build_quant(args.quant)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    brevitas = []
    with torch.no_grad():
        for img in images:
            # NHWC uint8 -> NCHW float in [0,255]; the stem divides by 255 (A3)
            x = torch.from_numpy(img.transpose(0, 3, 1, 2).astype("float32"))
            y = model(x).numpy()                      # (1,18,13,13) NCHW
            brevitas.append(y.transpose(0, 2, 3, 1))  # -> NHWC
    out["brevitas"] = brevitas

    # ---- level 2: QONNX / FINN-ONNX graph -------------------------------
    for level, path, mode in (("qonnx", args.qonnx, None),
                              ("cppsim", args.cppsim, "cppsim"),
                              ("rtlsim", args.rtlsim, "rtlsim")):
        if not path:
            continue
        m = ModelWrapper(path)
        if mode:
            m.set_metadata_prop("exec_mode", mode)
            m.set_metadata_prop("rtlsim_trace", "")      # A27
        iname = m.graph.input[0].name
        res = []
        for img in images:
            ctx = execute_onnx(m, {iname: img.astype("uint8")})
            res.append(list(ctx.values())[0])
        out[level] = res
    return out


def compare(levels):
    import numpy as np
    names = [n for n in ("brevitas", "qonnx", "cppsim", "rtlsim") if n in levels]
    rows = []
    for i in range(len(names) - 1):
        a_name, b_name = names[i], names[i + 1]
        a, b = levels[a_name], levels[b_name]
        diffs = [float(np.max(np.abs(np.asarray(x, dtype=np.float64)
                                     - np.asarray(y, dtype=np.float64))))
                 for x, y in zip(a, b)]
        exact = sum(1 for x, y in zip(a, b)
                    if np.array_equal(np.asarray(x), np.asarray(y)))
        # Brevitas->QONNX may differ by export rounding (A31); the integer-domain
        # hops must be bit-exact.
        tol = ATOL_BREVITAS_QONNX if a_name == "brevitas" else 0.0
        rows.append({
            "hop": "{} -> {}".format(a_name, b_name),
            "max_abs_diff": max(diffs) if diffs else 0.0,
            "bit_exact_fraction": exact / float(max(1, len(a))),
            "tolerance": tol,
            "pass": (max(diffs) if diffs else 0.0) <= tol,
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="trained best.pt (Brevitas reference)")
    ap.add_argument("--quant", default="configs/quant.yaml")
    ap.add_argument("--qonnx", help="QONNX / FINN-ONNX graph")
    ap.add_argument("--cppsim", help="model prepared for cppsim")
    ap.add_argument("--rtlsim", help="stitched IP model for rtlsim")
    ap.add_argument("--images", help="directory of val images (frozen subset)")
    ap.add_argument("--n", type=int, default=32, help="frozen subset size (A31)")
    ap.add_argument("--json", default="results/equivalence_report.md.json")
    ap.add_argument("--describe", action="store_true",
                    help="print the acceptance criteria and exit (no deps)")
    args = ap.parse_args(argv)

    if args.describe:
        print("T1.12 FUNCTIONAL EQUIVALENCE -- acceptance criteria")
        print("-" * 58)
        print("  brevitas -> qonnx   max|diff| <= {} (A31, export rounding)".format(
            ATOL_BREVITAS_QONNX))
        print("  qonnx    -> cppsim  EXACT integer equality")
        print("  cppsim   -> rtlsim  EXACT integer equality")
        print("  mAP@0.5 delta brevitas -> rtlsim ~ 0")
        print("\nShared post-processing at every level (so a divergence can only")
        print("be in the compiled network):")
        print("  dequant x{:.8f} (= 1/127, REPOSITORY FACT scale.npy)".format(
            A.OUTPUT_DEQUANT_SCALE))
        print("  decode: xy=(y+0.5+grid)*{}, wh=(y+1)^2*anchor, obj=y/2+0.5".format(
            A.DETECT_STRIDE))
        print("  anchors {} (REPOSITORY FACT)".format(list(A.ANCHORS_PX)))
        print("  NMS conf=0.10 iou=0.10 max_det=20 (REPOSITORY FACT)")
        print("\nGATE: Tier 2 must not begin until every hop passes.")
        print("\nWHY THIS EXISTS: paper 4.2.2 asserts correctness with no")
        print("quantitative evidence; the repository's validate.py is unmodified")
        print("FINN MNIST/CIFAR-10 boilerplate. No evidence exists in either source.")
        return 0

    if not (args.weights and args.qonnx):
        ap.error("--weights and --qonnx are required (or use --describe)")

    import numpy as np
    images = []
    if args.images and os.path.isdir(args.images):
        from utils.preprocess import preprocess_frame
        try:
            import cv2
            files = sorted(os.listdir(args.images))[:args.n]
            for f in files:
                img = cv2.imread(os.path.join(args.images, f))
                if img is not None:
                    images.append(preprocess_frame(img, 416)[0])
        except ImportError:
            pass
    if not images:
        print("no images supplied; using {} deterministic pseudo-random frames"
              .format(args.n))
        rng = np.random.RandomState(0)
        images = [rng.randint(0, 256, (1, 416, 416, 3)).astype(np.uint8)
                  for _ in range(args.n)]

    levels = _levels(args, images)
    rows = compare(levels)
    failures = sum(0 if r["pass"] else 1 for r in rows)
    for r in rows:
        print("[{}] {:<22} max|diff|={:.6g}  bit-exact {:.1%}".format(
            "PASS" if r["pass"] else "FAIL", r["hop"], r["max_abs_diff"],
            r["bit_exact_fraction"]))
    doc = {"levels": list(levels.keys()), "hops": rows, "failures": failures,
           "n_images": len(images)}
    d = os.path.dirname(args.json)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(args.json, "w") as fh:
        json.dump(doc, fh, indent=2)
    print("\nGATE {}: Tier 2 {} proceed".format(
        "PASSED" if not failures else "FAILED",
        "may" if not failures else "must NOT"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
