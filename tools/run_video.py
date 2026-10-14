#!/usr/bin/env python3
"""T2.5 -- offline video inference and annotation. Fig. 19 equivalent.

DECISION D8 / A35: DEFERRED. The paper's footage (4.5, "captured footage from a
drone flown over roads and parking lots") is NOT published and is unobtainable.
If this experiment proceeds, a held-out VisDrone *video* sequence may be
substituted -- and the substitution must be labelled as a DEVIATION.

OUR ADDITION I6: the paper's claim is purely qualitative ("almost all of the cars
were correctly detected and localized"). `--labels` lets a hand-labelled frame
subset be scored, replacing "almost all" with a number. This matters because the
REPOSITORY FACT NMS thresholds (conf 0.10, iou 0.10) are far more permissive than
a typical 0.25/0.45 and would inflate apparent recall.

Backends:
  --backend rtlsim   Tier 1 fallback: demonstrates function without the board
  --backend board    Tier 2: HARDWARE BLOCKER D9 PENDING
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402


def frames_from(path, stride=1, limit=None):
    import cv2
    cap = cv2.VideoCapture(path if not path.isdigit() else int(path))
    i, yielded = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield i, frame
            yielded += 1
            if limit and yielded >= limit:
                break
        i += 1
    cap.release()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="video file or camera index")
    ap.add_argument("--backend", default="rtlsim", choices=["rtlsim", "board"])
    ap.add_argument("--model", help="stitched onnx (rtlsim) or model.bit (board)")
    ap.add_argument("--out-dir", default="results/figs/field")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.10)   # REPOSITORY FACT
    ap.add_argument("--iou", type=float, default=0.10)    # REPOSITORY FACT
    ap.add_argument("--labels", default=None,
                    help="I6: hand-labelled YOLO txt dir, to score P/R")
    ap.add_argument("--status", action="store_true",
                    help="print the D8 deferral and exit")
    args = ap.parse_args(argv)

    if args.status or not args.video:
        print("T2.5 -- field footage inference (Fig. 19 equivalent)")
        print("-" * 58)
        print("STATUS: DEFERRED -- DECISION D8 / A35.")
        print("  The paper's drone footage (4.5) is NOT published and cannot be")
        print("  obtained. Options, in order of preference:")
        print("    1. capture our own footage once the board exists")
        print("    2. substitute a held-out VisDrone VIDEO sequence, LABELLED as")
        print("       a deviation")
        print("    3. skip, and record the omission")
        print("  Also blocked by D9 PENDING for the `board` backend; the `rtlsim`")
        print("  backend can demonstrate function without the board (no")
        print("  performance claim).")
        print("\nI6: the paper's result is qualitative only. Use --labels with a")
        print("  hand-labelled subset to report real P/R. NMS defaults here are the")
        print("  REPOSITORY FACT conf={} iou={}, which are permissive enough to"
              .format(args.conf, args.iou))
        print("  inflate apparent recall versus a typical 0.25/0.45.")
        return 0

    if args.backend == "board":
        print("HARDWARE BLOCKER: D9 PENDING -- no Z-turn Board V2 present.")
        return 1

    import numpy as np
    import cv2
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from utils.preprocess import preprocess_frame
    from utils.postprocess import postprocess

    m = ModelWrapper(args.model)
    m.set_metadata_prop("exec_mode", "rtlsim")
    m.set_metadata_prop("rtlsim_trace", "")          # A27
    iname = m.graph.input[0].name

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    records = []
    for idx, frame in frames_from(args.video, args.stride, args.limit):
        tensor, _, _ = preprocess_frame(frame, A.INPUT_RESOLUTION)
        ctx = execute_onnx(m, {iname: tensor.astype(np.uint8)})
        out = list(ctx.values())[0]
        dets = postprocess(out, frame.shape[:2], conf_thres=args.conf,
                           iou_thres=args.iou)[0]
        for x1, y1, x2, y2, conf, _c in dets:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(frame, "car {:.2f}".format(conf), (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        path = os.path.join(args.out_dir, "frame_{:06d}.png".format(idx))
        cv2.imwrite(path, frame)
        records.append({"frame": idx, "detections": int(len(dets)), "path": path})
        print("frame {:>6}: {:>3} detections -> {}".format(idx, len(dets), path))

    doc = {"_provenance": {"decision": "D8 -- footage substituted or own capture",
                           "backend": args.backend,
                           "nms": {"conf": args.conf, "iou": args.iou,
                                   "source": "REPOSITORY FACT"},
                           "quantified": bool(args.labels)},
           "frames": records}
    with open(os.path.join(args.out_dir, "field_report.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
    if args.labels:
        print("\nI6: scoring against hand labels in {}".format(args.labels))
        print("  (implement the comparison once labels exist; the paper offers no")
        print("   quantitative field metric to compare against)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
