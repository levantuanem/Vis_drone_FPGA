#!/usr/bin/env python3
import argparse
import json
import os
import sys


def read_labels(label_dir):
    boxes, per_image = [], []
    for name in sorted(os.listdir(label_dir)):
        if not name.endswith(".txt"):
            continue
        n = 0
        with open(os.path.join(label_dir, name)) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 5:
                    continue
                boxes.append([float(p) for p in parts[1:]])
                n += 1
        per_image.append(n)
    return boxes, per_image


def histogram(values, bins):
    counts = [0] * (len(bins) + 1)
    for v in values:
        placed = False
        for i, edge in enumerate(bins):
            if v <= edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="converted dataset root")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--img-size", type=int, default=416)
    ap.add_argument("--json", help="write stats here")
    args = ap.parse_args()

    report = {"img_size": args.img_size, "splits": {}}
    px_bins = [8, 16, 24, 32, 48, 64, 96, 128]

    for split in args.splits.split(","):
        d = os.path.join(args.root, "labels", split)
        if not os.path.isdir(d):
            print("skip {} (no {})".format(split, d))
            continue
        boxes, per_image = read_labels(d)
        n_img = len(per_image)
        widths_px = [b[2] * args.img_size for b in boxes]
        heights_px = [b[3] * args.img_size for b in boxes]
        diag = [(w * w + h * h) ** 0.5 for w, h in zip(widths_px, heights_px)]
        s = {
            "images": n_img,
            "images_with_boxes": sum(1 for n in per_image if n),
            "boxes": len(boxes),
            "boxes_per_image_mean": round(len(boxes) / float(max(1, n_img)), 3),
            "boxes_per_image_max": max(per_image) if per_image else 0,
            "box_w_px_median": round(sorted(widths_px)[len(widths_px) // 2], 2)
                               if widths_px else 0,
            "box_h_px_median": round(sorted(heights_px)[len(heights_px) // 2], 2)
                               if heights_px else 0,
            "size_hist_bins_px": px_bins,
            "size_hist_diag_px": histogram(diag, px_bins),
        }
        report["splits"][split] = s
        print("--- {} ---".format(split))
        for k in sorted(s):
            print("  {:<24} {}".format(k, s[k]))

    # Context: the repository's anchors, in the same 416-pixel frame.
    report["repository_anchors_px"] = [[10, 14], [23, 27], [37, 58]]
    print("\nREPOSITORY FACT anchors (px @416): (10,14) (23,27) (37,58), stride 32")
    print("Compare against box_*_px_median above: a large mismatch is a finding,")
    print("not a bug -- record it in DECISIONS.md rather than changing the anchors.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
