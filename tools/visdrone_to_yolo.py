#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import os
import sys

VISDRONE_CATEGORIES = {
    0: "ignored-regions", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car",
    5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus",
    10: "motor",
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def image_size(path):
    """Read width/height without a full decode. Pillow preferred; PNG/JPEG fallback."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size            # (w, h)
    except ImportError:
        pass
    with open(path, "rb") as fh:
        head = fh.read(32)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            import struct
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
        fh.seek(0)
        data = fh.read()
    # minimal JPEG SOF scan
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return int(w), int(h)
        seg = (data[i + 2] << 8) | data[i + 3]
        i += 2 + seg
    raise RuntimeError("cannot determine image size for {} (install Pillow)".format(path))


def parse_annotation(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip().rstrip(",")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                vals = [int(float(p)) for p in parts[:8]] + [0] * (8 - len(parts[:8]))
            except ValueError:
                continue
            rows.append({
                "left": vals[0], "top": vals[1], "w": vals[2], "h": vals[3],
                "score": vals[4], "category": vals[5],
                "truncation": vals[6], "occlusion": vals[7],
            })
    return rows


def survey_categories(ann_dir):
    """Count boxes per category id -- used to VERIFY A1 rather than assume it."""
    counts = collections.Counter()
    files = 0
    for name in sorted(os.listdir(ann_dir)):
        if not name.endswith(".txt"):
            continue
        files += 1
        for r in parse_annotation(os.path.join(ann_dir, name)):
            counts[r["category"]] += 1
    return counts, files


def convert_split(src, dst, split, car_index, drop_score_zero=True,
                  link_images=True, dry_run=False):
    img_dir = os.path.join(src, "images")
    ann_dir = os.path.join(src, "annotations")
    if not os.path.isdir(img_dir):
        raise SystemExit("missing {}".format(img_dir))
    has_ann = os.path.isdir(ann_dir)

    out_img = os.path.join(dst, "images", split)
    out_lbl = os.path.join(dst, "labels", split)
    if not dry_run:
        for d in (out_img, out_lbl):
            if not os.path.isdir(d):
                os.makedirs(d)

    stats = {"images": 0, "images_with_labels": 0, "boxes_kept": 0,
             "boxes_dropped_other_class": 0, "boxes_dropped_score_zero": 0,
             "boxes_dropped_degenerate": 0, "empty_label_files": 0,
             "boxes_per_image": [], "box_area_frac": []}

    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        stats["images"] += 1
        img_path = os.path.join(img_dir, name)
        stem = os.path.splitext(name)[0]
        lines = []
        if has_ann:
            ann_path = os.path.join(ann_dir, stem + ".txt")
            if os.path.isfile(ann_path):
                W, H = image_size(img_path)
                for r in parse_annotation(ann_path):
                    if r["category"] != car_index:
                        stats["boxes_dropped_other_class"] += 1
                        continue
                    if drop_score_zero and r["score"] == 0:   # A6
                        stats["boxes_dropped_score_zero"] += 1
                        continue
                    if r["w"] <= 0 or r["h"] <= 0:
                        stats["boxes_dropped_degenerate"] += 1
                        continue
                    cx = (r["left"] + r["w"] / 2.0) / W
                    cy = (r["top"] + r["h"] / 2.0) / H
                    bw = r["w"] / float(W)
                    bh = r["h"] / float(H)
                    if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                        stats["boxes_dropped_degenerate"] += 1
                        continue
                    cx = min(max(cx, 0.0), 1.0)
                    cy = min(max(cy, 0.0), 1.0)
                    lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(cx, cy, bw, bh))
                    stats["boxes_kept"] += 1
                    stats["box_area_frac"].append(bw * bh)
        stats["boxes_per_image"].append(len(lines))
        if lines:
            stats["images_with_labels"] += 1
        else:
            stats["empty_label_files"] += 1

        if dry_run:
            continue
        link = os.path.join(out_img, name)
        if not os.path.lexists(link):
            if link_images:
                os.symlink(os.path.abspath(img_path), link)
            else:
                import shutil
                shutil.copy2(img_path, link)
        with open(os.path.join(out_lbl, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
    return stats


def summarise(stats):
    bpi = stats.pop("boxes_per_image", [])
    areas = stats.pop("box_area_frac", [])
    out = dict(stats)
    if bpi:
        out["boxes_per_image_mean"] = round(sum(bpi) / float(len(bpi)), 3)
        out["boxes_per_image_max"] = max(bpi)
    if areas:
        areas = sorted(areas)
        out["box_area_frac_median"] = round(areas[len(areas) // 2], 6)
        out["box_area_frac_p95"] = round(areas[int(len(areas) * 0.95)], 6)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, action="append", metavar="SPLIT=DIR",
                    help="e.g. --src train=/data/VisDrone2019-DET-train "
                         "(repeat for val, test-dev)")
    ap.add_argument("--dst", required=True, help="output dataset root")
    ap.add_argument("--car-index", type=int, default=4,
                    help="VisDrone category id for 'car' (A1; default 4, VERIFIED "
                         "by --verify-index)")
    ap.add_argument("--verify-index", action="store_true",
                    help="survey category counts and sanity-check --car-index "
                         "before converting (strongly recommended; A1)")
    ap.add_argument("--keep-score-zero", action="store_true",
                    help="keep VisDrone ignored-region boxes (overrides A6)")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    srcs = {}
    for item in args.src:
        if "=" not in item:
            raise SystemExit("--src must be SPLIT=DIR, got {!r}".format(item))
        k, v = item.split("=", 1)
        srcs[k] = v

    manifest = {
        "_provenance": {
            "paper": "PAPER FACT 3.1.2 -- Task 1 only, car only, YOLO XYWH",
            "decision_D6": "car only, no merging of van/truck/bus",
            "A1_category_index": "VERIFIED, not assumed" if args.verify_index
                                 else "NOT VERIFIED -- rerun with --verify-index",
            "A6_score_zero": "kept" if args.keep_score_zero else "dropped",
            "A7_truncation_occlusion": "all levels kept",
            "D3_reporting_split": "val",
        },
        "car_index_used": args.car_index,
        "nc": 1, "names": ["car"],
        "splits": {},
    }

    if args.verify_index:
        print("=== A1 category-index verification ===")
        for split, src in sorted(srcs.items()):
            ann = os.path.join(src, "annotations")
            if not os.path.isdir(ann):
                print("  {}: no annotations/ (labels withheld upstream?) - skipped"
                      .format(split))
                continue
            counts, nfiles = survey_categories(ann)
            total = sum(counts.values())
            print("  {} ({} annotation files, {} boxes)".format(split, nfiles, total))
            for cid in sorted(counts):
                print("    id {:>2} {:<18} {:>9} boxes{}".format(
                    cid, VISDRONE_CATEGORIES.get(cid, "?"), counts[cid],
                    "   <-- --car-index" if cid == args.car_index else ""))
            manifest["splits"].setdefault(split, {})["category_survey"] = {
                str(k): v for k, v in sorted(counts.items())}
            if args.car_index not in counts:
                raise SystemExit(
                    "A1 VERIFICATION FAILED: category id {} absent from {}. "
                    "Inspect the survey above and pass the correct --car-index."
                    .format(args.car_index, split))
        print()

    for split, src in sorted(srcs.items()):
        print("converting {} <- {}".format(split, src))
        st = convert_split(src, args.dst, split, args.car_index,
                           drop_score_zero=not args.keep_score_zero,
                           link_images=not args.copy, dry_run=args.dry_run)
        st = summarise(st)
        manifest["splits"].setdefault(split, {}).update(st)
        for k in sorted(st):
            print("    {:<30} {}".format(k, st[k]))

    total_imgs = sum(s.get("images", 0) for s in manifest["splits"].values())
    total_boxes = sum(s.get("boxes_kept", 0) for s in manifest["splits"].values())
    manifest["totals"] = {"images": total_imgs, "car_boxes": total_boxes}
    # PAPER FACT 3.1.2 context: 10,209 images / 2.6M boxes across ALL 10 classes
    manifest["paper_reference"] = {"images_all_splits": 10209,
                                   "boxes_all_classes": 2600000}

    if not args.dry_run:
        blob = json.dumps(manifest, sort_keys=True).encode()
        manifest["content_hash"] = hashlib.sha256(blob).hexdigest()[:16]
        path = os.path.join(args.dst, "manifest.json")
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print("\nwrote {}".format(path))
    print("\ntotals: {} images, {} car boxes".format(total_imgs, total_boxes))
    if not args.verify_index:
        print("WARNING: A1 not verified. Re-run with --verify-index.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
