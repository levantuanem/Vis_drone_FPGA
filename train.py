#!/usr/bin/env python3
"""Training entry point for T1.2 (FP32 baseline) and T1.3 (4W4A QAT).

DEPENDENCY NOT INSTALLED: torch==1.13.1, torchvision==0.14.1, brevitas==0.9.0
and the rest of PAPER FACT Table 2. See requirements/README.md.
THIS SCRIPT IS NOT RUN in this phase -- source only.

PAPER FACT Table 3 (the only training values the paper gives):
    epochs 300 | batch 64 | img 416x416 | optimizer SGD | patience 30
Everything else is OUR ASSUMPTION A5 from configs/hyp.scratch-low.yaml.

Example (NOT executed here):
    python3 train.py --data data/visdrone_car.yaml \
        --cfg models/yolov3-tiny-pruned-4w4a.yaml \
        --quant configs/quant.yaml --hyp configs/hyp.scratch-low.yaml \
        --epochs 300 --batch-size 64 --img 416 --name pruned_4w4a

Interrupted (Ctrl+C / kill)? last.pt is written every epoch; continue with the
same --epochs/--batch-size/--img:
    python3 train.py ... --resume runs/pruned_4w4a/last.pt
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import arch_spec as A   # noqa: E402  stdlib-only


def load_yaml(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/visdrone_car.yaml")
    ap.add_argument("--cfg", default="models/yolov3-tiny-pruned-4w4a.yaml")
    ap.add_argument("--quant", default="configs/quant.yaml")
    ap.add_argument("--hyp", default="configs/hyp.scratch-low.yaml")
    ap.add_argument("--epochs", type=int, default=300)        # PAPER FACT Table 3
    ap.add_argument("--batch-size", type=int, default=64)     # PAPER FACT Table 3
    ap.add_argument("--img", type=int, default=416)           # PAPER FACT Table 3
    ap.add_argument("--optimizer", default="SGD",
                    choices=["SGD"])                          # PAPER FACT Table 3
    ap.add_argument("--patience", type=int, default=30)       # PAPER FACT Table 3
    ap.add_argument("--seed", type=int, default=0)            # A8
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--name", default="pruned_4w4a")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--batchnorm", dest="batchnorm", action="store_true",
                    default=None,
                    help="force BatchNorm ON (default: configs/quant.yaml, "
                         "which sets it ON -- A15)")
    ap.add_argument("--no-batchnorm", dest="batchnorm", action="store_false",
                    help="force BatchNorm OFF -- the A15 ablation, per Table 1 "
                         "which lists no BN (does not train: DECISIONS.md "
                         "2026-10-16)")
    ap.add_argument("--fp32", action="store_true",
                    help="T1.2 baseline: no quantisation (unpruned cfg expected)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the full configuration, build nothing")
    ap.add_argument("--resume", default=None, metavar="LAST_PT",
                    help="continue an interrupted run from its last.pt; output "
                         "goes to that file's directory")
    return ap.parse_args(argv)


def resolve_config(args):
    """Collect every effective setting with its provenance, for the run record."""
    hyp = load_yaml(args.hyp) if os.path.isfile(args.hyp) else {}
    quant = load_yaml(args.quant) if os.path.isfile(args.quant) else {}
    data = load_yaml(args.data) if os.path.isfile(args.data) else {}
    return {
        "paper_specified": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "img_size": args.img, "optimizer": args.optimizer,
            "patience": args.patience,
            "_source": "PAPER FACT Table 3",
        },
        "our_assumptions": {
            "hyp_file": args.hyp, "hyp": hyp, "_id": "A5 / DECISION D4",
            "seed": args.seed, "_seed_id": "A8",
            "batchnorm": (quant.get("structure", {}).get("batchnorm", False)
                          if args.batchnorm is None else args.batchnorm),
            "_batchnorm_id": "A15 (HIGHEST RISK)",
            "train_letterbox": True, "_letterbox_id": "A4",
        },
        "quantisation": quant,
        "dataset": data,
        "architecture": {
            "source": "models/arch_spec.py (PAPER FACT Table 1)",
            "conv_layers": len(A.CONV_LAYERS),
            "total_weights": A.totals()["weights_total"],
            "decisions_applied": ["D2 maxpool stride 1", "P3 final conv 3x3",
                                  "P11 INT8 output activation"],
        },
        "reporting_split": "val",     # DECISION D3 / A2
    }


def main(argv=None):
    args = parse_args(argv)
    cfg = resolve_config(args)
    out_dir = (os.path.dirname(os.path.abspath(args.resume)) if args.resume
               else os.path.join(args.project, args.name))
    if not args.resume and os.path.isfile(os.path.join(out_dir, "last.pt")):
        raise SystemExit(
            "{} already holds a run. Continue it with --resume {}, or pick "
            "another --name.".format(out_dir, os.path.join(out_dir, "last.pt")))

    print("=" * 72)
    print("TRAINING CONFIGURATION ({})".format("T1.2 FP32" if args.fp32 else "T1.3 QAT"))
    print("=" * 72)
    print(json.dumps(cfg, indent=2, default=str))

    if args.dry_run:
        print("\n--dry-run: configuration resolved, nothing built or trained.")
        return 0

    # ---- heavy imports happen only past this point -----------------------
    import torch
    import numpy as np
    from models.quant_yolov3_tiny import build as build_quant
    from models.detect import Detect
    from utils.datasets import build_loader
    from utils.loss import HardTanhDecodeLoss
    from utils import metrics as M, plots

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hyp = cfg["our_assumptions"]["hyp"]
    root = cfg["dataset"].get("path")
    if not root or not os.path.isdir(root):
        raise SystemExit(
            "dataset root {!r} not present. Run tools/visdrone_to_yolo.py first "
            "(nothing is downloaded automatically).".format(root))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_quant(args.quant, batchnorm=cfg["our_assumptions"]["batchnorm"])
    model.to(device)
    head = Detect(nc=cfg["dataset"].get("nc", 1)).to(device)
    criterion = HardTanhDecodeLoss(A.ANCHORS_PX, A.DETECT_STRIDE,
                                   nc=cfg["dataset"].get("nc", 1), hyp=hyp).to(device)

    train_loader = build_loader(root, "train", args.img, args.batch_size,
                                augment=True, hyp=hyp, workers=args.workers)
    val_loader = build_loader(root, "val", args.img, args.batch_size,
                              augment=False, hyp=hyp, workers=args.workers)

    opt = torch.optim.SGD(model.parameters(), lr=float(hyp.get("lr0", 0.01)),
                          momentum=float(hyp.get("momentum", 0.937)),
                          weight_decay=float(hyp.get("weight_decay", 0.0005)),
                          nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(hyp.get("lr0", 0.01)),
        total_steps=max(1, args.epochs * len(train_loader)),
        final_div_factor=1.0 / max(1e-8, float(hyp.get("lrf", 0.01))))

    history, best, bad_epochs, start_epoch, finished = [], -1.0, 0, 0, False
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        # epochs / batch / img fix OneCycleLR's total_steps and the data; a
        # mismatch would silently turn the resumed run into a different one.
        old = ck["config"]["paper_specified"]
        new = cfg["paper_specified"]
        diff = {k: (old[k], new[k]) for k in ("epochs", "batch_size", "img_size")
                if old[k] != new[k]}
        if diff:
            raise SystemExit("--resume: settings differ from the original run "
                             "(old, new): {}".format(diff))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        history, best, bad_epochs = ck["history"], ck["best"], ck["bad_epochs"]
        start_epoch, finished = ck["epoch"] + 1, ck["finished"]
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        np.random.set_state(ck["rng"]["numpy"])
        if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng"]["cuda"]])
        print("resumed from {} at epoch {} (best mAP50 {:.4f}, {} epochs without "
              "improvement)".format(args.resume, start_epoch, best, bad_epochs))

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)

    na, no = head.na, head.no

    for epoch in range(start_epoch, args.epochs):
        if finished:
            break
        model.train()
        t0 = time.time()
        agg = {"box": 0.0, "obj": 0.0, "cls": 0.0}
        for imgs, targets, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
            out = model(imgs)
            n, _, ny, nx = out.shape
            raw = out.view(n, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            loss, parts = criterion(raw, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for k in agg:
                agg[k] += parts[k]
        nb = max(1, len(train_loader))
        rec = {"epoch": epoch, "time_s": round(time.time() - t0, 1)}
        rec.update({k: agg[k] / nb for k in agg})

        val = evaluate_loader(model, head, val_loader, device, na, no)
        rec.update({"precision": val["precision"], "recall": val["recall"],
                    "map50": val["map50"]})
        history.append(rec)
        print("epoch {:>3}/{}  box {:.4f}  obj {:.4f}  P {:.4f}  R {:.4f}  "
              "mAP50 {:.4f}  ({:.0f}s)".format(
                  epoch, args.epochs, rec["box"], rec["obj"], rec["precision"],
                  rec["recall"], rec["map50"], rec["time_s"]))

        if val["map50"] > best:
            best, bad_epochs = val["map50"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "map50": best, "config": cfg},
                       os.path.join(out_dir, "best.pt"))
        else:
            bad_epochs += 1
        early_stop = bad_epochs >= args.patience        # PAPER FACT Table 3
        finished = early_stop or epoch == args.epochs - 1

        # Everything --resume needs, written every epoch; tmp + rename so an
        # interruption mid-write cannot leave a truncated last.pt behind.
        last = os.path.join(out_dir, "last.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "best": best, "bad_epochs": bad_epochs, "history": history,
                    "finished": finished, "config": cfg,
                    "rng": {"torch": torch.get_rng_state(),
                            "numpy": np.random.get_state(),
                            "cuda": (torch.cuda.get_rng_state_all()
                                     if torch.cuda.is_available() else None)}},
                   last + ".tmp")
        os.replace(last + ".tmp", last)
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(history, fh, indent=2)

        if early_stop:
            print("early stop: no improvement for {} epochs".format(args.patience))
            break

    plots.training_curves(history, os.path.join(out_dir, "fig15a_curves.png"))
    final = evaluate_loader(model, head, val_loader, device, na, no)
    if "pr_curve" in final:
        plots.pr_curve(final["pr_curve"]["recall"], final["pr_curve"]["precision"],
                       os.path.join(out_dir, "fig15b_pr.png"))
    print("best mAP@0.5 = {:.4f}  (split=val, DECISION D3)".format(best))
    return 0


def evaluate_loader(model, head, loader, device, na, no):
    """Shared evaluation used by train.py and val.py -- one mAP implementation."""
    import torch
    import numpy as np
    from utils import metrics as M
    from utils.postprocess import (non_max_suppression, EVAL_CONF_THRES,
                                   EVAL_IOU_THRES, EVAL_MAX_DET)
    from utils.boxes import xywh2xyxy

    model.eval()
    dets_all, gts_all = [], []
    with torch.no_grad():
        for imgs, targets, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            out = model(imgs)
            n, _, ny, nx = out.shape
            raw = out.view(n, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            pred, _ = head([out])
            dets = non_max_suppression(pred.cpu().numpy(), EVAL_CONF_THRES,
                                       EVAL_IOU_THRES, EVAL_MAX_DET)
            size = imgs.shape[-1]
            for i in range(n):
                dets_all.append(dets[i])
                t = targets[targets[:, 0] == i].cpu().numpy()
                if len(t):
                    boxes = xywh2xyxy(t[:, 2:6] * size)
                    gts_all.append(np.concatenate([t[:, 1:2], boxes], axis=1))
                else:
                    gts_all.append(np.zeros((0, 5)))
    return M.evaluate(dets_all, gts_all)


if __name__ == "__main__":
    sys.exit(main())
