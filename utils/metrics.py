"""Detection metrics -- P, R, mAP@0.5. numpy only.

PAPER FACT Table 5 reports exactly: Precision, Recall, mAP@0.5, box loss,
object loss, classification loss (the last identically 0 for nc == 1).

DECISION D3 / A2: the reporting split is `val`, labelled explicitly; `train` is
also computed so that if our numbers diverge from the paper's we can tell whether
the paper was reporting train-set figures (the paper contradicts itself -- spec
P6: Table 5's caption says "Validation results", 4.1's body says "training set").

ONE shared implementation is used for T1.2, T1.3, T1.12 and T1.13 so the
comparisons between them are internally valid even if none matches the paper's
absolute numbers.
"""

import numpy as np

from utils.boxes import box_iou


def ap_from_pr(recall, precision):
    """Average precision by the all-point interpolation (COCO/VOC2010 style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # make precision monotonically decreasing
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_detections(det, gt, iou_thres=0.5):
    """Greedy one-to-one matching at a single IoU threshold.

    det: (M,6) x1y1x2y2 conf cls, assumed sorted by conf descending.
    gt:  (G,5) cls x1y1x2y2
    Returns a boolean (M,) array of true positives and the number of gt boxes.
    """
    det = np.asarray(det, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    tp = np.zeros(len(det), dtype=bool)
    if len(gt) == 0 or len(det) == 0:
        return tp, len(gt)
    used = np.zeros(len(gt), dtype=bool)
    ious = box_iou(det[:, :4], gt[:, 1:5])
    for i in range(len(det)):
        order = np.argsort(-ious[i])
        for j in order:
            if ious[i, j] < iou_thres:
                break
            if used[j]:
                continue
            if det[i, 5] == gt[j, 0]:
                tp[i] = True
                used[j] = True
            break
    return tp, len(gt)


def evaluate(detections, ground_truths, iou_thres=0.5):
    """Dataset-level P / R / mAP@0.5 for a single class.

    detections:     list of (M,6) arrays, one per image
    ground_truths:  list of (G,5) arrays, one per image
    Returns dict with precision, recall, map50, tp, fp, n_gt.
    """
    all_conf, all_tp, n_gt = [], [], 0
    for det, gt in zip(detections, ground_truths):
        det = np.asarray(det, dtype=np.float64)
        if len(det):
            det = det[np.argsort(-det[:, 4])]
        tp, g = match_detections(det, gt, iou_thres)
        n_gt += g
        if len(det):
            all_conf.append(det[:, 4])
            all_tp.append(tp)

    if not all_conf or n_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0,
                "tp": 0, "fp": 0, "n_gt": int(n_gt)}

    conf = np.concatenate(all_conf)
    tp = np.concatenate(all_tp)
    order = np.argsort(-conf)
    tp = tp[order]
    tpc = np.cumsum(tp)
    fpc = np.cumsum(~tp)
    recall = tpc / float(n_gt)
    precision = tpc / np.maximum(tpc + fpc, 1e-12)
    return {
        "precision": float(precision[-1]),
        "recall": float(recall[-1]),
        "map50": ap_from_pr(recall, precision),
        "tp": int(tpc[-1]),
        "fp": int(fpc[-1]),
        "n_gt": int(n_gt),
        "pr_curve": {"recall": recall.tolist(), "precision": precision.tolist()},
    }
