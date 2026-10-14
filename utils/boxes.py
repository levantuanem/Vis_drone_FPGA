"""Box geometry and NMS -- numpy only.

The author repository imports these from a `utils` module that is MISSING
upstream (paper_vs_repository.md section 5), so there was nothing to copy even
in principle. These are our implementations of the standard operations.
"""

import numpy as np


def xywh2xyxy(x):
    """centre-x, centre-y, w, h  ->  x1, y1, x2, y2."""
    y = np.asarray(x, dtype=np.float64).copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0
    y[..., 2] = x[..., 0] + x[..., 2] / 2.0
    y[..., 3] = x[..., 1] + x[..., 3] / 2.0
    return y


def xyxy2xywh(x):
    y = np.asarray(x, dtype=np.float64).copy()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2.0
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2.0
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def clip_boxes(boxes, shape):
    """Clip x1y1x2y2 boxes to an (h, w) image in place-safe fashion."""
    h, w = shape[0], shape[1]
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    boxes[..., 0] = boxes[..., 0].clip(0, w)
    boxes[..., 1] = boxes[..., 1].clip(0, h)
    boxes[..., 2] = boxes[..., 2].clip(0, w)
    boxes[..., 3] = boxes[..., 3].clip(0, h)
    return boxes


def box_iou(a, b):
    """IoU matrix between (N,4) and (M,4) x1y1x2y2 arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def nms(boxes, scores, iou_thres):
    """Greedy NMS. Returns kept indices, highest score first."""
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = box_iou(boxes[i][None, :], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thres]
    return keep


def scale_coords(from_shape, coords, to_shape, letterboxed=True):
    """Map boxes from the network input frame back to the original image.

    `letterboxed=True` undoes the aspect-preserving pad applied by
    utils.preprocess.letterbox (REPOSITORY FACT infer_client_v3.py:241).
    """
    coords = np.asarray(coords, dtype=np.float64).copy()
    fh, fw = from_shape[0], from_shape[1]
    th, tw = to_shape[0], to_shape[1]
    if letterboxed:
        gain = min(fh / float(th), fw / float(tw))
        pad_x = (fw - tw * gain) / 2.0
        pad_y = (fh - th * gain) / 2.0
        coords[..., [0, 2]] -= pad_x
        coords[..., [1, 3]] -= pad_y
        coords[..., :4] /= gain
    else:
        coords[..., [0, 2]] *= tw / float(fw)
        coords[..., [1, 3]] *= th / float(fh)
    return clip_boxes(coords, (th, tw))
