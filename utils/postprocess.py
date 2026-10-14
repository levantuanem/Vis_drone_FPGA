"""Post-processing -- dequantise, decode, NMS, rescale. numpy only.

Pipeline (all REPOSITORY FACT except where noted):
  1. accelerator output: INT8, NHWC (1,13,13,18)          driver.py io_shape_dict
  2. dequantise: multiply by 1/127                        scale.npy = 0.00787402
  3. NHWC -> NCHW                                         infer_client_v3.py:338
  4. decode with the HardTanh-adapted formulas            client/models.py
  5. NMS: conf 0.10, iou 0.10, max_det 20                 infer_client_v3.py:342
  6. scale boxes back to the source frame                 utils.scale_coords

This is the numpy mirror of models/detect.py. Keeping both lets T1.12 compare the
torch path and the deployment path against each other.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A        # noqa: E402  stdlib-only
from utils.boxes import nms, scale_coords, xywh2xyxy   # noqa: E402

# REPOSITORY FACT infer_client_v3.py:342
DEFAULT_CONF_THRES = 0.10
DEFAULT_IOU_THRES = 0.10
DEFAULT_MAX_DET = 20


def dequantise(out_int8, scale=A.OUTPUT_DEQUANT_SCALE):
    """REPOSITORY FACT: scale.npy == 0.00787402 == 1/127.

    This is the trailing `Mul` node that PAPER FACT 3.4.4 leaves on the PS.
    """
    return np.asarray(out_int8, dtype=np.float64) * float(scale)


def nhwc_to_nchw(x):
    """PAPER FACT 3.4.3: YOLO is NCHW, FINN is NHWC; the driver transposes."""
    return np.transpose(np.asarray(x), (0, 3, 1, 2))


def make_grid(nx, ny, na):
    yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    grid = np.stack((xv, yv), axis=2)[None, None, ...]
    return np.broadcast_to(grid, (1, na, ny, nx, 2)).astype(np.float64)


def decode(raw_nchw, nc=A.NUM_CLASSES, anchors=A.ANCHORS_PX,
           stride=A.DETECT_STRIDE):
    """REPOSITORY FACT decode (client/models.py). NOT stock YOLOv3.

        xy      = (y + 0.5 + grid) * stride
        wh      = (y + 1.0) ** 2  * anchor_grid
        obj,cls =  y / 2 + 0.5

    Input:  (N, na*(nc+5), ny, nx) float.  Output: (N, na*ny*nx, nc+5).
    """
    x = np.asarray(raw_nchw, dtype=np.float64)
    na = len(anchors)
    no = nc + 5
    bs, ch, ny, nx = x.shape
    if ch != na * no:
        raise ValueError("expected {} channels, got {}".format(na * no, ch))
    y = x.reshape(bs, na, no, ny, nx).transpose(0, 1, 3, 4, 2).copy()

    grid = make_grid(nx, ny, na)
    anchor_grid = np.array(
        [[w, h] for (w, h) in anchors], dtype=np.float64).reshape(1, na, 1, 1, 2)
    anchor_grid = np.broadcast_to(anchor_grid, (1, na, ny, nx, 2))

    y[..., 0:2] = (y[..., 0:2] + 0.5 + grid) * float(stride)
    y[..., 2:4] = (y[..., 2:4] + 1.0) ** 2 * anchor_grid
    y[..., 4:] = y[..., 4:] / 2.0 + 0.5
    return y.reshape(bs, na * ny * nx, no)


def non_max_suppression(pred, conf_thres=DEFAULT_CONF_THRES,
                        iou_thres=DEFAULT_IOU_THRES, max_det=DEFAULT_MAX_DET):
    """Per-image NMS. pred: (N, A, nc+5) with xywh in pixels.

    Returns a list of (M, 6) arrays: x1 y1 x2 y2 conf cls.
    """
    pred = np.asarray(pred, dtype=np.float64)
    out = []
    for p in pred:
        obj = p[:, 4]
        cls_scores = p[:, 5:]
        if cls_scores.shape[1] == 0:
            conf = obj
            cls_idx = np.zeros(len(p), dtype=np.int64)
        else:
            cls_idx = cls_scores.argmax(axis=1)
            conf = obj * cls_scores[np.arange(len(p)), cls_idx]
        mask = conf > conf_thres
        if not mask.any():
            out.append(np.zeros((0, 6)))
            continue
        boxes = xywh2xyxy(p[mask, :4])
        c = conf[mask]
        k = cls_idx[mask]
        keep = nms(boxes, c, iou_thres)[:max_det]
        det = np.concatenate(
            [boxes[keep], c[keep, None], k[keep, None].astype(np.float64)], axis=1)
        out.append(det)
    return out


def postprocess(out_int8_nhwc, original_shape, input_size=416,
                conf_thres=DEFAULT_CONF_THRES, iou_thres=DEFAULT_IOU_THRES,
                max_det=DEFAULT_MAX_DET, nc=A.NUM_CLASSES,
                anchors=A.ANCHORS_PX, stride=A.DETECT_STRIDE,
                scale=A.OUTPUT_DEQUANT_SCALE, letterboxed=True):
    """Full accelerator-output -> detections path, in original image coordinates."""
    deq = dequantise(out_int8_nhwc, scale)
    nchw = nhwc_to_nchw(deq)
    pred = decode(nchw, nc=nc, anchors=anchors, stride=stride)
    dets = non_max_suppression(pred, conf_thres, iou_thres, max_det)
    for d in dets:
        if len(d):
            d[:, :4] = scale_coords((input_size, input_size), d[:, :4],
                                    original_shape, letterboxed=letterboxed)
    return dets
