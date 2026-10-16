"""Detection loss consistent with the repository's non-standard decode.

DEPENDENCY: torch==1.13.1 (PAPER FACT Table 2).

WHY THIS IS YOLOv5's ComputeLoss WITH sigmoid REPLACED
-----------------------------------------------------
models/detect.py implements the REPOSITORY FACT decode:
    xy      = (y + 0.5 + grid) * stride
    wh      = (y + 1.0) ** 2  * anchor_grid
    obj,cls =  y / 2 + 0.5
DERIVED FACT: this is exactly YOLOv5's decode
    xy = (2*sigmoid(x) - 0.5 + grid) * stride,  wh = (2*sigmoid(x))**2 * anchor
with sigmoid(x) replaced by the affine map p(y) = y/2 + 0.5 of the HardTanh
output (2p - 0.5 = y + 0.5, (2p)**2 = (y + 1)**2). Together with paper 3.1.4's
"standard low-scratch hyper-parameters" (YOLOv5's hyp.scratch-low), the
author's training loss was almost certainly YOLOv5's ComputeLoss under the
same substitution. This module is that loss (DECISIONS.md 2026-10-16, A5):

  * box  -- 1 - CIoU of the decoded box against the target, on the target cell
            and its two nearest neighbours (the [-0.5, 1.5] xy range of the
            decode exists for exactly this 3-cell assignment);
  * obj  -- BCE on p(y_obj) against target = CIoU (detached, clamped >= 0),
            so the score ranks boxes by localisation quality;
  * cls  -- BCE on p(y_cls), one-hot target; identically 0 when nc == 1
            (PAPER FACT Table 5 reports cls loss == 0);
  * gains from configs/hyp.scratch-low.yaml, scaled as YOLOv5 train.py does
    for nl detection layers and image size; obj balance 4.0 for the single
    layer; the total is multiplied by the batch size.

The MSE-on-inverted-targets loss this replaces trained (run pruned_4w4a,
mAP@0.5 0.017) but localised poorly: mAP@0.1 0.238 on the same checkpoint.

Deviation from YOLOv5, OUR ASSUMPTION: p is squeezed affinely into
[PROB_EPS, 1 - PROB_EPS] before the log. With sigmoid, BCE's gradient w.r.t.
the logit is bounded by 1; with the affine p it is ~1/p and would blow up for a
positive cell predicted at p ~ 0. The squeeze bounds it at ~1/PROB_EPS without
a clamp, which would add a zero-gradient region on top of HardTanh's own.
"""

from __future__ import absolute_import

import math

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "utils.loss requires torch==1.13.1 (PAPER FACT Table 2); "
        "not installed by design. See requirements/README.md."
    ) from exc

PROB_EPS = 1e-3

# YOLOv5 ComputeLoss: self.balance = {3: [4.0, 1.0, 0.4]}.get(nl, [4.0, ...]);
# with a single detection layer the obj balance is 4.0.
OBJ_BALANCE_1_LAYER = 4.0


def to_prob(y):
    """HardTanh output y in [-1, 1] -> probability, as models/detect.py decodes
    it (y/2 + 0.5), squeezed into [PROB_EPS, 1 - PROB_EPS]."""
    return PROB_EPS + (1.0 - 2.0 * PROB_EPS) * (y / 2.0 + 0.5)


def init_objectness_prior(model, na, no, img_size=416, stride=32):
    """YOLOv5 Detect._initialize_biases for obj, under the HardTanh decode.

    YOLOv5 starts every objectness at p = 8 / (img_size / stride)**2 (about 8
    objects per image); here p = y/2 + 0.5, so y = 2p - 1. The final conv has
    no bias (models/quant_common.QuantConvBlock), so the prior goes into its
    BatchNorm shift; without BN (the A15 ablation) there is nowhere to put it
    and the model is left unchanged. Returns True if the prior was set.
    """
    bn = model.conv_modules()[-1].bn
    if bn is None:
        return False
    p = 8.0 / (float(img_size) / float(stride)) ** 2
    with torch.no_grad():
        bn.bias.view(na, no)[:, 4] = 2.0 * p - 1.0
    return True


def bce(p, t):
    """Binary cross-entropy on probabilities, mean-reduced."""
    return -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p)).mean()


def bbox_ciou(b1, b2, eps=1e-7):
    """Complete IoU of matched xywh boxes, (T, 4) x (T, 4) -> (T,)."""
    (x1, y1, w1, h1), (x2, y2, w2, h2) = b1.unbind(-1), b2.unbind(-1)
    b1x1, b1x2, b1y1, b1y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
    b2x1, b2x2, b2y1, b2y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    inter = ((torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0)
             * (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0))
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps))
                              - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


class HardTanhDecodeLoss(nn.Module):
    """YOLOv5 ComputeLoss under the HardTanh decode (see module docstring).

    Operates on the RAW network output (N, na, ny, nx, no) -- i.e. the second
    return value of models.detect.Detect -- never on the decoded predictions.
    """

    def __init__(self, anchors_px, stride=32, nc=1, hyp=None, img_size=416):
        super(HardTanhDecodeLoss, self).__init__()
        hyp = hyp or {}
        self.nc = int(nc)
        self.stride = float(stride)
        self.na = len(anchors_px)
        # anchors in grid units, as the decode holds them (models/detect.py)
        self.register_buffer(
            "anchors",
            torch.tensor([[w, h] for (w, h) in anchors_px],
                         dtype=torch.float32) / self.stride)
        nl = 1
        # YOLOv5 train.py: hyp['box'] *= 3/nl; hyp['cls'] *= nc/80 * 3/nl;
        #                  hyp['obj'] *= (imgsz/640)**2 * 3/nl
        self.gain_box = float(hyp.get("box", 0.05)) * 3.0 / nl
        self.gain_cls = float(hyp.get("cls", 0.5)) * self.nc / 80.0 * 3.0 / nl
        self.gain_obj = (float(hyp.get("obj", 1.0))
                         * (float(img_size) / 640.0) ** 2 * 3.0 / nl)
        self.anchor_t = float(hyp.get("anchor_t", 4.0))

    # -- target assignment (YOLOv5 ComputeLoss.build_targets, nl == 1) ------

    def build_targets(self, raw, targets):
        """raw: (N, na, ny, nx, no). targets: (T, 6) = img, cls, cx, cy, w, h
        with box values NORMALISED to [0,1].

        A ground-truth box goes to every anchor whose width AND height ratio is
        within `anchor_t`, at the cell containing its centre and at the two
        neighbouring cells nearest to that centre.

        Returns (b, a, gj, gi, tbox, anchors, cls): tbox is xywh in grid units
        with xy relative to the assigned cell, i.e. in [-0.5, 1.5].
        """
        n, na, ny, nx, no = raw.shape
        dev = raw.device
        nt = targets.shape[0]
        gain = torch.ones(7, device=dev)
        gain[2:6] = torch.tensor([nx, ny, nx, ny], device=dev, dtype=gain.dtype)
        ai = torch.arange(na, device=dev).float().view(na, 1).repeat(1, nt)
        t = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2) * gain

        g = 0.5
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
                           device=dev).float() * g
        if nt:
            r = t[..., 4:6] / self.anchors[:, None]
            t = t[torch.max(r, 1.0 / r).max(2)[0] < self.anchor_t]
            gxy = t[:, 2:4]
            gxi = gain[[2, 3]] - gxy
            j, k = ((gxy % 1 < g) & (gxy > 1)).T
            l, m = ((gxi % 1 < g) & (gxi > 1)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = t[0]
            offsets = 0

        bc, gxy, gwh, a = t[:, 0:2], t[:, 2:4], t[:, 4:6], t[:, 6]
        a = a.long()
        b, c = bc.long().T
        gij = (gxy - offsets).long()
        gi = gij[:, 0].clamp(0, nx - 1)
        gj = gij[:, 1].clamp(0, ny - 1)
        tbox = torch.cat((gxy - gij, gwh), 1)
        return b, a, gj, gi, tbox, self.anchors[a], c

    # -- forward ------------------------------------------------------------

    def forward(self, raw, targets):
        """Returns (total, {'box':..,'obj':..,'cls':..}).

        The parts are gain-scaled per-batch means, as YOLOv5 logs them, so they
        are comparable with PAPER FACT Table 5's loss values; the total is
        their sum times the batch size, as YOLOv5 back-propagates it.
        """
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        n = raw.shape[0]
        b, a, gj, gi, tbox, anch, tcls = self.build_targets(raw, targets)

        tobj = torch.zeros(raw.shape[:4], device=raw.device, dtype=raw.dtype)
        l_box = raw.new_zeros(())
        l_cls = raw.new_zeros(())

        if b.numel():
            ps = raw[b, a, gj, gi]                          # (T, no)
            pxy = ps[:, 0:2] + 0.5                          # = 2*p - 0.5
            pwh = (ps[:, 2:4] + 1.0) ** 2 * anch            # = (2*p)**2 * a
            iou = bbox_ciou(torch.cat((pxy, pwh), 1), tbox)
            l_box = (1.0 - iou).mean()
            tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)
            if self.nc > 1:
                onehot = torch.zeros_like(ps[:, 5:])
                onehot[torch.arange(b.numel(), device=raw.device), tcls] = 1.0
                l_cls = bce(to_prob(ps[:, 5:]), onehot)
            # nc == 1: classification loss is identically zero -- PAPER FACT
            # Table 5 reports cls loss == 0.

        l_obj = bce(to_prob(raw[..., 4]), tobj) * OBJ_BALANCE_1_LAYER
        l_box = l_box * self.gain_box
        l_obj = l_obj * self.gain_obj
        l_cls = l_cls * self.gain_cls
        total = (l_box + l_obj + l_cls) * n
        return total, {"box": float(l_box), "obj": float(l_obj),
                       "cls": float(l_cls)}
