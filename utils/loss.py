"""Detection loss consistent with the repository's non-standard decode.

DEPENDENCY NOT INSTALLED: torch==1.13.1 (PAPER FACT Table 2).

WHY THIS CANNOT BE STOCK YOLOv5 LOSS
------------------------------------
models/detect.py implements the REPOSITORY FACT decode:
    xy      = (y + 0.5 + grid) * stride
    wh      = (y + 1.0) ** 2  * anchor_grid
    obj,cls =  y / 2 + 0.5
Stock YOLOv5 loss assumes sigmoid(xy)*2-0.5 and (sigmoid(wh)*2)**2*anchor, and
BCEWithLogits on objectness. Applying it here would optimise a different
parameterisation than inference decodes, so the targets below are built by
INVERTING the formulas above:
    t_xy  = gt_xy / stride - grid - 0.5
    t_wh  = sqrt(gt_wh / anchor) - 1.0
    t_obj = 2 * target_obj - 1          (so decode maps it back to [0,1])
Loss gains come from configs/hyp.scratch-low.yaml (OUR ASSUMPTION A5).

Because the final activation is QuantHardTanh, raw outputs are bounded in
[-1, 1]; the inverted targets are therefore clamped to that range.
"""

from __future__ import absolute_import

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "utils.loss requires torch==1.13.1 (PAPER FACT Table 2); "
        "not installed by design. See requirements/README.md."
    ) from exc


class HardTanhDecodeLoss(nn.Module):
    """Loss matching the HardTanh-bounded, non-standard decode.

    Operates on the RAW network output (N, na, ny, nx, no) -- i.e. the second
    return value of models.detect.Detect -- never on the decoded predictions.
    """

    def __init__(self, anchors_px, stride=32, nc=1, hyp=None):
        super(HardTanhDecodeLoss, self).__init__()
        hyp = hyp or {}
        self.nc = int(nc)
        self.stride = float(stride)
        self.na = len(anchors_px)
        self.register_buffer(
            "anchors_px",
            torch.tensor([[w, h] for (w, h) in anchors_px], dtype=torch.float32))
        self.gain_box = float(hyp.get("box", 0.05))
        self.gain_obj = float(hyp.get("obj", 1.0))
        self.gain_cls = float(hyp.get("cls", 0.5))
        self.anchor_t = float(hyp.get("anchor_t", 4.0))
        self.mse = nn.MSELoss(reduction="mean")

    # -- target inversion ---------------------------------------------------

    def invert_xy(self, gt_xy, grid_xy):
        """gt_xy in pixels -> raw target in [-1, 1]."""
        return (gt_xy / self.stride - grid_xy - 0.5).clamp(-1.0, 1.0)

    def invert_wh(self, gt_wh, anchor_wh):
        """gt_wh in pixels -> raw target in [-1, 1] (inverts the square)."""
        ratio = (gt_wh / anchor_wh).clamp(min=1e-9)
        return (ratio.sqrt() - 1.0).clamp(-1.0, 1.0)

    @staticmethod
    def invert_obj(target01):
        """[0,1] objectness -> raw target in [-1, 1] (inverts y/2 + 0.5)."""
        return (2.0 * target01 - 1.0).clamp(-1.0, 1.0)

    # -- target assignment --------------------------------------------------

    def build_targets(self, raw, targets):
        """raw: (N, na, ny, nx, no). targets: (T, 6) = img, cls, cx, cy, w, h
        with box values NORMALISED to [0,1].

        Returns index tensors and the inverted regression/objectness targets.
        Assignment rule: a ground-truth box is assigned to every anchor whose
        width AND height ratio is within `anchor_t` (the Ultralytics rule), at
        the grid cell containing its centre.
        """
        n, na, ny, nx, no = raw.shape
        dev = raw.device
        if targets.numel() == 0:
            empty = torch.zeros(0, dtype=torch.long, device=dev)
            return (empty, empty, empty, empty,
                    torch.zeros((0, 2), device=dev),
                    torch.zeros((0, 2), device=dev),
                    torch.zeros(0, dtype=torch.long, device=dev))

        img_i = targets[:, 0].long()
        cls_i = targets[:, 1].long()
        gxy_px = targets[:, 2:4] * torch.tensor([nx, ny], device=dev) * self.stride
        gwh_px = targets[:, 4:6] * torch.tensor([nx, ny], device=dev) * self.stride

        anchors = self.anchors_px.to(dev)
        ratio = gwh_px[:, None, :] / anchors[None, :, :]
        fit = torch.max(ratio, 1.0 / ratio).max(dim=2)[0] < self.anchor_t
        t_idx, a_idx = torch.nonzero(fit, as_tuple=True)
        if t_idx.numel() == 0:                      # fall back to best anchor
            a_idx = torch.max(ratio, 1.0 / ratio).max(dim=2)[0].argmin(dim=1)
            t_idx = torch.arange(targets.shape[0], device=dev)

        gxy = gxy_px[t_idx]
        gwh = gwh_px[t_idx]
        gi = (gxy[:, 0] / self.stride).long().clamp(0, nx - 1)
        gj = (gxy[:, 1] / self.stride).long().clamp(0, ny - 1)
        grid_xy = torch.stack([gi.float(), gj.float()], dim=1)

        t_xy = self.invert_xy(gxy, grid_xy)
        t_wh = self.invert_wh(gwh, anchors[a_idx])
        return (img_i[t_idx], a_idx, gj, gi, t_xy, t_wh, cls_i[t_idx])

    # -- forward ------------------------------------------------------------

    def forward(self, raw, targets):
        """Returns (total, {'box':..,'obj':..,'cls':..})."""
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        n, na, ny, nx, no = raw.shape
        dev = raw.device
        bi, ai, gj, gi, t_xy, t_wh, t_cls = self.build_targets(raw, targets)

        obj_target = torch.zeros((n, na, ny, nx), device=dev)
        l_box = raw.new_zeros(())
        l_cls = raw.new_zeros(())

        if bi.numel():
            pred = raw[bi, ai, gj, gi]            # (T, no)
            l_box = self.mse(pred[:, 0:2], t_xy) + self.mse(pred[:, 2:4], t_wh)
            obj_target[bi, ai, gj, gi] = 1.0
            if self.nc > 1:
                onehot = torch.zeros((bi.numel(), self.nc), device=dev)
                onehot[torch.arange(bi.numel(), device=dev), t_cls] = 1.0
                l_cls = self.mse(pred[:, 5:], self.invert_obj(onehot))
            # nc == 1: classification loss is identically zero -- PAPER FACT
            # Table 5 reports cls loss == 0.

        l_obj = self.mse(raw[..., 4], self.invert_obj(obj_target))
        total = (self.gain_box * l_box + self.gain_obj * l_obj
                 + self.gain_cls * l_cls)
        return total, {"box": float(l_box), "obj": float(l_obj),
                       "cls": float(l_cls)}
