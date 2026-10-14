"""Detection head -- decode only, runs on the PS / host, never on the FPGA.

CLASSIFICATION: REPOSITORY FACT. The paper describes this layer only as
"basically only contains scaling operations to convert between bounding box
coordinate spaces" (3.1) and detaches it at export (3.2). The actual formulas
come from the author repository, client/models.py:

    xy      = (y[..., 0:2] + 0.5 + grid) * stride
    wh      = (y[..., 2:4] + 1.0) ** 2  * anchor_grid
    obj,cls =  y[..., 4:] / 2 + 0.5

These are NOT stock YOLOv3 (which uses sigmoid(xy)+grid, exp(wh)*anchor,
sigmoid(obj)). They are affine/quadratic maps that bring a QuantHardTanh output
bounded in [-1, 1] into the correct domains.

CRITICAL: this parameterisation must be used at TRAINING time as well, or the
loss optimises a different function than inference decodes. utils/loss.py
inverts exactly these formulas.

Anchors (REPOSITORY FACT infer_client_v3.py:25): (10,14), (23,27), (37,58) at
stride 32 -- YOLOv3-Tiny's small-object set, reused on the 13x13 branch.
"""

from __future__ import absolute_import

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "models.detect requires torch==1.13.1; not installed by design. "
        "A numpy-only equivalent lives in utils/postprocess.py."
    ) from exc


class Detect(nn.Module):
    """Single-scale decode head (nl == 1), matching the repository reference."""

    def __init__(self, nc=A.NUM_CLASSES, anchors=A.ANCHORS_PX,
                 stride=A.DETECT_STRIDE):
        super(Detect, self).__init__()
        self.nc = int(nc)
        self.no = self.nc + 5
        self.nl = 1
        self.na = len(anchors)
        self.stride = [float(stride)]
        # REPOSITORY FACT: the repo divides pixel anchors by the stride before
        # passing them in, so anchors are held in grid units.
        flat = []
        for (w, h) in anchors:
            flat += [w / float(stride), h / float(stride)]
        self.register_buffer(
            "anchors", torch.tensor(flat).float().view(self.nl, -1, 2))
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl

    def _make_grid(self, nx, ny, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid(torch.arange(ny, device=d),
                                torch.arange(nx, device=d), indexing="ij")
        grid = torch.stack((xv, yv), 2).expand(
            (1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view(
            (1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

    def forward(self, x):
        """x: list of one tensor, (N, no*na, ny, nx). Returns (pred, raw)."""
        z = []
        for i in range(self.nl):
            t = x[i]
            bs, _, ny, nx = t.shape
            t = t.view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()
            x[i] = t
            if self.grid[i].shape[2:4] != t.shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            y = t.clone()
            # ---- REPOSITORY FACT decode (client/models.py) -----------------
            y[..., 0:2] = (y[..., 0:2] + 0.5 + self.grid[i]) * self.stride[i]
            y[..., 2:4] = (y[..., 2:4] + 1.0) ** 2 * self.anchor_grid[i]
            y[..., 4:] = y[..., 4:] / 2 + 0.5
            # ----------------------------------------------------------------
            z.append(y.view(bs, -1, self.no))
        return torch.cat(z, 1), x
