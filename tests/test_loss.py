"""utils/loss.py must optimise the parameterisation models/detect.py decodes."""

import pytest

torch = pytest.importorskip("torch")

from models import arch_spec as A                 # noqa: E402
from models.detect import Detect                  # noqa: E402
from utils.loss import HardTanhDecodeLoss, bbox_ciou   # noqa: E402

NA, NO, G = 3, 6, 13


def _loss():
    return HardTanhDecodeLoss(A.ANCHORS_PX, A.DETECT_STRIDE, nc=1, img_size=416)


def _target(cx, cy, w, h, img=0):
    """One box, centre/size in grid cells -> normalised (1, 6) target row."""
    return torch.tensor([[img, 0, cx / G, cy / G, w / G, h / G]])


def test_three_cell_assignment():
    # centre at (6.3, 6.3): own cell plus the left and upper neighbours
    b, a, gj, gi, tbox, _, _ = _loss().build_targets(
        torch.zeros(1, NA, G, G, NO), _target(6.3, 6.3, 0.6, 0.6))
    cells = set(zip(gi.tolist(), gj.tolist()))
    assert cells == {(6, 6), (5, 6), (6, 5)}
    # every assigned xy offset lies inside the decode's [-0.5, 1.5] range
    assert (tbox[:, :2] >= -0.5).all() and (tbox[:, :2] <= 1.5).all()


def test_anchor_ratio_filter():
    # 0.25 x 0.25 cells = 8 px: within 4x of anchors 0 and 1 only (37/8 > 4)
    _, a, _, _, _, _, _ = _loss().build_targets(
        torch.zeros(1, NA, G, G, NO), _target(6.3, 6.3, 0.25, 0.25))
    assert set(a.tolist()) == {0, 1}


def test_loss_parameterisation_matches_detect_decode():
    """Box the loss scores == box Detect returns at inference, same raw."""
    torch.manual_seed(0)
    raw = torch.rand(1, NA, G, G, NO) * 1.6 - 0.8
    crit = _loss()
    b, a, gj, gi, tbox, anch, _ = crit.build_targets(raw, _target(6.3, 4.7, 0.6, 0.9))
    ps = raw[b, a, gj, gi]
    pxy = (ps[:, 0:2] + 0.5 + torch.stack([gi, gj], 1).float()) * A.DETECT_STRIDE
    pwh = (ps[:, 2:4] + 1.0) ** 2 * anch * A.DETECT_STRIDE

    head = Detect(nc=1)
    nchw = raw.permute(0, 1, 4, 2, 3).reshape(1, NA * NO, G, G)
    dec, _ = head([nchw.clone()])
    dec = dec.view(1, NA, G, G, NO)[0, a, gj, gi]
    assert torch.allclose(dec[:, 0:2], pxy, atol=1e-4)
    assert torch.allclose(dec[:, 2:4], pwh, atol=1e-4)


def _perfect_raw(crit, targets):
    """Raw output whose decode reproduces every assigned target exactly."""
    raw = torch.full((1, NA, G, G, NO), -1.0)
    b, a, gj, gi, tbox, anch, _ = crit.build_targets(raw, targets)
    raw[b, a, gj, gi, 0:2] = tbox[:, :2] - 0.5
    raw[b, a, gj, gi, 2:4] = (tbox[:, 2:4] / anch).sqrt() - 1.0
    raw[b, a, gj, gi, 4] = 1.0
    return raw


def test_perfect_prediction_has_near_zero_loss():
    crit = _loss()
    t = _target(6.3, 6.3, 0.6, 0.6)
    _, parts = crit(_perfect_raw(crit, t), t)
    assert parts["box"] < 1e-5
    assert parts["obj"] < 0.01         # bounded below only by PROB_EPS
    assert parts["cls"] == 0.0         # nc == 1, PAPER FACT Table 5


def test_objectness_target_is_iou():
    crit = _loss()
    t = _target(6.3, 6.3, 0.6, 0.6)
    raw = _perfect_raw(crit, t)
    b, a, gj, gi, _, _, _ = crit.build_targets(raw, t)
    raw[b, a, gj, gi, 2:4] -= 0.2      # shrink the boxes: IoU < 1
    _, worse = crit(raw, t)
    assert worse["box"] > 1e-3
    raw[b, a, gj, gi, 4] = 0.0          # p = 0.5, closer to the lowered target
    _, lower_obj = crit(raw, t)
    raw[b, a, gj, gi, 4] = 1.0
    _, full_obj = crit(raw, t)
    assert lower_obj["obj"] < full_obj["obj"]


def test_gradients_finite_with_saturated_outputs():
    crit = _loss()
    raw = torch.full((2, NA, G, G, NO), -1.0, requires_grad=True)
    t = torch.cat([_target(6.3, 6.3, 0.6, 0.6), _target(2.8, 9.1, 0.3, 0.4, img=1)])
    total, _ = crit(raw, t)
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0


def test_empty_targets():
    crit = _loss()
    raw = torch.zeros(2, NA, G, G, NO, requires_grad=True)
    total, parts = crit(raw, torch.zeros(0, 6))
    total.backward()
    assert parts["box"] == 0.0 and parts["obj"] > 0


def test_total_scales_with_batch_size():
    crit = _loss()
    t = _target(6.3, 6.3, 0.6, 0.6)
    raw1 = torch.zeros(1, NA, G, G, NO)
    total1, _ = crit(raw1, t)
    total2, _ = crit(raw1.repeat(2, 1, 1, 1, 1),
                     torch.cat([t, _target(6.3, 6.3, 0.6, 0.6, img=1)]))
    assert torch.allclose(total2, 2 * total1, rtol=1e-5)


def test_ciou_identity_and_disjoint():
    box = torch.tensor([[1.0, 1.0, 0.5, 0.5]])
    assert torch.allclose(bbox_ciou(box, box), torch.ones(1), atol=1e-5)
    far = torch.tensor([[5.0, 5.0, 0.5, 0.5]])
    assert bbox_ciou(box, far).item() < 0


def test_objectness_prior_initialisation():
    pytest.importorskip("brevitas")
    from models.quant_yolov3_tiny import build
    from utils.loss import init_objectness_prior
    m = build("configs/quant.yaml", batchnorm=True)
    assert init_objectness_prior(m, NA, NO, 416, A.DETECT_STRIDE)
    bias = m.conv_modules()[-1].bn.bias.view(NA, NO)
    p = 8.0 / 13 ** 2
    assert torch.allclose(bias[:, 4], torch.full((NA,), 2 * p - 1))
    assert (bias[:, :4] == 0).all() and (bias[:, 5] == 0).all()
    assert not init_objectness_prior(build("configs/quant.yaml", batchnorm=False),
                                     NA, NO)
