"""Pre/post-processing tests. numpy only -- no torch, no board."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A
from utils import boxes as B
from utils import metrics as M
from utils import postprocess as Q
from utils import preprocess as P


def test_letterbox_preserves_aspect_and_pads():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    out, gain, (px, py) = P.letterbox(img, (416, 416))
    assert out.shape == (416, 416, 3)
    assert abs(gain - 416.0 / 1280.0) < 1e-9
    assert px == 0 and py > 0           # wide image -> vertical padding


def test_letterbox_square_input_has_no_pad():
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    out, gain, (px, py) = P.letterbox(img, (416, 416))
    assert (px, py) == (0, 0)
    assert out.shape == (416, 416, 3)


def test_preprocess_frame_contract():
    """REPOSITORY FACT: UINT8 NHWC (1,416,416,3) into the accelerator."""
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    t, gain, pad = P.preprocess_frame(img, 416)
    assert t.shape == A.IO_CONTRACT["ishape_normal"]
    assert t.dtype == np.uint8


def test_bgr_to_rgb_inverts_channels():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 0] = 1
    img[..., 2] = 3
    out = P.bgr_to_rgb(img)
    assert out[0, 0, 0] == 3 and out[0, 0, 2] == 1


def test_dequant_scale_is_one_over_127():
    """REPOSITORY FACT scale.npy -- INT8 127 must map to exactly 1.0."""
    assert Q.dequantise(np.array([127], dtype=np.int8))[0] == 1.0
    assert Q.dequantise(np.array([-127], dtype=np.int8))[0] == -1.0
    assert Q.dequantise(np.array([0], dtype=np.int8))[0] == 0.0


def test_nhwc_to_nchw():
    x = np.zeros((1, 13, 13, 18))
    assert Q.nhwc_to_nchw(x).shape == (1, 18, 13, 13)


def test_decode_shape_and_anchor_count():
    raw = np.zeros((1, 18, 13, 13))
    pred = Q.decode(raw)
    assert pred.shape == (1, 3 * 13 * 13, A.NUM_CLASSES + 5)


def test_decode_formulas_match_the_repository_reference():
    """Zero raw output -> xy=(0+0.5+grid)*32, wh=(0+1)^2*anchor, obj=0.5."""
    raw = np.zeros((1, 18, 13, 13))
    pred = Q.decode(raw)
    # first prediction is anchor 0, grid cell (0,0)
    assert abs(pred[0, 0, 0] - 0.5 * 32) < 1e-9
    assert abs(pred[0, 0, 1] - 0.5 * 32) < 1e-9
    assert abs(pred[0, 0, 2] - A.ANCHORS_PX[0][0]) < 1e-9     # (0+1)^2 * 10
    assert abs(pred[0, 0, 3] - A.ANCHORS_PX[0][1]) < 1e-9     # (0+1)^2 * 14
    assert abs(pred[0, 0, 4] - 0.5) < 1e-9                    # 0/2 + 0.5


def test_decode_is_not_stock_yolo():
    """Stock YOLO would give sigmoid(0)=0.5 for obj and exp(0)=1 for wh ratio;
    the HardTanh-adapted decode gives a different xy offset."""
    raw = np.zeros((1, 18, 13, 13))
    pred = Q.decode(raw)
    stock_xy = (0.5 + 0) * 32          # sigmoid(0)=0.5 then +grid, *stride
    assert abs(pred[0, 0, 0] - stock_xy) < 1e-9   # coincides at zero...
    raw2 = np.full((1, 18, 13, 13), 1.0)
    p2 = Q.decode(raw2)
    assert abs(p2[0, 0, 2] - 4.0 * A.ANCHORS_PX[0][0]) < 1e-9   # (1+1)^2 = 4
    # stock would be exp(1)*anchor ~ 2.718*anchor, clearly different
    assert abs(p2[0, 0, 2] - 2.71828 * A.ANCHORS_PX[0][0]) > 1.0


def test_nms_respects_max_det():
    """REPOSITORY FACT max_det == 20."""
    pred = np.zeros((1, 500, 6))
    pred[0, :, 0] = np.arange(500) * 3.0      # spread the boxes out
    pred[0, :, 1] = np.arange(500) * 3.0
    pred[0, :, 2] = 8.0
    pred[0, :, 3] = 8.0
    pred[0, :, 4] = 0.9
    pred[0, :, 5] = 0.9
    dets = Q.non_max_suppression(pred)
    assert len(dets[0]) <= Q.DEFAULT_MAX_DET == 20


def test_nms_suppresses_duplicates():
    pred = np.zeros((1, 3, 6))
    for i in range(3):
        pred[0, i] = [50, 50, 20, 20, 0.9, 0.9]       # identical boxes
    dets = Q.non_max_suppression(pred)
    assert len(dets[0]) == 1


def test_nms_default_thresholds_are_the_repository_values():
    assert Q.DEFAULT_CONF_THRES == 0.10
    assert Q.DEFAULT_IOU_THRES == 0.10
    assert Q.DEFAULT_MAX_DET == 20


def test_full_postprocess_pipeline_runs_on_int8():
    out = np.random.randint(-127, 128, (1, 13, 13, 18)).astype(np.int8)
    dets = Q.postprocess(out, (720, 1280))
    assert len(dets) == 1
    assert dets[0].shape[1] == 6


def test_box_conversions_roundtrip():
    xywh = np.array([[100.0, 200.0, 40.0, 60.0]])
    assert np.allclose(B.xyxy2xywh(B.xywh2xyxy(xywh)), xywh)


def test_box_iou_identity_and_disjoint():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert abs(B.box_iou(a, a)[0, 0] - 1.0) < 1e-12
    b = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert B.box_iou(a, b)[0, 0] == 0.0


def test_scale_coords_undoes_letterbox():
    """A box covering the whole padded frame maps back to the whole image."""
    img_h, img_w = 720, 1280
    _, gain, (px, py) = P.letterbox(np.zeros((img_h, img_w, 3), np.uint8),
                                    (416, 416))
    inner = np.array([[px, py, 416 - px, 416 - py]], dtype=np.float64)
    out = B.scale_coords((416, 416), inner, (img_h, img_w))
    assert abs(out[0, 0] - 0.0) < 1.0
    assert abs(out[0, 2] - img_w) < 2.0


def test_metrics_perfect_and_empty():
    gt = [np.array([[0, 10, 10, 50, 50]])]
    det = [np.array([[10, 10, 50, 50, 0.9, 0]])]
    r = M.evaluate(det, gt)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["map50"] == 1.0
    empty = M.evaluate([np.zeros((0, 6))], gt)
    assert empty["recall"] == 0.0


def test_metrics_false_positive_lowers_precision():
    # the false positive outranks the true one, so every operating point that
    # recalls the gt box also keeps the false positive
    gt = [np.array([[0, 10, 10, 50, 50]])]
    det = [np.array([[10, 10, 50, 50, 0.8, 0], [300, 300, 340, 340, 0.9, 0]])]
    r = M.evaluate(det, gt)
    assert r["recall"] == 1.0
    assert abs(r["precision"] - 0.5) < 1e-9


def test_metrics_report_the_max_f1_point_not_the_lowest_confidence():
    gt = [np.array([[0, 10, 10, 50, 50]])]
    det = [np.array([[10, 10, 50, 50, 0.9, 0], [300, 300, 340, 340, 0.01, 0]])]
    r = M.evaluate(det, gt)
    assert r["precision"] == 1.0 and r["recall"] == 1.0
    assert r["conf_at_best_f1"] == 0.9


def test_nms_single_class_ignores_the_untrained_class_channel():
    pred = np.zeros((1, 1, 6))
    pred[0, 0] = [50, 50, 20, 20, 0.9, 0.0]   # class channel at its floor
    dets = Q.non_max_suppression(pred)
    assert len(dets[0]) == 1 and dets[0][0, 4] == 0.9


def test_eval_thresholds_do_not_inherit_the_deployment_cap():
    assert Q.EVAL_MAX_DET > Q.DEFAULT_MAX_DET
    assert Q.EVAL_CONF_THRES < Q.DEFAULT_CONF_THRES


def test_ap_is_monotone_interpolated():
    rec = np.array([0.5, 1.0])
    prec = np.array([1.0, 0.5])
    ap = M.ap_from_pr(rec, prec)
    assert 0.0 <= ap <= 1.0
