"""Augmentation -- OUR ASSUMPTION A5 (hyper-parameters) / A4 (geometry).

The paper gives no augmentation detail beyond "standard low-scratch
hyper-parameters" (3.1.4), so every operation here is driven by
configs/hyp.scratch-low.yaml and nothing is hard-coded.

Deliberately conservative: only HSV jitter and horizontal flip are implemented,
because those are the two the low-augmentation preset actually enables with
non-zero gain for this task (fliplr 0.5, hsv_*). Mosaic and mixup are declared in
the hyp file but NOT implemented here -- enabling them would change the
train-time geometry away from the inference letterbox (A4). Recorded as an
explicit limitation rather than silently differing from the preset.
"""

import numpy as np


def augment_hsv(img_bgr, hyp):
    h_gain = float(hyp.get("hsv_h", 0.0))
    s_gain = float(hyp.get("hsv_s", 0.0))
    v_gain = float(hyp.get("hsv_v", 0.0))
    if not (h_gain or s_gain or v_gain):
        return img_bgr
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return img_bgr
    r = np.random.uniform(-1, 1, 3) * np.array([h_gain, s_gain, v_gain]) + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
    dtype = img_bgr.dtype
    x = np.arange(0, 256, dtype=np.int16)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s),
                        cv2.LUT(val, lut_v)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def random_flip(img, labels, hyp):
    """Horizontal flip. labels: (N,5) cls,cx,cy,w,h normalised."""
    p = float(hyp.get("fliplr", 0.0))
    if p > 0 and np.random.rand() < p:
        img = np.ascontiguousarray(img[:, ::-1])
        if len(labels):
            labels = labels.copy()
            labels[:, 1] = 1.0 - labels[:, 1]
    p_ud = float(hyp.get("flipud", 0.0))
    if p_ud > 0 and np.random.rand() < p_ud:
        img = np.ascontiguousarray(img[::-1, :])
        if len(labels):
            labels = labels.copy()
            labels[:, 2] = 1.0 - labels[:, 2]
    return img, labels


UNIMPLEMENTED_FROM_PRESET = ("mosaic", "mixup", "copy_paste", "degrees",
                             "translate", "scale", "shear", "perspective")
