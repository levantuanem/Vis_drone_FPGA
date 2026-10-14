"""Preprocessing -- REPOSITORY FACT.

The paper says only "resizing and color channel inversion" (3.11.7). The
repository is more specific (infer_client_v3.py:236-243):

    frame_resized = utils.letterbox(frame, (416, 416), auto=False)[0]
    frame_resized = frame_resized[:, :, ::-1]          # BGR -> RGB

Letterboxing (aspect-preserving pad) rather than a plain resize materially
changes mAP, which is why this is pinned from the repository rather than guessed.
Data stays UINT8: the accelerator's idt is UINT8 and normalisation happens
on-chip via the inserted Div node (PAPER FACT 3.4.1).

OUR ASSUMPTION A4: the same letterbox is used at TRAINING time, so train and
inference preprocessing cannot drift.

cv2 is optional -- a numpy nearest/bilinear fallback keeps this importable (and
testable) without opencv.
"""

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:  # pragma: no cover - opencv is a documented dependency
    _HAVE_CV2 = False

LETTERBOX_FILL = 114   # conventional grey pad value


def _resize(img, new_w, new_h):
    if _HAVE_CV2:
        interp = cv2.INTER_LINEAR if (new_w > img.shape[1]) else cv2.INTER_AREA
        return cv2.resize(img, (new_w, new_h), interpolation=interp)
    # nearest-neighbour fallback: adequate for shape/geometry tests only
    ys = (np.linspace(0, img.shape[0] - 1, new_h)).astype(np.int64)
    xs = (np.linspace(0, img.shape[1] - 1, new_w)).astype(np.int64)
    return img[ys][:, xs]


def letterbox(img, new_shape=(416, 416), fill=LETTERBOX_FILL, auto=False):
    """Aspect-preserving resize with centre padding.

    Returns (padded_image, gain, (pad_x, pad_y)). Mirrors Ultralytics'
    letterbox with auto=False, which is what the repository calls.
    """
    img = np.asarray(img)
    h, w = img.shape[:2]
    new_h, new_w = new_shape
    gain = min(new_h / float(h), new_w / float(w))
    rw, rh = int(round(w * gain)), int(round(h * gain))
    resized = _resize(img, rw, rh)

    pad_w, pad_h = new_w - rw, new_h - rh
    top = pad_h // 2
    left = pad_w // 2
    if img.ndim == 3:
        out = np.full((new_h, new_w, img.shape[2]), fill, dtype=img.dtype)
        out[top:top + rh, left:left + rw, :] = resized
    else:
        out = np.full((new_h, new_w), fill, dtype=img.dtype)
        out[top:top + rh, left:left + rw] = resized
    return out, gain, (left, top)


def bgr_to_rgb(img):
    """REPOSITORY FACT infer_client_v3.py:242 -- channel inversion."""
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def preprocess_frame(frame_bgr, size=416):
    """Camera/file frame (BGR uint8 HWC) -> accelerator input (RGB uint8 NHWC).

    Returns (tensor_nhwc_uint8, gain, pad) so detections can be mapped back.
    """
    padded, gain, pad = letterbox(frame_bgr, (size, size))
    rgb = bgr_to_rgb(padded)
    assert rgb.dtype == np.uint8, "accelerator idt is UINT8 (REPOSITORY FACT)"
    return rgb[None, ...], gain, pad
