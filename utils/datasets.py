"""Dataset / dataloader for the converted VisDrone car dataset (T1.2, T1.3).

DEPENDENCY NOT INSTALLED: torch==1.13.1 (PAPER FACT Table 2). numpy and the
preprocessing helpers are used directly so geometry stays identical to inference.

OUR ASSUMPTION A4: training uses the same letterbox as inference
(REPOSITORY FACT infer_client_v3.py:241), so train/infer geometry cannot drift.
"""

from __future__ import absolute_import

import os

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "utils.datasets requires torch==1.13.1 (PAPER FACT Table 2); "
        "not installed by design. See requirements/README.md."
    ) from exc

from utils.preprocess import letterbox, bgr_to_rgb
from utils.augment import augment_hsv, random_flip

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _read_image(path):
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)   # BGR
        if img is None:
            raise IOError(path)
        return img
    except ImportError:  # pragma: no cover
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]


class VisDroneCarDataset(Dataset):
    """Images + YOLO-XYWH labels for the single `car` class."""

    def __init__(self, root, split, img_size=416, augment=False, hyp=None):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.hyp = hyp or {}
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(self.img_dir)
        self.files = [f for f in sorted(os.listdir(self.img_dir))
                      if f.lower().endswith(IMAGE_EXTS)]

    def __len__(self):
        return len(self.files)

    def _labels(self, stem):
        path = os.path.join(self.lbl_dir, stem + ".txt")
        if not os.path.isfile(path):
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        with open(path) as fh:
            for line in fh:
                p = line.split()
                if len(p) == 5:
                    rows.append([float(x) for x in p])
        return (np.array(rows, dtype=np.float32) if rows
                else np.zeros((0, 5), dtype=np.float32))

    def __getitem__(self, i):
        name = self.files[i]
        stem = os.path.splitext(name)[0]
        img = _read_image(os.path.join(self.img_dir, name))
        h0, w0 = img.shape[:2]
        labels = self._labels(stem)          # cls, cx, cy, w, h (normalised)

        padded, gain, (pad_x, pad_y) = letterbox(img, (self.img_size, self.img_size))

        if len(labels):
            lb = labels.copy()
            # normalised source coords -> letterboxed normalised coords
            lb[:, 1] = (lb[:, 1] * w0 * gain + pad_x) / self.img_size
            lb[:, 2] = (lb[:, 2] * h0 * gain + pad_y) / self.img_size
            lb[:, 3] = lb[:, 3] * w0 * gain / self.img_size
            lb[:, 4] = lb[:, 4] * h0 * gain / self.img_size
            labels = lb

        if self.augment:
            padded = augment_hsv(padded, self.hyp)
            padded, labels = random_flip(padded, labels, self.hyp)

        rgb = bgr_to_rgb(padded)
        # Model input stays in [0,255]: QuantInputStem divides by 255 on-device
        # (PAPER FACT 3.4.1 Div node; OUR ASSUMPTION A3 divisor).
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        return tensor, torch.from_numpy(labels), name

    @staticmethod
    def collate_fn(batch):
        imgs, labels, names = zip(*batch)
        out = []
        for i, lb in enumerate(labels):
            if lb.numel():
                idx = torch.full((lb.shape[0], 1), float(i))
                out.append(torch.cat([idx, lb], dim=1))
        targets = (torch.cat(out, 0) if out else torch.zeros((0, 6)))
        return torch.stack(imgs, 0), targets, list(names)


def build_loader(root, split, img_size=416, batch_size=64, augment=False,
                 hyp=None, workers=4, shuffle=None):
    ds = VisDroneCarDataset(root, split, img_size, augment, hyp)
    if shuffle is None:
        shuffle = augment
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=ds.collate_fn,
                      pin_memory=True, drop_last=False)
