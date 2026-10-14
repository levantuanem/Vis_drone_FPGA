""".fnn model package -- PAPER FACT 3.11.11-3.11.13.

A `.fnn` is a zip containing exactly:
    model.bit          FPGA bitstream
    model.hwh          hardware handoff
    model.pkl          shape dictionaries (ishape_* / oshape_*)
    model_info.json    max_clock_freq, model_name, trained_on_dataset

Confirmed byte-for-byte against the repository's reference package
`client/test_package/yolov3-tiny-4w4a-visdrone.fnn`, whose model_info.json is:
    {"model_name": "yolov3-tiny", "trained_on_dataset": "visdrone",
     "max_clock_freq": 200.0, "clock_mode": "aggressive"}

PAPER FACT 3.11.4 / 3.11.13: integrity is checked with an MD5 of the .fnn, and
the server verifies all four files are present before configuring the FPGA.

Stdlib only -- unit-testable without PYNQ or a board.
"""

import hashlib
import json
import os
import pickle
import zipfile

REQUIRED_FILES = ("model.bit", "model.hwh", "model.pkl", "model_info.json")
REQUIRED_INFO_KEYS = ("max_clock_freq", "model_name", "trained_on_dataset")


class PackageError(ValueError):
    pass


def md5_of_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build(out_path, bit, hwh, shape_dict, model_info):
    """Create a .fnn. `shape_dict` is pickled as model.pkl."""
    missing = [k for k in REQUIRED_INFO_KEYS if k not in model_info]
    if missing:
        raise PackageError("model_info.json missing {} (REPOSITORY FACT "
                           "requires {})".format(missing, REQUIRED_INFO_KEYS))
    d = os.path.dirname(out_path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(bit, "model.bit")
        z.write(hwh, "model.hwh")
        z.writestr("model.pkl", pickle.dumps(shape_dict, protocol=2))
        z.writestr("model_info.json", json.dumps(model_info, indent=2))
    return out_path


def validate(path):
    """Check structure and metadata. Returns (model_info, md5). Raises on error."""
    if not zipfile.is_zipfile(path):
        raise PackageError("{} is not a zip archive".format(path))
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        missing = [f for f in REQUIRED_FILES if f not in names]
        if missing:
            raise PackageError(
                "incomplete .fnn, missing {} (PAPER FACT 3.11.12 requires "
                "{})".format(missing, list(REQUIRED_FILES)))
        info = json.loads(z.read("model_info.json").decode())
        missing_keys = [k for k in REQUIRED_INFO_KEYS if k not in info]
        if missing_keys:
            raise PackageError("model_info.json missing {}".format(missing_keys))
    return info, md5_of_file(path)


def extract(path, dest_dir):
    """PAPER FACT 3.11.13 step 1-2: extract then verify completeness."""
    info, digest = validate(path)
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    with zipfile.ZipFile(path) as z:
        z.extractall(dest_dir)
    for f in REQUIRED_FILES:
        p = os.path.join(dest_dir, f)
        if not os.path.isfile(p):
            raise PackageError("extraction incomplete: {}".format(p))
    return {"dir": dest_dir, "info": info, "md5": digest,
            "shape_dict_path": os.path.join(dest_dir, "model.pkl")}
