"""Protocol and .fnn package tests. Stdlib + numpy -- no websockets, no board."""

import json
import os
import pickle
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A
from streamer import fnn_package as FP
from streamer import protocol as P


def test_header_is_exactly_16_bytes():
    """PAPER FACT 3.11.5."""
    assert P.HEADER_SIZE == 16
    assert len(P.pack_header(1, 416, 416, 3, 0)) == 16


def test_header_field_layout_roundtrip():
    blob = P.pack_header(12345, 1280, 720, 3, P.DTYPE_UINT8)
    hdr = P.unpack_header(blob)
    assert hdr == {"frame_id": 12345, "width": 1280, "height": 720,
                   "channels": 3, "dtype_code": P.DTYPE_UINT8}


def test_encode_decode_frame_roundtrip():
    arr = (np.random.rand(416, 416, 3) * 255).astype(np.uint8)
    msg = P.encode_frame(7, arr, P.DTYPE_UINT8)
    assert len(msg) == 16 + arr.nbytes
    hdr, payload = P.decode_frame(msg)
    assert hdr["frame_id"] == 7
    back = P.payload_to_array(hdr, payload)
    assert np.array_equal(back, arr)


def test_decode_rejects_truncated_payload():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    msg = P.encode_frame(1, arr, P.DTYPE_UINT8)
    try:
        P.decode_frame(msg[:-5])
    except P.ProtocolError:
        pass
    else:
        raise AssertionError("truncated payload must raise")


def test_decode_rejects_unknown_dtype():
    bad = P.pack_header(1, 2, 2, 1, 99) + b"\x00" * 4
    try:
        P.decode_frame(bad)
    except P.ProtocolError:
        pass
    else:
        raise AssertionError("unknown dtype_code must raise")


def test_int8_output_frame_roundtrip():
    """REPOSITORY FACT: odt is INT8, shape (1,13,13,18)."""
    out = np.random.randint(-127, 128, (13, 13, 18)).astype(np.int8)
    hdr, payload = P.decode_frame(P.encode_frame(3, out, P.DTYPE_INT8))
    assert np.array_equal(P.payload_to_array(hdr, payload), out)
    assert hdr["channels"] == A.OUTPUT_CHANNELS


def test_protocol_constants_match_sources():
    assert P.DEFAULT_PORT == 8765              # REPOSITORY FACT
    assert P.DEFAULT_CLIENT_QUEUE == 30        # REPOSITORY FACT
    assert P.SERVER_BATCH_CAP == 100           # PAPER FACT 3.11.9
    assert P.CMD_LOAD_MODEL == "load_model"    # PAPER FACT 3.11.4
    assert P.CMD_STATUS == "status"


def _make_fnn(tmp, info=None, omit=None):
    bit = os.path.join(tmp, "x.bit")
    hwh = os.path.join(tmp, "x.hwh")
    open(bit, "wb").write(b"BITSTREAM")
    open(hwh, "w").write("<EDKSYSTEM/>")
    out = os.path.join(tmp, "m.fnn")
    shapes = {"ishape_normal": [A.IO_CONTRACT["ishape_normal"]],
              "oshape_normal": [A.IO_CONTRACT["oshape_normal"]]}
    info = info or {"model_name": "yolov3-tiny", "trained_on_dataset": "visdrone",
                    "max_clock_freq": 200.0, "clock_mode": "aggressive"}
    FP.build(out, bit, hwh, shapes, info)
    if omit:
        keep = [n for n in zipfile.ZipFile(out).namelist() if n != omit]
        src = zipfile.ZipFile(out)
        tmp2 = out + ".tmp"
        with zipfile.ZipFile(tmp2, "w") as z:
            for n in keep:
                z.writestr(n, src.read(n))
        src.close()
        os.replace(tmp2, out)
    return out


def test_fnn_contains_exactly_the_four_required_files():
    """PAPER FACT 3.11.12, confirmed against the repository's reference .fnn."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_fnn(tmp)
        names = set(zipfile.ZipFile(path).namelist())
        assert names == set(FP.REQUIRED_FILES)


def test_fnn_validate_returns_info_and_md5():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_fnn(tmp)
        info, md5 = FP.validate(path)
        assert info["max_clock_freq"] == 200.0      # REPOSITORY FACT
        assert info["model_name"] == "yolov3-tiny"
        assert len(md5) == 32


def test_fnn_rejects_incomplete_package():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_fnn(tmp, omit="model.pkl")
        try:
            FP.validate(path)
        except FP.PackageError as exc:
            assert "model.pkl" in str(exc)
        else:
            raise AssertionError("incomplete .fnn must raise")


def test_fnn_rejects_missing_metadata_keys():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _make_fnn(tmp, info={"model_name": "x"})
        except FP.PackageError as exc:
            assert "max_clock_freq" in str(exc) or "missing" in str(exc)
        else:
            raise AssertionError("missing metadata keys must raise")


def test_fnn_extract_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_fnn(tmp)
        res = FP.extract(path, os.path.join(tmp, "out"))
        assert os.path.isfile(res["shape_dict_path"])
        with open(res["shape_dict_path"], "rb") as fh:
            shapes = pickle.load(fh)
        assert shapes["oshape_normal"][0] == A.IO_CONTRACT["oshape_normal"]


def test_md5_changes_with_content():
    with tempfile.TemporaryDirectory() as tmp:
        a = _make_fnn(tmp)
        d1 = FP.md5_of_file(a)
        with open(a, "ab") as fh:
            fh.write(b"x")
        assert FP.md5_of_file(a) != d1
