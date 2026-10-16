import struct

HEADER_FORMAT = "<IIIHH"           # little-endian; 4+4+4+2+2 = 16 bytes
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_SIZE == 16, "PAPER FACT 3.11.5 specifies a 16-byte header"

DEFAULT_PORT = 8765                # REPOSITORY FACT infer_server.py
DEFAULT_CLIENT_QUEUE = 30          # REPOSITORY FACT infer_client_v3.py:423
SERVER_BATCH_CAP = 100             # PAPER FACT 3.11.9 ("batches of up to 100")

# dtype_code registry. uint8 in / int8 out matches the accelerator's
# REPOSITORY FACT io contract (idt UINT8, odt INT8).
DTYPE_UINT8 = 0
DTYPE_INT8 = 1
DTYPE_FLOAT32 = 2
DTYPE_NAMES = {DTYPE_UINT8: "uint8", DTYPE_INT8: "int8", DTYPE_FLOAT32: "float32"}
DTYPE_ITEMSIZE = {DTYPE_UINT8: 1, DTYPE_INT8: 1, DTYPE_FLOAT32: 4}

CMD_LOAD_MODEL = "load_model"       # PAPER FACT 3.11.4
CMD_STATUS = "status"               # PAPER FACT 3.11.4
STATUS_UNINIT = "uninit"            # REPOSITORY FACT infer_server.py
STATUS_INIT_DONE = "init_done"      # REPOSITORY FACT infer_server.py


class ProtocolError(ValueError):
    pass


def pack_header(frame_id, width, height, channels, dtype_code):
    return struct.pack(HEADER_FORMAT, int(frame_id), int(width), int(height),
                       int(channels), int(dtype_code))


def unpack_header(buf):
    if len(buf) < HEADER_SIZE:
        raise ProtocolError("header needs {} bytes, got {}".format(
            HEADER_SIZE, len(buf)))
    frame_id, width, height, channels, dtype_code = struct.unpack(
        HEADER_FORMAT, bytes(buf[:HEADER_SIZE]))
    return {"frame_id": frame_id, "width": width, "height": height,
            "channels": channels, "dtype_code": dtype_code}


def encode_frame(frame_id, array_hwc, dtype_code=DTYPE_UINT8):
    """array_hwc: numpy (H, W, C). Returns one binary message."""
    h, w = array_hwc.shape[0], array_hwc.shape[1]
    c = array_hwc.shape[2] if array_hwc.ndim == 3 else 1
    payload = memoryview(array_hwc.tobytes())
    return pack_header(frame_id, w, h, c, dtype_code) + bytes(payload)


def decode_frame(message):
    """Returns (header_dict, payload_bytes), validating the declared size."""
    hdr = unpack_header(message)
    payload = bytes(message[HEADER_SIZE:])
    itemsize = DTYPE_ITEMSIZE.get(hdr["dtype_code"])
    if itemsize is None:
        raise ProtocolError("unknown dtype_code {}".format(hdr["dtype_code"]))
    expect = hdr["width"] * hdr["height"] * hdr["channels"] * itemsize
    if len(payload) != expect:
        raise ProtocolError(
            "payload {} bytes but header declares {} ({}x{}x{} {})".format(
                len(payload), expect, hdr["height"], hdr["width"],
                hdr["channels"], DTYPE_NAMES[hdr["dtype_code"]]))
    return hdr, payload


def payload_to_array(hdr, payload):
    import numpy as np
    dtype = {DTYPE_UINT8: np.uint8, DTYPE_INT8: np.int8,
             DTYPE_FLOAT32: np.float32}[hdr["dtype_code"]]
    arr = np.frombuffer(payload, dtype=dtype)
    return arr.reshape(hdr["height"], hdr["width"], hdr["channels"])
