#!/usr/bin/env python3
"""InferClient -- 4-thread streaming client (T2.4). PAPER FACT 3.11.2.

Threads, exactly as paper 3.11.2 lists them:
  1 Capture          OpenCV frames -> letterbox 416x416 RGB -> bounded queue
  2 Frame Dispatcher binary messages -> WebSocket (non-blocking)
  3 Post-processing  match results by frame_id -> dequant, decode, NMS, draw
  4 Display          render annotated frames

OUR ADDITION (I7): `--synthetic-fps` drives the pipeline from a generator faster
than 30 FPS, so a 30 FPS result can be shown to be pipeline-limited rather than
camera-limited. The paper reports 30 FPS without separating the two.

DEPENDENCIES NOT INSTALLED: `websockets`, `opencv-python`.
HARDWARE BLOCKER: D9 PENDING -- requires a reachable InferServer on the board.
"""

import argparse
import asyncio
import base64
import json
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A          # noqa: E402
from streamer import protocol as P         # noqa: E402
from streamer import fnn_package           # noqa: E402


class Pipeline(object):
    """Thread/queue skeleton. Transport-agnostic so it is testable offline."""

    def __init__(self, send_fn, queue_size=P.DEFAULT_CLIENT_QUEUE, img_size=416,
                 draw=True):
        self.send_fn = send_fn
        self.img_size = int(img_size)
        self.draw = draw
        self.capture_q = queue.Queue(maxsize=queue_size)
        self.original_q = queue.Queue(maxsize=2 * queue_size)
        self.result_q = queue.Queue(maxsize=queue_size)
        self.display_q = queue.Queue(maxsize=queue_size)
        self.stop = threading.Event()
        self.stats = {"captured": 0, "sent": 0, "received": 0, "displayed": 0,
                      "dropped": 0}
        self._threads = []

    # -- thread 1 ---------------------------------------------------------
    def capture_loop(self, frame_source):
        from utils.preprocess import preprocess_frame
        fid = 0
        for frame in frame_source:
            if self.stop.is_set():
                break
            tensor, gain, pad = preprocess_frame(frame, self.img_size)
            try:
                self.capture_q.put((fid, tensor[0]), timeout=0.5)
                self.original_q.put((fid, frame, gain, pad), timeout=0.5)
                self.stats["captured"] += 1
            except queue.Full:
                self.stats["dropped"] += 1
            fid += 1

    # -- thread 2 ---------------------------------------------------------
    def sender_loop(self):
        while not self.stop.is_set():
            try:
                fid, arr = self.capture_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.send_fn(P.encode_frame(fid, arr, P.DTYPE_UINT8))
            self.stats["sent"] += 1

    # -- thread 3 ---------------------------------------------------------
    def postprocess_loop(self):
        from utils.postprocess import postprocess
        pending = {}
        while not self.stop.is_set():
            try:
                fid, out = self.result_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.stats["received"] += 1
            while True:
                try:
                    ofid, frame, gain, pad = self.original_q.get_nowait()
                    pending[ofid] = (frame, gain, pad)
                except queue.Empty:
                    break
            if fid not in pending:
                continue
            frame, gain, pad = pending.pop(fid)
            dets = postprocess(out[None, ...], frame.shape[:2],
                               input_size=self.img_size)[0]
            try:
                self.display_q.put((fid, frame, dets), timeout=0.2)
            except queue.Full:
                self.stats["dropped"] += 1

    # -- thread 4 ---------------------------------------------------------
    def display_loop(self):
        while not self.stop.is_set():
            try:
                fid, frame, dets = self.display_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.stats["displayed"] += 1
            if self.draw:
                self._render(frame, dets)

    @staticmethod
    def _render(frame, dets):           # pragma: no cover (needs a GUI)
        try:
            import cv2
        except ImportError:
            return
        for x1, y1, x2, y2, conf, _cls in dets:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(frame, "car {:.2f}".format(conf), (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("FINN VisDrone YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def start(self, frame_source):
        targets = [(self.capture_loop, (frame_source,)), (self.sender_loop, ()),
                   (self.postprocess_loop, ()), (self.display_loop, ())]
        for fn, a in targets:
            t = threading.Thread(target=fn, args=a, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def shutdown(self, timeout=2.0):
        self.stop.set()
        for t in self._threads:
            t.join(timeout=timeout)
        return self.stats


def synthetic_source(n, size=720, fps=None):
    """OUR ADDITION I7 -- a source faster than 30 FPS, to separate the camera
    limit from the pipeline limit (the paper does not do this)."""
    import numpy as np
    period = (1.0 / fps) if fps else 0.0
    for _ in range(n):
        t0 = time.time()
        yield (np.random.rand(size, int(size * 16 / 9), 3) * 255).astype(np.uint8)
        if period:
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.2.99")
    ap.add_argument("--port", type=int, default=P.DEFAULT_PORT)
    ap.add_argument("--model", help=".fnn package to upload")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--img", type=int, default=416)
    ap.add_argument("--queue-size", type=int, default=P.DEFAULT_CLIENT_QUEUE)
    ap.add_argument("--synthetic-fps", type=float, default=None,
                    help="I7: drive from a synthetic source at this rate")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--validate-model", action="store_true",
                    help="validate a .fnn package offline and exit (no board)")
    ap.add_argument("--pipeline-self-test", action="store_true",
                    help="run the 4-thread pipeline against a loopback stub "
                         "(no board, no network)")
    args = ap.parse_args(argv)

    if args.validate_model:
        if not args.model:
            ap.error("--validate-model needs --model")
        info, md5 = fnn_package.validate(args.model)
        print(json.dumps({"model_info": info, "md5": md5,
                          "required_files": list(fnn_package.REQUIRED_FILES)},
                         indent=2))
        return 0

    if args.pipeline_self_test:
        import numpy as np
        pipe_box = {}

        def loopback(message):
            hdr, _payload = P.decode_frame(message)
            out = np.zeros((13, 13, A.OUTPUT_CHANNELS), dtype=np.int8)
            pipe_box["pipe"].result_q.put((hdr["frame_id"], out))

        pipe = Pipeline(loopback, queue_size=args.queue_size, img_size=args.img,
                        draw=False)
        pipe_box["pipe"] = pipe
        pipe.start(synthetic_source(args.frames, fps=args.synthetic_fps))
        t0 = time.time()
        while time.time() - t0 < 20 and pipe.stats["displayed"] < args.frames:
            time.sleep(0.05)
        elapsed = time.time() - t0
        stats = pipe.shutdown()
        print(json.dumps(stats, indent=2))
        print("elapsed {:.2f}s -> {:.1f} frames/s end-to-end through the "
              "4-thread pipeline (loopback stub, NO accelerator)".format(
                  elapsed, stats["displayed"] / max(elapsed, 1e-9)))
        return 0 if stats["displayed"] > 0 else 1

    print("HARDWARE BLOCKER: D9 PENDING -- no reachable InferServer (board absent).")
    print("Offline options: --validate-model, --pipeline-self-test")
    return 1


if __name__ == "__main__":
    sys.exit(main())
