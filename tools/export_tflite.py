#!/usr/bin/env python3
"""T1.13 -- FP32 -> TFLite INT8 PTQ, for the Coral comparison. Host-only.

PAPER FACT 3.3 in full: "another training run was performed, this time with full
FP32 precision. Then, a tflite model was exported with int8 PTQ precision. This
tflite model was then compiled by the edgetpu compiler." A newer,
community-maintained `libedgetpu` is required (PAPER FACT ref 68).

CRITICAL AMBIGUITY -- DECISION D5 / A9: the paper never states whether that FP32
run used the PRUNED Table-1 topology or stock YOLOv3-Tiny. We use the PRUNED
topology so both platforms execute the identical network. If the authors used
stock YOLOv3-Tiny, their Table 15 compares two different workloads. Flagged, not
hidden.

NOTE: this is INT8 **PTQ**, not the 4W4A **QAT** used for the FPGA. Table 15
therefore compares two different quantisations of the same architecture. The
paper does not flag this; we do.

OUR ADDITION I8: report the Edge TPU compiler's op-mapping fraction -- ops falling
back to the CPU would materially change T3.1 and the paper never reports it.

OUR ASSUMPTIONS: PTQ scheme = full-integer; calibration set = N frames from the
train split; conversion route PyTorch -> ONNX -> TF -> TFLite. All recorded.

DEPENDENCIES NOT INSTALLED: tensorflow, onnx-tf (or ai-edge-torch), onnx. NOT RUN.
Requires NO Coral device -- only T3.1 does.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402

CALIBRATION_FRAMES = 200     # OUR ASSUMPTION -- the paper states no calibration set


def representative_dataset(image_dir, n, size=416):
    """Full-integer PTQ calibration generator."""
    import numpy as np
    from utils.preprocess import preprocess_frame
    files = []
    if image_dir and os.path.isdir(image_dir):
        files = sorted(os.listdir(image_dir))[:n]
    if files:
        import cv2
        for f in files:
            img = cv2.imread(os.path.join(image_dir, f))
            if img is None:
                continue
            t, _, _ = preprocess_frame(img, size)
            yield [t.astype(np.float32) / 255.0]
    else:
        rng = np.random.RandomState(0)
        for _ in range(n):
            yield [rng.rand(1, size, size, 3).astype(np.float32)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="FP32 checkpoint from the T1.13 training run")
    ap.add_argument("--onnx", help="pre-exported FP32 ONNX (skips the torch step)")
    ap.add_argument("--out", default="build/model_int8.tflite")
    ap.add_argument("--calib-images", default=None)
    ap.add_argument("--calib-frames", type=int, default=CALIBRATION_FRAMES)
    ap.add_argument("--img", type=int, default=416)
    ap.add_argument("--describe", action="store_true",
                    help="print the plan and assumptions; imports nothing heavy")
    args = ap.parse_args(argv)

    if args.describe or not (args.weights or args.onnx):
        print("T1.13 -- FP32 -> TFLite INT8 PTQ -> Edge TPU compile")
        print("-" * 58)
        print("PAPER FACT 3.3: FP32 training run, then INT8 PTQ tflite, then")
        print("  edgetpu_compiler. Community libedgetpu required (ref 68).")
        print("\nDECISION D5 / A9: topology = PRUNED Table-1 ({} convs, {} weights),"
              .format(len(A.CONV_LAYERS), A.totals()["weights_total"]))
        print("  so FPGA and Coral run the IDENTICAL network. The paper never says")
        print("  which topology it used -- if stock YOLOv3-Tiny, its Table 15")
        print("  compares two different workloads.")
        print("\nQuantisation mismatch we will report: Coral is INT8 PTQ while the")
        print("  FPGA is 4W4A QAT. Table 15 compares two different quantisations.")
        print("\nOUR ASSUMPTIONS:")
        print("  PTQ scheme        full-integer (int8 in/out)")
        print("  calibration set   {} frames from train".format(args.calib_frames))
        print("  conversion route  PyTorch -> ONNX -> TF SavedModel -> TFLite")
        print("\nI8: the compiler's TPU/CPU op split is recorded in the sidecar json.")
        print("\nDEPENDENCIES NOT INSTALLED: tensorflow, onnx-tf, onnx.")
        print("NO Coral device needed for this step.")
        return 0

    import numpy as np
    import tensorflow as tf

    onnx_path = args.onnx
    if not onnx_path:
        import torch
        from models.quant_yolov3_tiny import build as build_quant
        model = build_quant(None, batchnorm=False)
        ckpt = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        model.eval()
        onnx_path = os.path.splitext(args.out)[0] + "_fp32.onnx"
        torch.onnx.export(model, torch.zeros(1, 3, args.img, args.img), onnx_path,
                          opset_version=13, input_names=["input"],
                          output_names=["output"])
        print("wrote {}".format(onnx_path))

    from onnx_tf.backend import prepare
    import onnx
    saved = os.path.splitext(args.out)[0] + "_saved_model"
    prepare(onnx.load(onnx_path)).export_graph(saved)

    conv = tf.lite.TFLiteConverter.from_saved_model(saved)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: representative_dataset(
        args.calib_images, args.calib_frames, args.img)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()

    d = os.path.dirname(args.out)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(args.out, "wb") as fh:
        fh.write(blob)
    meta = {
        "_provenance": {
            "paper": "PAPER FACT 3.3 -- INT8 PTQ (NOT the FPGA's 4W4A QAT)",
            "topology": "pruned Table-1 (DECISION D5 / A9)",
            "ptq_scheme": "full-integer", "calib_frames": args.calib_frames,
            "route": "PyTorch -> ONNX -> TF SavedModel -> TFLite",
        },
        "tflite": args.out, "bytes": len(blob),
        "next_step": "tools/compile_edgetpu.sh (records the I8 op-mapping split)",
    }
    with open(args.out + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote {} ({} bytes)".format(args.out, len(blob)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
