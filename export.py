#!/usr/bin/env python3
"""QONNX export (T1.4).

PAPER FACT 3.2 gives the command verbatim:
    python export.py --weights /path/to/train_run/weights/best.pt \
                     --img 416 --include finn_onnx
and states that "the detect layer was detached from the network, due to the
aforementioned FINN's compatibility issues".

Acceptance gates enforced here:
  - output tensor shape (1,13,13,18) or (1,18,13,13) pre-transpose  REPOSITORY FACT
  - the detection head is absent from the exported graph            PAPER FACT 3.2

DEPENDENCY NOT INSTALLED: torch==1.13.1, brevitas==0.9.0, onnx, qonnx.
OUR ASSUMPTION A18 export API `export_qonnx`; A19 opset pinned in configs/quant.yaml.
NOT RUN in this phase.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import arch_spec as A   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img", type=int, default=416)        # PAPER FACT 3.2
    ap.add_argument("--include", default="finn_onnx",
                    choices=["finn_onnx"])                 # PAPER FACT 3.2
    ap.add_argument("--quant", default="configs/quant.yaml")
    ap.add_argument("--out", default="build/model_qonnx.onnx")
    ap.add_argument("--opset", type=int, default=None,
                    help="override A19; default read from configs/quant.yaml")
    args = ap.parse_args(argv)

    import torch
    import yaml
    from brevitas.export import export_qonnx
    from models.quant_yolov3_tiny import build as build_quant

    cfg = {}
    if os.path.isfile(args.quant):
        with open(args.quant) as fh:
            cfg = yaml.safe_load(fh) or {}
    opset = args.opset or cfg.get("export", {}).get("onnx_opset", 13)   # A19

    model = build_quant(args.quant)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    # PAPER FACT 3.2: the detect head is detached. Our backbone never contained
    # it (models/quant_yolov3_tiny.py), so there is nothing to strip -- assert it.
    assert not hasattr(model, "detect"), \
        "backbone must not contain a detection head (PAPER FACT 3.2)"

    dummy = torch.zeros(1, A.INPUT_CHANNELS, args.img, args.img)
    with torch.no_grad():
        out = model(dummy)
    expected = (1, A.OUTPUT_CHANNELS, 13, 13)
    assert tuple(out.shape) == expected, \
        "output shape {} != expected {} (REPOSITORY FACT oshape, pre-transpose)".format(
            tuple(out.shape), expected)

    d = os.path.dirname(args.out)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    export_qonnx(model, input_t=dummy, export_path=args.out, opset_version=opset)
    print("wrote {} (opset {}, A19)".format(args.out, opset))
    print("NHWC/NCHW note: FINN works in NHWC; the two residual Transpose nodes")
    print("that PAPER FACT 3.4.3 describes are handled by the driver.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
