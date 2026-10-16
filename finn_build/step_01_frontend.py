#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finn_build import custom_steps as CS   # noqa: E402


def run(onnx_in, out_dir, divisor=255, save_intermediates=True):
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.general import (
        GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs,
        RemoveUnusedTensors)
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.fold_constants import FoldConstants
    from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
    from finn.transformation.streamline import Streamline
    from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
    from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
    from finn.transformation.fpgadataflow.convert_to_hw_layers import (
        InferBinaryMatrixVectorActivation, InferQuantizedMatrixVectorActivation,
        InferThresholdingLayer, InferStreamingMaxPool, InferConvInpGen)
    from finn.transformation.fpgadataflow.create_dataflow_partition import (
        CreateDataflowPartition)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    def snap(m, tag):
        if save_intermediates:
            p = os.path.join(out_dir, "step01_{}.onnx".format(tag))
            m.save(p)
            return p
        return None

    manifest = {"step": "01_frontend", "artifacts": {},
                "tidy_up_passes_A20": [
                    "InferShapes", "FoldConstants", "GiveUniqueNodeNames",
                    "GiveReadableTensorNames", "InferDataTypes",
                    "RemoveStaticGraphInputs", "RemoveUnusedTensors"]}

    model = ModelWrapper(onnx_in)

    # 1 + 2 ---------------------------------------------------------------
    model = model.transform(ConvertQONNXtoFINN())
    for t in (InferShapes(), FoldConstants(), GiveUniqueNodeNames(),
              GiveReadableTensorNames(), InferDataTypes(),
              RemoveStaticGraphInputs(), RemoveUnusedTensors()):
        model = model.transform(t)
    manifest["artifacts"]["after_convert_tidy"] = snap(model, "a_convert_tidy")

    # 3 -- Fig. 4 ---------------------------------------------------------
    model = CS.insert_input_div(model, divisor)
    model = model.transform(InferShapes()).transform(InferDataTypes())
    manifest["artifacts"]["fig4_finn_onnx"] = snap(model, "b_fig4_div")

    # 4 -- Fig. 5 ---------------------------------------------------------
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(Streamline())
    for t in (InferShapes(), InferDataTypes(), GiveUniqueNodeNames()):
        model = model.transform(t)
    manifest["artifacts"]["fig5_streamlined"] = snap(model, "c_fig5_streamline")

    # 5 -- Fig. 6 ---------------------------------------------------------
    for t in (InferConvInpGen(), InferStreamingMaxPool(),
              InferQuantizedMatrixVectorActivation(),
              InferBinaryMatrixVectorActivation(), InferThresholdingLayer()):
        model = model.transform(t)
    model = model.transform(GiveUniqueNodeNames())
    manifest["artifacts"]["fig6_hw_layers"] = snap(model, "d_fig6_hw")

    # 6 -- Fig. 7 ---------------------------------------------------------
    parent = model.transform(CreateDataflowPartition())
    manifest["artifacts"]["fig7_parent"] = snap(parent, "e_fig7_parent")

    # ---- PAPER-FACT assertions (build gates) ----------------------------
    CS.assert_single_dataflow_partition(parent)
    CS.assert_residual_parent_nodes(parent)
    sdp = [n for n in parent.graph.node
           if n.op_type == "StreamingDataflowPartition"][0]
    child_path = [a for a in sdp.attribute if a.name == "model"][0].s.decode()
    child = ModelWrapper(child_path)
    CS.assert_no_float_ops_inside(child)
    CS.assert_mvu_inventory(child)
    child_out = os.path.join(out_dir, "step01_child_dataflow.onnx")
    child.save(child_out)
    manifest["artifacts"]["child_dataflow"] = child_out
    manifest["assertions_passed"] = list(CS.ALL_ASSERTIONS[:2] + CS.ALL_ASSERTIONS[2:])

    with open(os.path.join(out_dir, "step01_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return child_out, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="QONNX model from export.py")
    ap.add_argument("--out-dir", default="finn_build_out")
    ap.add_argument("--divisor", type=int, default=255, help="A3")
    args = ap.parse_args(argv)
    child, manifest = run(args.onnx, args.out_dir, args.divisor)
    print(json.dumps(manifest, indent=2))
    print("\nchild dataflow graph: {}".format(child))
    return 0


if __name__ == "__main__":
    sys.exit(main())
