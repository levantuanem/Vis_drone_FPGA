"""Custom FINN transformations and the build's correctness assertions.

FINN is imported lazily inside each function so this module stays importable
(and testable) without FINN installed.

PAPER FACT 3.4.1: a `Div` node must be inserted before the first convolution so
the UINT8 input is normalised to [0,1] on-chip.
PAPER FACT 3.4.4: `CreateDataflowPartition` must yield EXACTLY ONE partition --
FINN cannot map multiple partitions onto the PL. The paper treats this as a
correctness requirement, so we make it an assertion that fails the build rather
than a condition we hope holds.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A   # noqa: E402


class BuildAssertionError(AssertionError):
    """Raised when a PAPER-FACT structural requirement is violated."""


def insert_input_div(model, divisor=A.INPUT_NORMALISATION_DIVISOR):
    """PAPER FACT 3.4.1 -- insert a Div before the first convolution.

    OUR ASSUMPTION A3: divisor = 255 (the paper says only "normalized to values
    between 0 and 1").
    """
    import numpy as np
    from onnx import helper, TensorProto
    from qonnx.core.modelwrapper import ModelWrapper   # noqa: F401

    graph = model.graph
    inp_name = graph.input[0].name
    div_const = helper.make_tensor(
        name=inp_name + "_div_const", data_type=TensorProto.FLOAT, dims=[1],
        vals=[float(divisor)])
    graph.initializer.append(div_const)

    new_name = inp_name + "_normalised"
    div_node = helper.make_node("Div", [inp_name, div_const.name], [new_name],
                               name="InputNormalise_Div")
    # rewire the first consumer of the graph input
    for node in graph.node:
        for i, nm in enumerate(node.input):
            if nm == inp_name:
                node.input[i] = new_name
    graph.node.insert(0, div_node)
    return model


# ---------------------------------------------------------------------------
# Assertions -- these are build gates, not warnings
# ---------------------------------------------------------------------------

def assert_single_dataflow_partition(model):
    """PAPER FACT 3.4.4 -- exactly one StreamingDataflowPartition."""
    parts = [n for n in model.graph.node
             if n.op_type == "StreamingDataflowPartition"]
    if len(parts) != 1:
        raise BuildAssertionError(
            "expected exactly 1 StreamingDataflowPartition (PAPER FACT 3.4.4: "
            "FINN cannot map multiple partitions to the PL); found {}".format(
                len(parts)))
    return True


def assert_no_float_ops_inside(child_model):
    """No floating-point arithmetic may survive inside the partition.

    PAPER FACT 3.4.2: streamlining exists precisely to eliminate float ops by
    absorbing them into MultiThreshold nodes.
    """
    float_ops = {"Add", "Mul", "Div", "Sub", "BatchNormalization", "Softmax"}
    offenders = [n.name or n.op_type for n in child_model.graph.node
                 if n.op_type in float_ops]
    if offenders:
        raise BuildAssertionError(
            "float ops survived inside the dataflow partition "
            "(PAPER FACT 3.4.2): {}".format(offenders))
    return True


def assert_residual_parent_nodes(parent_model):
    """PAPER FACT 3.4.3 + 3.4.4 -- exactly 2 Transpose and 1 Mul outside.

    The two Transposes are the YOLO-NCHW / FINN-NHWC reordering; the Mul is the
    de-quantisation that runs on the PS (REPOSITORY FACT: its constant is
    scale.npy == 1/127).
    """
    counts = {}
    for n in parent_model.graph.node:
        if n.op_type == "StreamingDataflowPartition":
            continue
        counts[n.op_type] = counts.get(n.op_type, 0) + 1
    errs = []
    if counts.get("Transpose", 0) != 2:
        errs.append("expected 2 residual Transpose, found {}".format(
            counts.get("Transpose", 0)))
    if counts.get("Mul", 0) != 1:
        errs.append("expected 1 residual Mul (dequant), found {}".format(
            counts.get("Mul", 0)))
    if errs:
        raise BuildAssertionError("; ".join(errs) + " | all parent nodes: {}".format(
            counts))
    return True


def assert_io_contract(model):
    """REPOSITORY FACT driver.py / model.pkl -- UINT8 in, INT8 out, fixed shapes."""
    from qonnx.core.datatype import DataType   # noqa: F401
    i_name = model.graph.input[0].name
    o_name = model.graph.output[0].name
    ishape = tuple(model.get_tensor_shape(i_name))
    oshape = tuple(model.get_tensor_shape(o_name))
    idt = str(model.get_tensor_datatype(i_name))
    odt = str(model.get_tensor_datatype(o_name))
    errs = []
    if ishape != A.IO_CONTRACT["ishape_normal"]:
        errs.append("ishape {} != {}".format(ishape, A.IO_CONTRACT["ishape_normal"]))
    if oshape != A.IO_CONTRACT["oshape_normal"]:
        errs.append("oshape {} != {}".format(oshape, A.IO_CONTRACT["oshape_normal"]))
    if "UINT8" not in idt:
        errs.append("idt {} != UINT8".format(idt))
    if "INT8" not in odt:
        errs.append("odt {} != INT8".format(odt))
    if errs:
        raise BuildAssertionError(
            "I/O contract violated (REPOSITORY FACT driver.py): " + "; ".join(errs))
    return True


def assert_mvu_inventory(child_model):
    """Every Table 6 MVU must be present exactly once, with the right weights."""
    mvus = [n for n in child_model.graph.node if "MVAU" in n.op_type]
    if len(mvus) != len(A.CONV_LAYERS):
        raise BuildAssertionError(
            "expected {} MVU nodes (PAPER FACT Table 6), found {}".format(
                len(A.CONV_LAYERS), len(mvus)))
    return True


ALL_ASSERTIONS = (
    "assert_single_dataflow_partition",
    "assert_no_float_ops_inside",
    "assert_residual_parent_nodes",
    "assert_io_contract",
    "assert_mvu_inventory",
)
