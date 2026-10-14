"""Architecture structure tests. Stdlib only -- no torch, no FINN, no dataset."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arch_spec as A


def test_layer_counts():
    assert len(A.LAYERS) == 16            # PAPER FACT Table 1 rows
    assert len(A.CONV_LAYERS) == 10
    assert len(A.POOL_LAYERS) == 6


def test_shape_chain_consistent():
    assert A.shape_chain_is_consistent() == []


def test_input_and_output_shapes():
    assert (1, A.INPUT_RESOLUTION, A.INPUT_RESOLUTION, A.INPUT_CHANNELS) == \
        A.IO_CONTRACT["ishape_normal"]
    last = A.LAYERS[-1]
    assert (1, last.out_hw, last.out_hw, last.cout) == \
        A.IO_CONTRACT["oshape_normal"]


def test_decision_d2_maxpool_stride_one():
    """DECISION D2 / spec P2 -- ARITHMETICALLY FORCED."""
    l12 = A.LAYERS[11]
    assert l12.kind == "maxpool"
    assert l12.stride == 1, "D2: Table 1 prints 2, which is impossible"
    assert l12.in_hw == l12.out_hw == 13
    assert l12.pad == "same"


def test_d2_counterfactual_would_break_table6():
    """Stride 2 would make 13x13 -> 6x6 and layer 13's MAC count wrong."""
    l13 = A.LAYERS[12]
    assert l13.macs == 32_902_272                 # at 13x13, PAPER Table 6
    counterfactual = 6 * 6 * l13.k * l13.k * l13.cin * l13.cout
    assert counterfactual == 7_008_768
    assert counterfactual != A.TABLE6_REFERENCE["MVAU_hls_3"]["macs"]


def test_spec_p3_final_conv_is_3x3():
    l16 = A.LAYERS[15]
    assert l16.k == 3, "spec P3: 3x3, not stock YOLOv3-Tiny's 1x1"
    assert (l16.cin, l16.cout) == (104, 18)


def test_spec_p11_output_activation_is_int8():
    l16 = A.LAYERS[15]
    assert l16.a_out_bits == 8, "spec P11: ARITHMETICALLY FORCED by odt + 1/127"
    assert l16.act == "QuantHardTanh"
    assert A.OUTPUT_ACT_BITS == 8


def test_dequant_scale_forces_8_bit_output():
    """1/127 maps HardTanh [-1,1] onto INT8 [-127,127]; 4-bit would need 1/7."""
    assert abs(A.OUTPUT_DEQUANT_SCALE - 1.0 / 127.0) < 1e-12
    assert abs(A.OUTPUT_DEQUANT_SCALE * 127 - 1.0) < 1e-12
    assert abs(A.OUTPUT_DEQUANT_SCALE - 1.0 / 7.0) > 0.1


def test_first_and_last_conv_weights_are_int8():
    """PAPER FACT 3.1.1."""
    assert A.CONV_LAYERS[0].w_bits == 8
    assert A.CONV_LAYERS[-1].w_bits == 8
    for l in A.CONV_LAYERS[1:-1]:
        assert l.w_bits == 4


def test_hidden_activations_are_int4():
    for l in A.CONV_LAYERS[1:]:
        assert l.a_in_bits == 4
    assert A.CONV_LAYERS[0].a_in_bits == 8      # the UINT8 image


def test_rtl_hls_specialisation_from_table6():
    """PAPER FACT decoded from Table 6 instance names."""
    rtl = [l.idx for l in A.CONV_LAYERS if l.impl == "RTL"]
    hls = [l.idx for l in A.CONV_LAYERS if l.impl == "HLS"]
    assert rtl == [1, 3, 7, 15, 16]
    assert hls == [5, 9, 11, 13, 14]
    assert len(rtl) == len(hls) == 5


def test_every_mvu_name_is_unique_and_maps_to_one_layer():
    names = [l.mvu for l in A.CONV_LAYERS]
    assert len(names) == len(set(names)) == 10
    for name in names:
        assert A.layer_by_mvu(name).mvu == name


def test_dsp_pack_factors_follow_table4():
    """PAPER FACT Table 4: 4 MAC/DSP at 4b4b, 2 at >4b, 1 for HLS DSP style."""
    conv1 = A.CONV_LAYERS[0]      # 8b-8b, RTL
    assert conv1.dsp_pack_factor == A.DSP48E1_PACK_8B == 2
    conv2 = A.CONV_LAYERS[1]      # 4b-4b, RTL
    assert conv2.dsp_pack_factor == A.DSP48E1_PACK_4B == 4
    conv10 = A.CONV_LAYERS[-1]    # 4b-8b, RTL
    assert conv10.dsp_pack_factor == 2
    conv3 = A.CONV_LAYERS[2]      # HLS
    assert conv3.dsp_pack_factor == A.HLS_DSP_PACK == 1


def test_anchors_and_head_geometry():
    """REPOSITORY FACT infer_client_v3.py:25 + oshape."""
    assert A.ANCHORS_PX == ((10, 14), (23, 27), (37, 58))
    assert A.DETECT_STRIDE == 32
    assert A.NUM_ANCHORS * (A.NUM_CLASSES + 5) == A.OUTPUT_CHANNELS == 18


def test_hardware_target_decisions():
    assert A.FPGA_PART == "xc7z020clg400-1"       # DECISION D1
    assert A.VIVADO_VERSION == "2022.1"           # REPOSITORY FACT
    assert A.FINN_VERSION == "0.10"               # PAPER FACT
    assert A.FPGA_BOARD == "myir.com:mys-7z020:part0:2.1"
    assert A.TARGET_CLOCK_MHZ == 200.0
    assert (A.DEVICE_LUT, A.DEVICE_DSP48E1, A.DEVICE_BRAM18K) == (53200, 220, 140)


def test_weights_all_internal():
    """PAPER FACT 3.8 / REPOSITORY FACT number_of_external_weights == 0."""
    assert A.IO_CONTRACT["number_of_external_weights"] == 0
