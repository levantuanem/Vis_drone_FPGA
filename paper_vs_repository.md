# Paper vs. Author Repository — Evidence Base

**Paper:** Calì, Falaschetti, Biagetti, *"Optimized Implementation of YOLOv3-Tiny for Real-Time
Image and Video Recognition on FPGA"*, Electronics 2025, 14, 3993 (40 pp., MDPI, CC BY).
**Repository (ref [62]):** `https://github.com/sn0wst0rm/FINN-VisDrone-YOLO`
**Inspected at:** commit `c106c44` ("fix: typo and old client fix"), branch `master`, 3 commits total.
**Clone location:** scratchpad only (`.../scratchpad/ref_repo`) — deliberately *not* in the project tree,
so it can never be mistaken for our implementation.

> Status of this document: read-only inspection. Nothing installed, downloaded (beyond the repo
> itself), trained, synthesised, or executed.

---

## 1. Headline finding: the repository does not contain what the paper says it contains

The paper makes two explicit claims about the repository:

| Paper location | Claim | Reality |
|---|---|---|
| §3 (end) | "The complete code base, along with the **Jupyter notebooks used for training**, can be found in the repository [62]." | **False.** No notebooks. No training code. No model definition. No `export.py`. No `quant_common.py`. No dataset converter. |
| §3.7.2 | "Final full **hardware layer configurations, including all optimized FIFO depths**, can be found in the repository [62] and are not included in this document due to their length." | **False.** No folding JSON, no FIFO depth config, no FINN build script. Only a *compiled* bitstream whose internals are opaque. |
| Repo `README.md` | "This repository contains all the code to define, **train**, convert (to a suitable hardware representation) and then execute a YOLOv3-Tiny network…" | **False** for define/train/convert; true only for execute. |

**Actual repository content — complete file inventory (47 files, of which 20 are `.pyc`):**

```
README.md                                  (310 B, 3 lines)
finn_streamer/
  README.md                                usage doc for the streamer only
  scale.npy                                float32 scalar = 0.00787402  (= 1/127)
  client/
    infer_client_v3.py        586 LOC       4-thread WebSocket client
    models.py                  57 LOC       Detect head (decode) only
    saved_dict.pkl                          io_shape_dict copy
    driver/qonnx/{core,util}                vendored QONNX DataType stubs
    test_package/
      model.bit / model.hwh / model.pkl / model_info.json
      yolov3-tiny-4w4a-visdrone.fnn         zip of the 4 files above
  fpga/
    infer_server.py           376 LOC       WebSocket inference server (PS side)
    bitfile/visdrone.bit, visdrone.hwh
    driver/
      driver_base.py          491 LOC       stock FINN PYNQ driver (Xilinx 2020 BSD header)
      driver.py               159 LOC       stock FINN generated entry point
      validate.py             107 LOC       stock FINN MNIST/CIFAR10 example — UNMODIFIED, irrelevant
      resizer.bit, resizer.hwh              a DIFFERENT, older build (see §4)
      nw_metrics.txt                        one stale throughput measurement (see §4)
      finn/util/data_packing.py             vendored FINN helper
      qonnx/{core,util}                     vendored QONNX stubs
```

**Consequence for us:** the repository is a *deployment-and-streaming reference*, not a
reproduction package. Experiments T1.1–T1.11 (dataset, training, quantisation, FINN compilation,
folding, FIFO sizing, synthesis, power estimation) have **no code to reference at all** and must be
built from the paper text. This is consistent with our stated objective — reimplement from the
paper — but it removes the fallback of "check what they did" for the whole front half of the flow.

---

## 2. What the repository *does* give us that the paper does not

These are genuine, load-bearing recoveries. All are **REPOSITORY-REFERENCE**.

### 2.1 Accelerator I/O contract (`fpga/driver/driver.py`, `model.pkl`)
```python
idt  = UINT8      ishape_normal = (1, 416, 416, 3)    ishape_folded = (1, 416, 416, 1, 3)
odt  = INT8       oshape_normal = (1, 13,  13,  18)   oshape_folded = (1, 13,  13, 18, 1)
input_dma_name = ['idma0']   output_dma_name = ['odma0']
number_of_external_weights = 0        # confirms §3.8: all weights in BRAM
```
- Input is **UINT8 NHWC**, un-normalised → corroborates the inserted `Div` node (§3.4.1).
- `ishape_folded` last dim `3` ⇒ **input SIMD = 3** (first layer fully unfolded on input channels).
- `oshape_folded` last dim `1` ⇒ **output PE = 1** (final layer left folded), consistent with §3.6.1
  ("smaller layers … kept fully folded").
- `number_of_external_weights = 0` is hard confirmation of the all-BRAM weight claim.

### 2.2 Dequantisation scale (`scale.npy`)
`0.00787402 = 1/127`. This is the trailing `Mul` node that §3.4.4 says stays on the PS. The paper
never gives the value.

### 2.3 Anchors and detection decode (`client/infer_client_v3.py:25`, `client/models.py`)
```python
anchors = np.array([[10,14, 23,27, 37,58]]) / np.array([32])   # 3 anchors, stride 32
nc = 1 ;  no = nc + 5 = 6 ;  na = 3   →   3 × 6 = 18 output channels  ✓ matches (13,13,18)
```
- The paper never states the anchors. These are YOLOv3-Tiny's **small-object** anchor set, which in
  stock YOLOv3-Tiny belongs to the *stride-16* branch — here reused at **stride 32**. A deliberate
  choice for small aerial targets, entirely undocumented in the paper.
- The decode in `models.py` is **non-standard YOLO**, adapted to the `QuantHardTanh` output range:
  ```python
  xy      = (y[..., 0:2] + 0.5 + grid) * stride
  wh      = (y[..., 2:4] + 1.0) ** 2   * anchor_grid
  obj/cls =  y[..., 4:] / 2 + 0.5
  ```
  Stock YOLOv3 uses `sigmoid(xy)+grid`, `exp(wh)*anchor`, `sigmoid(obj)`. These affine/quadratic
  substitutions map a `[-1,1]`-bounded HardTanh output into the correct domains. **This is essential
  to reproduce and is nowhere in the paper.** Getting it wrong silently destroys all detections.

### 2.4 Post-processing thresholds (`infer_client_v3.py:342`)
`non_max_suppression(conf_thres=0.10, iou_thres=0.10, classes=None, max_det=20)` — not in the paper.
Note these are unusually permissive thresholds; they inflate apparent recall in the Fig. 19 field
demonstration relative to the Table 5 validation numbers.

### 2.5 Pre-processing (`infer_client_v3.py:236-243`)
`letterbox(frame, (416,416), auto=False)` then `[:, :, ::-1]` BGR→RGB, kept **uint8**. The paper
says only "resizing and color channel inversion" (§3.11.7); letterboxing (aspect-preserving pad) vs.
plain resize materially changes mAP.

### 2.6 Build provenance (`visdrone.hwh` XML header)
```xml
<EDKSYSTEM VIVADOVERSION="2022.1" TIMESTAMP="Sat Feb 15 16:53:39 2025">
<SYSTEMINFO ARCH="zynq" BOARD="myir.com:mys-7z020:part0:2.1"
            DEVICE="7z020" PACKAGE="clg400" SPEEDGRADE="-1"/>
```
Three corrections to assumptions:
1. **Vivado/Vitis 2022.1**, not 2022.2. FINN pins its tool version strictly; we must match 2022.1.
2. Board file is **`myir.com:mys-7z020:part0:2.1`** — a vendor MYIR board definition, not a stock
   PYNQ/Digilent platform. We must obtain or author this board file.
3. **`SPEEDGRADE="-1"`**, while §3 of the paper specifies `XC7Z020CLG400-**2**`. The synthesised
   design was closed on the *slower* speed grade. Timing results (Fig. 17) therefore correspond to
   `-1`. We must pick one and record it; `-1` is the conservative, evidence-backed choice.
4. `rst_zynq_ps_200M` + `PCW_UIPARAM_DDR_FREQ_MHZ = 533.333333`, AXI HP data width 64 — consistent
   with §3's description of the SoC.

### 2.7 Per-layer RTL/HLS specialisation is recoverable — from the *paper*, not the repo
Paper **Table 6** lists MVU instance names. Cross-checking each weight count against Table 1's
channel dimensions confirms the mapping exactly (`3·3·Cin·Cout`):

| # | Conv (Table 1) | Weights | Check | Table 6 instance | Style |
|---|---|---|---|---|---|
| 1 | 3→8, 3×3 | 216 | 3·3·3·8 ✓ | `MVAU_rtl_0` | **RTL** |
| 2 | 8→8, 3×3 | 576 | 3·3·8·8 ✓ | `MVAU_rtl_1` | **RTL** |
| 3 | 8→16, 3×3 | 1 152 | 3·3·8·16 ✓ | `MVAU_hls_0` | **HLS** |
| 4 | 16→32, 3×3 | 4 608 | 3·3·16·32 ✓ | `MVAU_rtl_2` | **RTL** |
| 5 | 32→56, 3×3 | 16 128 | 3·3·32·56 ✓ | `MVAU_hls_1` | **HLS** |
| 6 | 56→104, 3×3 | 52 416 | 3·3·56·104 ✓ | `MVAU_hls_2` | **HLS** |
| 7 | 104→208, 3×3 | 194 688 | 3·3·104·208 ✓ | `MVAU_hls_3` | **HLS** |
| 8 | 208→56, 1×1 | 11 648 | 1·1·208·56 ✓ | `MVAU_hls_4` | **HLS** |
| 9 | 56→104, 3×3 | 52 416 | 3·3·56·104 ✓ | `MVAU_rtl_3` | **RTL** |
| 10 | 104→18, 3×3 | 16 848 | 3·3·104·18 ✓ | `MVAU_rtl_4` | **RTL** |

**5 RTL + 5 HLS.** So the specialisation half of §3.6.2 *is* paper-specified after all — it just has
to be decoded from Table 6's instance names. Only **PE/SIMD values and FIFO depths remain unknown.**

---

## 3. Internal inconsistencies in the paper itself

These must be resolved by us before implementation; we cannot satisfy both readings.

| # | Issue | Evidence | Resolution we propose |
|---|---|---|---|
| **P1** | §3.1 says channel counts were "reduced by a factor of 5". Table 1 shows factors of 2–4.9: 16→8 (÷2), 32→8 (÷4), 64→16 (÷4), 128→32 (÷4), 256→56 (÷4.57), 512→104 (÷4.92), 1024→208 (÷4.92). | Table 1 vs. §3.1 prose | **Table 1 is authoritative** (independently corroborated by Table 6 weight counts). Treat "factor of 5" as loose prose. |
| **P2** | Table 1 row 12: `MaxPooling 13×13×104 → 13×13×104, kernel 2×2, **stride 2**`. A stride-2 pool cannot preserve 13×13. | Table 1 | Stock YOLOv3-Tiny uses **stride 1 with padding** at this position. Use stride 1; shape then matches. Paper's "stride 2" is a typo. |
| **P3** | Final detection conv is **3×3** (Table 1, and 16 848 = 3·3·104·18 in Table 6). Stock YOLOv3-Tiny uses 1×1. | Tables 1 & 6 agree | Deviation is real and consistent. **Use 3×3.** |
| **P4** | Abstract & §6 claim "**three to seven**×" FPS/W improvement; §5.2 claims "**7 to 24 times**" and "7.3× over LPYOLO". | Abstract vs. §5.2 vs. Table 16 | Report both; do not target either. Out of scope at this stage. |
| **P5** | §3 specifies speed grade **-2**; the shipped build used **-1**. | §3 vs. `visdrone.hwh` | Build `-1` (matches the evidence); record the choice. |
| **P6** | Table 5 is labelled "Validation results" but §4.1 text says precision/recall "on the **training** set". | Table 5 caption vs. §4.1 | Evaluate on the **val** split and label it unambiguously. |
| **P7** | §3.1 says upsample *and* concat layers were removed, i.e. one detection branch. Table 1 confirms a single 13×13 head. Table 5 attributes halved recall to this. | consistent | No conflict — noted for completeness. |

---

## 4. The repository's own numbers do not corroborate the paper

`fpga/driver/nw_metrics.txt` is the only committed measurement in the repo:

```json
{"runtime[ms]": 425.649, "throughput[images/s]": 23.4935,
 "DRAM_in_bandwidth[MB/s]": 12.197, "DRAM_out_bandwidth[MB/s]": 0.0715,
 "fclk[mhz]": 100.0, "batch_size": 10,
 "fold_input[ms]": 0.142, "pack_input[ms]": 0.145,
 "copy_input_data_to_device[ms]": 36.421, "copy_output_data_from_device[ms]": 0.535,
 "unpack_output[ms]": 1.704, "unfold_output[ms]": 0.213}
```

- **23.49 FPS at 100 MHz** vs. paper Table 9's **104.39 FPS** at 100 MHz (batch 100) and 86.47 FPS
  (batch 1). A ~4.4× shortfall.
- Likely explanation, not proof: `driver.py`'s default is `--bitfile resizer.bit`, and
  `md5sum` shows **`resizer.bit` is a different build** from `visdrone.bit`:
  ```
  72929bb6…  fpga/bitfile/visdrone.bit   ==  client/test_package/model.bit
  381e995f…  fpga/driver/resizer.bit         <-- different artifact
  ```
  So this file is plausibly a stale measurement from an earlier, unoptimised build. We cannot verify
  which bitstream produced it.
- **Takeaway:** the repository contains **no artifact that corroborates Table 9, 10, 13 or 14.** Our
  Tier-2 measurements will be the only evidence we have, and we should not expect the repo to
  arbitrate disagreements.

Also note what `throughput_test()` actually measures (`driver_base.py:421-448`):
```python
start = time.time(); self.execute_on_buffers(); end = time.time()
res["throughput[images/s]"] = self.batch_size / (end - start)
```
It times **only `execute_on_buffers()`** — buffers are already resident in DRAM. `fold_input`,
`pack_input`, `copy_*`, `unpack_output` are timed *separately* and excluded. Therefore paper
**Table 9's FPS column is a fabric+DMA figure that excludes host-side packing**, while Table 10 is
the end-to-end Python-driver breakdown. Our T2.1 must use this identical definition or the numbers
are not comparable. This is the single most important methodological detail recovered from the repo.

---

## 5. Defects and loose ends in the repository code

Relevant because we must not inherit them.

| Item | Detail |
|---|---|
| **`utils` module is missing** | `client/models.py:4` and `infer_client_v3.py:20` import `clip_coords, scale_coords, letterbox, xywh2xyxy, non_max_suppression, check_version, visualize_boxes` from a `utils` module that **is not in the repository**. The client cannot run as shipped. These are Ultralytics YOLOv5 helpers; we will write our own. |
| **`names = ['face']`** | `infer_client_v3.py:23`. Leftover from LPYOLO [63], which is a *face detection* paper. Confirms the lineage (paper §3.1.1 cites Okcu's quantised YOLOv5 fork, ref [64]) and that the class label was never updated to `car`. Cosmetic, but it is direct evidence the client was not re-validated for this work. |
| **`validate.py` is stock FINN boilerplate** | Hard-codes MNIST/CIFAR-10 and top-1 classification accuracy. **There is no hardware-side detection-accuracy or bit-exactness validation anywhere in the repository.** Nothing demonstrates that the FPGA output matches the PyTorch model. We are adding that as experiment **T1.12**. |
| **No Z-turn board files, no PYNQ image recipe** | Despite the board being non-standard. |
| **Comments in Italian**, `infer_client_v3` suffix, commented-out `time.sleep` tuning | Prototype-grade; fine as a reference, not as a base. |
| **No licence file** | Only the Xilinx BSD header inside the vendored FINN driver. If we ever vendored anything we would need to clarify; our plan avoids this by reimplementing. |

---

## 6. Summary scorecard

| Flow stage | Paper | Repository | Verdict |
|---|---|---|---|
| Dataset prep (VisDrone→YOLO, car-only) | described, no code | **absent** | reimplement |
| Model definition / Brevitas quant layers | Table 1 + §3.1.1 prose | **absent** | reimplement from Table 1 |
| Training recipe | Tables 2 & 3 | **absent** | reimplement |
| Anchors | **absent** | **present** (§2.3) | take from repo |
| Detect decode | "scaling operations" only | **present** (§2.3) | take from repo |
| QONNX export | CLI invocation shown | **absent** | reimplement |
| FINN front-end transforms | §3.4 named transforms | **absent** | reimplement |
| RTL/HLS specialisation | **recoverable from Table 6** (§2.7) | bitstream only, opaque | derive from paper |
| PE/SIMD folding | method only; Fig. 10/11 targets | **absent / opaque** | **re-derive — see plan T1.6** |
| FIFO depths | method only | **absent / opaque** | **re-derive — see plan T1.7** |
| Synthesis / ZynqBuild | §3.9-3.10 named transforms | bitstream only | reimplement |
| I/O contract, dequant scale | partial | **present** (§2.1, §2.2) | take from repo |
| PYNQ driver | §3.10.2 | stock FINN, present | regenerate ourselves |
| Streaming server/client | §3.11 detailed | **present but incomplete** (§5) | reimplement, reference design |
| Any measurement | Tables 7-15 | one stale, unmatched file | reimplement |

**Bottom line:** roughly 15 % of the pipeline exists as reusable reference; ~85 % must be written
from the paper. The two values the paper explicitly delegates to the repository — folding
configuration and FIFO depths — are **the two things the repository does not contain.**
