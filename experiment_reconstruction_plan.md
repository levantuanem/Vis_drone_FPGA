# Experiment Reconstruction Plan

**Target paper:** Calì, Falaschetti, Biagetti, *"Optimized Implementation of YOLOv3-Tiny for
Real-Time Image and Video Recognition on FPGA"*, Electronics **2025**, 14, 3993.

**Objective.** Build **our own** implementation of the methodology, experiment by experiment, using
the paper as the primary specification. The author repository (`sn0wst0rm/FINN-VisDrone-YOLO`) is a
**reference only**, consulted where the paper is silent. We are *not* re-running the authors' code,
and we are *not* tuning toward their reported numbers at this stage.

**Status.** Design only. Nothing installed, downloaded, trained, synthesised, simulated, or built.

### Legend — applied to every field of every experiment

| Tag | Meaning |
|---|---|
| **[PAPER]** | Explicitly specified in the paper. Authoritative. |
| **[REPO]** | Not in the paper; recovered from the author repository. Cite file:line. |
| **[OURS]** | We design/implement this. No upstream equivalent exists. |
| **[UNSPEC]** | Specified nowhere. A decision we must make and record. **No value invented here.** |

Companion documents: `paper_vs_repository.md` (evidence base, discrepancy log),
`required_implementation.md` (file-by-file build list).

---

## 0. Master mapping table

| ID | Paper § | Experiment | Tier | Required implementation | Key output |
|---|---|---|---|---|---|
| **T1.1** | §3.1.2 | VisDrone → YOLO car-only dataset | 1 | converter + split manifests | `visdrone_car/` + yaml |
| **T1.2** | §4.1 Table 5 (left) | Unpruned YOLOv3-Tiny FP32 baseline | 1 | model def + train loop | P/R/mAP baseline |
| **T1.3** | §3.1, §4.1 Table 5 (right), Fig. 15 | Pruned + 4W4A QAT training | 1 | Brevitas model + QAT train | P/R/mAP + curves |
| **T1.4** | §3.2, Fig. 3 | QONNX export (detect head detached) | 1 | export module | `*.onnx` |
| **T1.5** | §3.4, Figs. 4–7 | FINN front-end transforms | 1 | build step 1 script | FINN-ONNX graphs |
| **T1.6** | §3.5–3.6, Figs. 8, 10, 11 | Specialisation + folding search | 1 | folding search driver | `folding.json` + cycle report |
| **T1.7** | §3.7 | FIFO depth sizing + MaxPool anomaly | 1 | FIFO sizing step | FIFO config + anomaly note |
| **T1.8** | §4.2.3 Table 6, §4.2.5 Table 8 (est.) | Analytical MAC / cycle / throughput model | 1 | static analysis script | Table 6 + est. column |
| **T1.9** | §3.9–3.10, §4.2.1, §4.2.4, Figs. 12, 13, 16, 17, Table 7 | HLS synth → stitch → ZynqBuild | 1 | build step 2 script | bitstream + util/timing |
| **T1.10** | §4.2.5 Table 8 (RTLSim) | RTL simulation throughput | 1 | rtlsim harness | RTLSim FPS |
| **T1.11** | §4.3.1 Tables 11, 12 | Vivado power estimation | 1 | report parser | power breakdown |
| **T1.12** | — (**gap we close**) | Functional equivalence PyTorch ↔ FINN | 1 | equivalence harness | bit-exactness + mAP-on-sim |
| **T1.13** | §3.3 | FP32 → TFLite INT8 PTQ → Edge TPU compile | 1 | export + compile script | `*_edgetpu.tflite` |
| **T2.1** | §4.2.5 Table 8 (actual), §4.2.6 Table 9 | On-board throughput sweep (freq × batch) | 2 | deploy + bench script | Table 9 equivalent |
| **T2.2** | §4.2.6 Table 10 | Per-stage latency breakdown | 2 | instrumented pipeline | Table 10 equivalent |
| **T2.3** | §4.3.2 Tables 13, 14, Fig. 18 | Measured power + thermal | 2 | bench protocol | W, FPS/W, °C |
| **T2.4** | §3.11, §4.4 | End-to-end WebSocket streaming | 2 | server + client | e2e FPS, latency |
| **T2.5** | §4.5 Fig. 19 | Field footage inference | 2 | offline video runner | annotated frames |
| **T3.1** | §5.1 Table 15 | Coral Edge TPU benchmark | 3 | Coral bench script | latency, FPS, FPS/W |
| **T3.2** | §5.2 Table 16 | Literature comparison | 3 (desk) | table compiler | comparison table |

**Reachability on this host:** T1.1–T1.13 (13 of 20). T2.* needs the Zynq board; T3.1 needs a Coral.

---

# TIER 1 — Reconstructable on the host machine

Host: 1× TITAN RTX 24 GB, 6 cores, **15 GB RAM**, `/mnt/data10tb` (2.9 TB free). See §"Host risks".

---

## T1.1 — VisDrone → YOLO single-class dataset

1. **Research objective.** Produce a reproducible, versioned `car`-only detection dataset from
   VisDrone2019-DET, in Ultralytics YOLO format, so every downstream experiment trains on identical
   data.
2. **Input.** VisDrone2019-DET archives (train / val / test-dev).
3. **Dataset.** VisDrone [65] **[PAPER]**: "10,209 images annotated with 2.6 million bounding boxes
   across 10 object categories"; **Task 1 only** (object detection in images) **[PAPER]**; **only
   the `car` category retained** **[PAPER]**; annotations converted to **YOLO XYWH** **[PAPER]**.
4. **Model.** N/A.
5. **Model configuration.** N/A.
6. **Quantization method.** N/A.
7. **Software pipeline.** `tools/visdrone_to_yolo.py`: parse VisDrone `annotations/*.txt`
   (`x,y,w,h,score,category,truncation,occlusion`) → filter category → normalise to
   `class cx cy w h` → write `labels/`, symlink `images/` → emit `data/visdrone_car.yaml` →
   emit a `manifest.json` with per-split image/box counts and a content hash.
8. **Hardware pipeline.** N/A.
9. **Configuration parameters.**
   - `nc: 1`, `names: ['car']` **[PAPER]** (class choice), `nc=1` corroborated **[REPO]**
     (`infer_client_v3.py:24`).
   - VisDrone `car` category id **[UNSPEC in paper]** — VisDrone's documented ordering is
     `0 ignored, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle,
     8 awning-tricycle, 9 bus, 10 motor`. We must **verify against the downloaded annotation files**,
     not assume. Decision recorded in `manifest.json`.
   - Handling of `score=0` (ignored-region) boxes, `truncation`, `occlusion` flags: **[UNSPEC]**.
   - Whether `van`/`truck`/`bus` are merged into `car`: **[UNSPEC]** — the paper says "car objects",
     singular category. **Our decision: do not merge.** Record it.
   - Split used for the reported metrics: **[UNSPEC/contradictory]** — see discrepancy **P6**.
     **Our decision: evaluate on `val`.**
10. **Expected output.** `visdrone_car/{train,val,test}/`; `data/visdrone_car.yaml`; `manifest.json`.
11. **Evaluation metrics.** Sanity only: image count per split, retained-box count, boxes/image
    distribution, zero-label-file count, box-size histogram (to justify the anchor choice in T1.3).
12. **[PAPER]** dataset identity, task, single class, XYWH target format, 10 209 / 2.6 M figures.
13. **[REPO]** `nc=1` only.
14. **[UNSPEC]** category index verification, ignored-region policy, merge policy, split policy,
    letterbox-vs-resize at train time (see T1.3).

- **Required files:** `tools/visdrone_to_yolo.py`, `data/visdrone_car.yaml`
- **Required dependencies:** Python 3.10, numpy, Pillow, tqdm, PyYAML
- **Required hardware:** none (CPU, ~3 GB disk on `/mnt/data10tb`)
- **Expected measurements:** dataset statistics table (ours; the paper reports none)

---

## T1.2 — Unpruned YOLOv3-Tiny FP32 baseline

1. **Research objective.** Establish the reference accuracy that the pruning + quantisation cost is
   measured against, i.e. Table 5's left column.
2. **Input.** T1.1 dataset.
3. **Dataset.** `visdrone_car`, train split; evaluate on val.
4. **Model.** Stock YOLOv3-Tiny **[PAPER]** — full channel widths, **both** detection branches
   (13×13 and 26×26), upsample + concat intact.
5. **Model configuration.** Standard YOLOv3-Tiny at 416×416 **[PAPER]**, `nc=1`.
6. **Quantization method.** None (FP32) — baseline by construction.
7. **Software pipeline.** `train.py --cfg models/yolov3-tiny.yaml --data data/visdrone_car.yaml`.
8. **Hardware pipeline.** N/A (GPU training only).
9. **Configuration parameters.** Table 3 **[PAPER]**: 300 epochs, batch 64, 416×416, SGD,
   patience 30. Learning-rate schedule, momentum, weight decay, warmup, augmentation: **[UNSPEC]** —
   §3.1.4 says only "standard low-scratch hyper-parameters", which points at Ultralytics'
   `hyp.scratch-low.yaml`. **Our decision: adopt that file verbatim and version it in-repo.**
10. **Expected output.** `runs/baseline/weights/best.pt`, results CSV, PR curve.
11. **Evaluation metrics.** P, R, mAP@0.5, box loss, object loss, classification loss — the exact
    rows of Table 5 **[PAPER]**. Paper reference values: P 67.06 %, R 38.44 %, mAP@0.5 41.76 %,
    box 0.1220, obj 0.1653, cls 0 **[PAPER]** — recorded as *context*, not a target.
12. **[PAPER]** architecture, image size, all of Table 3, the metric set, the reference values.
13. **[REPO]** nothing (no training code exists upstream).
14. **[UNSPEC]** full hyper-parameter file, augmentation policy, anchors for the 2-branch baseline
    (stock YOLOv3-Tiny anchors are the natural choice — record it), seed, mAP implementation
    (we will fix one and use it for *both* T1.2 and T1.3 so the comparison is internally valid).

- **Required files:** `models/yolov3_tiny.py`, `models/yolov3-tiny.yaml`, `train.py`, `val.py`, `utils/*`
- **Required dependencies:** torch 1.13.1 + torchvision 0.14.1 **[PAPER Table 2]**, plus all of Table 2
- **Required hardware:** 1 GPU (TITAN RTX 24 GB ≥ the paper's P100 16 GB **[PAPER §3.1.3]**)
- **Expected measurements:** Table 5 left column, independently produced

---

## T1.3 — Pruned + 4W4A quantisation-aware training  ← **core training experiment**

1. **Research objective.** Train the hardware-synthesisable network: the FINN-compatible,
   width-reduced, single-branch YOLOv3-Tiny with mixed INT8/INT4 QAT. Quantify the accuracy cost
   versus T1.2.
2. **Input.** T1.1 dataset.
3. **Dataset.** `visdrone_car`.
4. **Model.** **Table 1 is the authoritative specification** **[PAPER]** (see discrepancy **P1**:
   the "factor of 5" prose in §3.1 disagrees with Table 1; Table 1 is independently corroborated by
   Table 6's weight counts — `paper_vs_repository.md` §2.7):

   | # | Layer | In | Out | K | S | Act |
   |---|---|---|---|---|---|---|
   | 1 | QuantConv | 416×416×3 | 416×416×8 | 3×3 | 1 | QuantReLU |
   | 2 | MaxPool | 416×416×8 | 208×208×8 | 2×2 | 2 | — |
   | 3 | QuantConv | 208×208×8 | 208×208×8 | 3×3 | 1 | QuantReLU |
   | 4 | MaxPool | 208×208×8 | 104×104×8 | 2×2 | 2 | — |
   | 5 | QuantConv | 104×104×8 | 104×104×16 | 3×3 | 1 | QuantReLU |
   | 6 | MaxPool | 104×104×16 | 52×52×16 | 2×2 | 2 | — |
   | 7 | QuantConv | 52×52×16 | 52×52×32 | 3×3 | 1 | QuantReLU |
   | 8 | MaxPool | 52×52×32 | 26×26×32 | 2×2 | 2 | — |
   | 9 | QuantConv | 26×26×32 | 26×26×56 | 3×3 | 1 | QuantReLU |
   | 10 | MaxPool | 26×26×56 | 13×13×56 | 2×2 | 2 | — |
   | 11 | QuantConv | 13×13×56 | 13×13×104 | 3×3 | 1 | QuantReLU |
   | 12 | MaxPool | 13×13×104 | 13×13×104 | 2×2 | **1** ⚠ | — |
   | 13 | QuantConv | 13×13×104 | 13×13×208 | 3×3 | 1 | QuantReLU |
   | 14 | QuantConv | 13×13×208 | 13×13×56 | 1×1 | 1 | QuantReLU |
   | 15 | QuantConv | 13×13×56 | 13×13×104 | 3×3 | 1 | QuantReLU |
   | 16 | QuantConv | 13×13×104 | 13×13×18 | 3×3 | 1 | QuantHardTanh |

   ⚠ Row 12: Table 1 prints **stride 2** but input and output are both 13×13, which is impossible.
   Discrepancy **P2**. **Our decision: stride 1 with padding** (stock YOLOv3-Tiny behaviour at this
   position), which makes the printed shapes consistent.
   Row 16 is **3×3**, not the stock 1×1 — discrepancy **P3**; Table 6's 16 848 = 3·3·104·18 confirms
   3×3 is intended. **Our decision: 3×3.**
   Structural removals **[PAPER §3.1]**: detect layer detached at export; **all upsample and concat
   layers removed** (single 13×13 head) because FINN cannot map branched topologies; widths reduced.
5. **Model configuration.** 416×416×3 input **[PAPER]**; `nc=1`; head produces 13×13×18.
   **Anchors [REPO]** (`infer_client_v3.py:25`): `(10,14), (23,27), (37,58)`, applied at **stride 32**
   → `na=3`, `no=nc+5=6`, `3×6=18` ✓ consistent with the 18-channel head. Paper never states these.
   **Detect decode [REPO]** (`client/models.py`) — non-standard, matched to the HardTanh output range:
   `xy=(y+0.5+grid)*stride`, `wh=(y+1.0)²·anchor_grid`, `obj/cls=y/2+0.5`. **Must be used at training
   time too**, or the loss is computed against a different parameterisation than inference uses.
6. **Quantization method.** QAT with **Brevitas 0.9.0** **[PAPER Table 2]**, integrated into a
   modified Ultralytics YOLOv5 training harness **[PAPER §3.1.1]** (`common.py`, `train.py` modified,
   new `quant_common.py`). Bit allocation **[PAPER §3.1.1]**:
   - **first and last convolution weights: INT8**
   - **all other weights and all activations: INT4**
   - Activations `QuantReLU` (layers 1–15) and `QuantHardTanh` (layer 16) **[PAPER Table 1]**
   - Input tensor is UINT8, normalised on-chip by an inserted `Div` **[PAPER §3.4.1 / REPO idt=UINT8]**
   **[UNSPEC]:** per-tensor vs. per-channel scaling; signed/unsigned activation quantisers;
   scaling-factor parameterisation (power-of-two vs. float); `bit_width` of the *input* quantiser;
   whether BatchNorm is present and folded (Table 1 lists no BN, but §"Abbreviations" defines BN);
   restore/init strategy; rounding mode ("configurable rounding modes" is mentioned, no value given).
   **Our decision: adopt Brevitas defaults for `Int8WeightPerTensorFloat`-family quantisers,
   document every choice in `configs/quant.yaml`, and treat these as first-class variables.**
7. **Software pipeline.** `train.py --cfg models/yolov3-tiny-pruned-4w4a.yaml --quant configs/quant.yaml`
   → QAT 300 epochs → `best.pt`. Logging to TensorBoard **[PAPER Table 2: tensorboard 2.15.2]**.
8. **Hardware pipeline.** N/A.
9. **Configuration parameters.** Table 3 **[PAPER]**: 300 epochs, batch 64, 416×416, SGD, patience 30.
   Package versions pinned by **Table 2 [PAPER]** (torch 1.13.1, brevitas 0.9.0, numpy 1.26.3,
   opencv-python 4.10.0.84, pillow 11.1.0, PyYAML 6.0.2, requests 2.32.3, scipy 1.15.0, tqdm 4.67.1,
   matplotlib 3.5.1, tensorboard 2.15.2, pandas 2.2.3, seaborn 0.13.2, thop 0.1.1.post2209072238).
   **[UNSPEC]** the same hyper-parameter gaps as T1.2, plus: QAT from scratch vs. FP32 warm start
   (§3.1.4 implies from scratch — "standard low-scratch"), loss weights for a 1-class 1-branch head.
10. **Expected output.** `runs/pruned_4w4a/weights/best.pt`; results CSV; Fig. 15(a) training curves;
    Fig. 15(b) PR curve.
11. **Evaluation metrics.** P, R, mAP@0.5, box/obj/cls loss. Paper reference **[PAPER Table 5]**:
    P 48.16 %, R 17.16 %, mAP@0.5 17.68 %, box 0.1531, obj 0.3049, cls 0. Also reproduce the paper's
    *interpretation* check: recall should roughly halve versus T1.2, attributed to the removed branch.
12. **[PAPER]** Table 1 architecture, the INT8/INT4 split, activation types, structural removals,
    Brevitas as the QAT tool, Tables 2 and 3, the metric set and reference values, the
    halved-recall explanation.
13. **[REPO]** anchors, stride-32 assignment, the Detect decode formulas, `nc=1`.
14. **[UNSPEC]** every quantiser detail in field 6; full hyper-parameter file; BN presence/folding;
    seed; augmentation; letterbox-vs-resize for training (inference letterboxes **[REPO]**, so we
    letterbox in training too for consistency — recorded as our decision).

- **Required files:** `models/quant_yolov3_tiny.py`, `models/quant_common.py`,
  `models/yolov3-tiny-pruned-4w4a.yaml`, `configs/quant.yaml`, `train.py`, `val.py`, `utils/*`
- **Required dependencies:** all of Table 2 **[PAPER]**
- **Required hardware:** 1 GPU. Note the paper used a P100 **16 GB** at batch 64/416² **[PAPER]**,
  so 24 GB is sufficient headroom.
- **Expected measurements:** Table 5 right column + Fig. 15, independently produced

---

## T1.4 — QONNX export

1. **Research objective.** Emit a FINN-ingestible quantised ONNX graph from the trained Brevitas
   model, with the detection head removed.
2. **Input.** `runs/pruned_4w4a/weights/best.pt`.
3. **Dataset.** N/A (one dummy 416×416×3 tensor for tracing).
4. **Model.** T1.3 network.
5. **Model configuration.** As T1.3.
6. **Quantization method.** Brevitas QONNX export (`BrevitasONNXManager` / `export_qonnx`).
7. **Software pipeline.** **[PAPER §3.2]** gives the exact CLI:
   ```
   python export.py --weights /path/to/train_run/weights/best.pt --img 416 --include finn_onnx
   ```
   The **detect layer is detached during export [PAPER §3.2, §3.1]** because its operations are
   coordinate-space scaling, better done in PS floating point.
8. **Hardware pipeline.** N/A.
9. **Configuration parameters.** `--img 416` **[PAPER]**; opset, `export_qonnx` vs. deprecated
   `FINNManager`: **[UNSPEC]**.
10. **Expected output.** `build/model_qonnx.onnx`; a Netron rendering equivalent to **Fig. 3**
    **[PAPER]** ("Quant layers fed into MaxPool or Conv layers").
11. **Evaluation metrics.** Graph validity (`qonnx` checker); node-type inventory matching Fig. 3;
    **output tensor shape must be (1,13,13,18)** or (1,18,13,13) pre-transpose **[REPO oshape]**.
12. **[PAPER]** the CLI, `--img 416`, detect detachment and its rationale, expected graph shape.
13. **[REPO]** expected output shape and dtype (INT8).
14. **[UNSPEC]** opset version, export API variant, whether the trailing transpose is emitted here
    or later.

- **Required files:** `export.py`
- **Required dependencies:** brevitas 0.9.0, onnx, qonnx
- **Required hardware:** CPU
- **Expected measurements:** Fig. 3 equivalent

---

## T1.5 — FINN front-end transformations

1. **Research objective.** Reproduce the paper's transformation chain from QONNX to a single
   hardware-mappable dataflow partition, and the four graph states it illustrates.
2. **Input.** `build/model_qonnx.onnx`.
3. **Dataset.** N/A.
4. **Model.** As exported.
5. **Model configuration.** N/A.
6. **Quantization method.** Quantisation is already baked in; `Quant` nodes are absorbed here.
7. **Software pipeline.** **[PAPER §3.4]**, named transforms, in order:
   1. `ConvertQONNXtoFINN` — folds `Quant` into `MaxPool`/`Conv`, turns activations into
      **`MultiThreshold`** nodes **[PAPER §3.4.1]**.
   2. Tidy-up passes — infer shapes/datatypes, give every node a unique name (needed later for
      folding/specialisation) **[PAPER §3.4.1]**.
   3. **Insert a `Div` node before the first convolution** to normalise UINT8 input to [0,1]
      **[PAPER §3.4.1]** → **Fig. 4**.
   4. `Streamline` — remove float ops, absorb them into `MultiThreshold`; **lower Conv → `Im2Col` +
      `MatMul`** **[PAPER §3.4.2]** → **Fig. 5**.
   5. Hardware mapping — `MatMul` → **MVU**, `Im2Col` → **SWG**. **Two `Transpose` nodes remain**,
      because YOLO is NCHW and FINN is NHWC; these are handled in the driver **[PAPER §3.4.3]**
      → **Fig. 6**.
   6. `CreateDataflowPartition` — **must yield exactly one partition** (FINN cannot map multiple
      partitions to the PL). The only node left outside is a final `Mul` (dequantisation), which runs
      on the PS **[PAPER §3.4.4]** → **Fig. 7**.
8. **Hardware pipeline.** N/A yet (graph level).
9. **Configuration parameters.** Normalisation divisor **255** is implied by "normalized to values
   between 0 and 1" **[PAPER §3.4.1]** but the literal value is **[UNSPEC]**; the dequant `Mul`
   constant is **[REPO] `scale.npy` = 0.00787402 = 1/127**.
10. **Expected output.** Four graph snapshots (Figs. 4, 5, 6, 7 equivalents); a
    `StreamingDataflowPartition` parent/child pair.
11. **Evaluation metrics.** **Hard assertions**: exactly 1 dataflow partition; 0 residual float ops
    inside the partition; node counts of each HW type; exactly 2 residual `Transpose` + 1 `Mul`
    outside. These are pass/fail, not approximate.
12. **[PAPER]** every transform name, the ordering, the `Div` insertion and its purpose, the
    Conv-lowering, the MVU/SWG mapping, the NCHW/NHWC explanation, the single-partition constraint,
    the residual `Mul`-on-PS decision.
13. **[REPO]** the dequant scale value; UINT8 input / INT8 output datatypes.
14. **[UNSPEC]** exact tidy-up transform list ("the FINN source code can be consulted" — §3.4.1
    declines to enumerate them); the literal `Div` divisor; whether input normalisation is folded
    into the first `MultiThreshold`.

- **Required files:** `finn_build/step_01_frontend.py`, `finn_build/custom_steps.py`
- **Required dependencies:** FINN **0.10** **[PAPER §3]**, qonnx, onnx; runs inside the FINN Docker image
- **Required hardware:** CPU
- **Expected measurements:** Figs. 4–7 equivalents + the assertion report

---

## T1.6 — Layer specialisation and folding search  ← **the paper's actual contribution**

1. **Research objective.** Reproduce the optimisation that produces the paper's speed-up: choose
   RTL-vs-HLS per layer and PE/SIMD per layer so the per-layer cycle profile is flat at the
   bottleneck, within the device's DSP/BRAM/LUT budget.
2. **Input.** T1.5 dataflow partition.
3. **Dataset.** N/A.
4. **Model.** T1.3 network as HW nodes.
5. **Model configuration.** N/A.
6. **Quantization method.** Determines the DSP packing factor — see field 9.
7. **Software pipeline.** `SpecializeLayers(fpga_part)` → `ApplyConfig(folding.json)`, plus a
   **search driver of ours** that iterates: estimate cycles per node → unfold the slowest node →
   re-estimate resources → repeat until the slowest node is fully unfolded **[PAPER §3.6.1]**.
   §3.6.2 mentions "a custom script to facilitate agile switching between specialization styles …
   using Python 3.10" **[PAPER]** — we write our own equivalent.
8. **Hardware pipeline.** Target `xc7z020clg400`. **Speed grade: [UNSPEC/contradictory]** — §3 says
   `-2`, the shipped `visdrone.hwh` says `SPEEDGRADE="-1"` **[REPO]** (discrepancy **P5**).
   **Our decision: `-1`**, matching the only hard evidence.
9. **Configuration parameters.**
   - **Specialisation per layer: [PAPER, decodable from Table 6]** — the MVU instance names in
     Table 6 give the exact RTL/HLS assignment, verified against Table 1 by weight count
     (`paper_vs_repository.md` §2.7): **conv 1,2,4,9,10 → RTL; conv 3,5,6,7,8 → HLS.** 5 and 5.
   - **DSP packing [PAPER Table 4]**: DSP48E1 at 4b×4b → **4 MAC/DSP**; at (4,8]b → **2 MAC/DSP**;
     HLS DSP → 1 MAC/DSP. Device has **220 DSP48E1** **[PAPER §3]**.
   - **RTL MVU has no LUT style; HLS MVU supports LUT style** **[PAPER §3.5.2–3.5.3]**. The paper's
     key move: once DSPs are exhausted, switch MVUs to **HLS-LUT** to keep unfolding **[PAPER §3.6.2]**.
   - **RTL SWG parallel-window mode requires full SIMD unfolding** **[PAPER §3.5.4]**.
   - Tail layers (MaxPool, SWG, MultiThreshold on small feature maps) stay **fully folded**
     **[PAPER §3.6.1]**.
   - **PE / SIMD per layer: [UNSPEC — the central gap].** §3.7.2 delegates these to the repository;
     the repository **does not contain them** (`paper_vs_repository.md` §1). The bitstream's
     `StreamingDataflowPartition_1` is a black box — only 7 top-level modules, no per-layer params.
     **→ We re-derive them. See "Folding derivation method" below.**
   - Known fixed points **[REPO]**: input SIMD = 3 (`ishape_folded=(1,416,416,1,3)`); output PE = 1
     (`oshape_folded=(1,13,13,18,1)`); weights internal, `number_of_external_weights=0`
     — consistent with **[PAPER §3.8]** (all weights in BRAM).
10. **Expected output.** `configs/folding.json`; a flat cycles-per-layer plot (**Fig. 10**
    equivalent); a per-layer resource plot (**Fig. 11** equivalent); the default-specialised
    fully-folded baseline plot (**Fig. 8** equivalent).
11. **Evaluation metrics.** max-cycles node (the bottleneck); cycle-profile flatness (e.g. max/median
    ratio); estimated BRAM/DSP/LUT vs. device limits (140 / 220 / 53 200 **[PAPER §3]**);
    predicted FPS = f_clk / max_node_cycles.
12. **[PAPER]** the whole optimisation *method*; Table 4 packing table; the RTL/HLS capability
    matrix; the HLS-LUT escape hatch; parallel-window constraint; keep-tail-folded rule;
    specialisation assignment via Table 6; device resource limits; the targets in Figs. 8/10/11.
13. **[REPO]** input SIMD, output PE, internal-weights confirmation, fpga part/board/speed grade.
14. **[UNSPEC]** **all PE/SIMD values**; FINN's `target_fps` vs. manual iteration; `ram_style` /
    `resType` per layer; `mem_mode`; the tie-break rule when two nodes are equally slow; the stopping
    criterion ("reasonable resource limits" is not quantified).

### Folding derivation method **[OURS]** — and why it is a derivation, not a guess

The paper's Table 8 is internally consistent with a closed-form FINN cost model, which we verified
arithmetically against the paper's own numbers:

```
MVU cycles  =  (C_out / PE) · (k²·C_in / SIMD) · (H_out · W_out)
latency     =  total_cycles / f_clk
throughput  =  f_clk / max_node_cycles
```

Checks against **[PAPER Table 8]** (all at 100 MHz):
- `4,586,895 cycles / 100 MHz = 45.87 ms` → matches the stated latency **exactly**.
- `100 MHz / 216,320 cycles = 462.3 FPS` → matches the stated 462.28 FPS **exactly**.
- The bottleneck `StreamingMaxPool_hls_0` at 216,320 cycles decomposes as
  `416·416 + 208·208 = 173,056 + 43,264 = 216,320` — i.e. one cycle per input pixel plus one per
  output pixel, which is the expected HLS streaming-maxpool behaviour.
- Conv 1 with full input unfolding (`SIMD = k²C_in = 27`, consistent with RTL SWG parallel-window
  **[PAPER §3.5.4]**, and `PE = C_out = 8`) gives `1 · 1 · 416·416 = 173,056 cycles < 216,320` ✓ —
  and `SIMD = 27` with a 3-channel input stream is exactly what **[REPO]** `ishape_folded`'s
  trailing `3` encodes.

So the **balance target is ~216,320 cycles/frame**, and for each MVU we solve for the smallest
`(PE, SIMD)` satisfying `(C_out/PE)·(k²C_in/SIMD)·H·W ≤ 216,320` under the DSP/BRAM/LUT budget, with
the RTL/HLS assignment fixed by Table 6. This reproduces the paper's *procedure* and its *stated
targets* without copying any value. **No PE/SIMD numbers are asserted in this document** — they are
the output of T1.6, to be reported with the derivation.

- **Required files:** `finn_build/step_02_folding.py`, `tools/folding_search.py`,
  `tools/cycle_model.py`, `configs/folding.json` (generated), `configs/specialize.json`
- **Required dependencies:** FINN 0.10 analysis passes; matplotlib for Figs. 8/10/11
- **Required hardware:** CPU (estimation only — no synthesis in this experiment)
- **Expected measurements:** Figs. 8, 10, 11 equivalents; a folding table with derivation shown

---

## T1.7 — FIFO depth sizing and the StreamingMaxPool anomaly

1. **Research objective.** Size inter-layer FIFOs so no layer stalls, without wasting BRAM; and
   independently confirm the paper's reported HLS MaxPool pathology.
2. **Input.** T1.6 folded, specialised graph.
3. **Dataset.** N/A (random/representative tensors).
4. **Model.** As folded.
5. **Model configuration.** N/A.
6. **Quantization method.** N/A.
7. **Software pipeline.** FINN FIFO insertion + depth characterisation, driven by **RTL simulation
   with `pyverilator`** **[PAPER §3.7.1]**; iterate depths, measure per-layer stall fraction.
8. **Hardware pipeline.** Simulation only.
9. **Configuration parameters.** FIFO depths: **[UNSPEC — delegated by §3.7.2 to a repository that
   does not contain them]**. Sizing strategy (FINN's `InsertAndSetFIFODepths` characterisation vs.
   analytic vs. manual): **[UNSPEC]** — **our decision: use FINN's automated characterisation first,
   then hand-tune only nodes that still stall.** Stall-fraction acceptance threshold: **[UNSPEC]**.
10. **Expected output.** `configs/fifo_depths.json`; a per-node stall report; a written
    reproduction note on the MaxPool anomaly.
11. **Evaluation metrics.** Per-node stall %, total FIFO BRAM cost, resulting max-cycles node.
    **Specific claim to test [PAPER §3.7.2]:** `StreamingMaxPool_hls_0` consumes **≈570,000 cycles**
    against a predicted **216,320** (≈2.6×), because "Vitis implementations on large feature maps
    fail to optimize loop unrollings". §4.2.5 attributes a **3.43×** effective throughput drop to
    this bottleneck. Table 8's RTLSim column reports the max node at **741,670** cycles.
    ⚠ Note 570 000 and 741 670 are *different* numbers in the same paper for the same layer; we will
    report whichever our simulation yields and flag the ambiguity rather than reconcile it.
12. **[PAPER]** the necessity and purpose of FIFOs; the pyverilator/RTL-sim methodology; the
    short-vs-long FIFO trade-off; the MaxPool anomaly with its cause, magnitude and consequence; the
    note that an RTL MaxPool would fix it but was out of scope.
13. **[REPO]** nothing — no FIFO configuration exists upstream.
14. **[UNSPEC]** all depth values; sizing algorithm; acceptance threshold; which of 570 k / 741 670
    is the intended figure.

- **Required files:** `finn_build/step_03_fifos.py`, `configs/fifo_depths.json` (generated)
- **Required dependencies:** FINN 0.10, pyverilator, Verilator **[PAPER ref 72]**
- **Required hardware:** CPU. ⚠ **Memory-heavy — see "Host risks".**
- **Expected measurements:** FIFO table, stall report, MaxPool cycle count

---

## T1.8 — Analytical MAC / cycle / throughput model

1. **Research objective.** Independently produce the paper's static cost accounting, and use it as
   the oracle that T1.10 (RTLSim) and T2.1 (hardware) are compared against.
2. **Input.** T1.6 configured graph.
3. **Dataset.** N/A.
4. **Model.** As configured.
5. **Model configuration.** Table 1.
6. **Quantization method.** Determines per-layer data-type pairs (8b–8b, 4b–4b, 4b–8b).
7. **Software pipeline.** `tools/cycle_model.py` + FINN's analysis passes → per-node MAC count,
   weight count, data type, cycle estimate; aggregate.
8. **Hardware pipeline.** N/A.
9. **Configuration parameters.** From T1.6.
10. **Expected output.** A **Table 6** equivalent (per-MVU MACs, weights, data type) and the
    *Estimated* column of **Table 8**.
11. **Evaluation metrics.** Paper reference values **[PAPER]** — reproduced here as cross-checks our
    model must satisfy, since they are architecture-determined and therefore *not* dependent on
    folding:
    - Table 6 totals: **113,330,048 MAC @4b–4b**, **37,380,096 @8b–8b**, **2,847,312 @4b–8b**;
      **333,632 weights @4b**, **17,064 @8b** (350,696 total).
    - Per-layer weight counts all satisfy `k²·C_in·C_out` against Table 1 — verified in
      `paper_vs_repository.md` §2.7. **This is a strong, folding-independent correctness test of our
      T1.3 model: if our weight counts do not match Table 6 exactly, our architecture is wrong.**
    - Table 8 estimated: total 4,586,895 cycles; max node 216,320; 45.87 ms; 462.28 FPS.
12. **[PAPER]** Table 6 in full; Table 8's estimated column; the cycle/latency/throughput
    relationships (verified consistent, see T1.6).
13. **[REPO]** nothing.
14. **[UNSPEC]** whether "total cycles" is a sum over nodes or a critical-path figure — Table 8
    prints "4,586,895 → critical path", which is ambiguous. We will report both.

- **Required files:** `tools/cycle_model.py`, `tools/report_tables.py`
- **Required dependencies:** FINN 0.10 analysis, pandas
- **Required hardware:** CPU
- **Expected measurements:** Table 6 + Table 8 (estimated) equivalents

---

## T1.9 — HLS synthesis, IP stitching, ZynqBuild

1. **Research objective.** Produce a placed-and-routed bitstream for the Zynq-7020 and report real
   resource utilisation and timing closure.
2. **Input.** T1.6/T1.7 configured graph.
3. **Dataset.** N/A.
4. **Model.** As configured.
5. **Model configuration.** N/A.
6. **Quantization method.** N/A.
7. **Software pipeline.** **[PAPER §3.9–3.10]**: `PrepareIP` (emit RTL) → `HLSSynthIP` (Vitis HLS
   C→RTL) → `CreateStitchedIP` (adds **`idma0`** and **`odma0`** for DRAM in/out **[PAPER §3.10]**)
   → `ZynqBuild` (Vivado synth + P&R) → PYNQ driver generation **[PAPER §3.10.2]**.
8. **Hardware pipeline.** Part `xc7z020clg400-1` (our decision, **P5**); board
   **`myir.com:mys-7z020:part0:2.1`** **[REPO]** — a MYIR vendor board file we must obtain;
   PL clock target **200 MHz**, `clock_mode: aggressive` **[REPO `model_info.json`]**;
   Vivado/Vitis **2022.1** **[REPO `visdrone.hwh`]**.
9. **Configuration parameters.** Target clock, part, board, `clock_mode`. Synthesis/implementation
   strategies: **[UNSPEC]**. Number of parallel HLS workers: **[OURS]** — must be 1–2 on this host.
10. **Expected output.** `model.bit`, `model.hwh`, generated PYNQ driver; the stitched block design
    (**Fig. 12** equivalent); the device floorplan (**Fig. 13** equivalent); Vivado utilisation
    report (**Fig. 16**) and timing report (**Fig. 17**).
11. **Evaluation metrics.** **Table 7** equivalent — estimated vs. actual:
    | Resource | Estimated **[PAPER]** | Actual **[PAPER]** | Available | Util % |
    |---|---|---|---|---|
    | BRAM_18K | 200 | 138 | 140 | 98.57 |
    | LUT | 26,694 | 41,605 | 53,200 | 78.20 |
    | DSP | 202 | 204 | 220 | 92.73 |
    Plus WNS/TNS (timing closure must be met). The paper's explanation for the gap **[PAPER §4.2.4]**:
    estimated BRAM (200) exceeded the 140 available, so Vivado converted some to **LUTRAM**, and
    Vitis HLS blocks inflate LUT use.
12. **[PAPER]** the full transform sequence; idma/odma roles; Table 7 with its explanation; the
    figures to produce; device limits.
13. **[REPO]** board file identity, Vivado version, speed grade, 200 MHz / aggressive clock mode.
14. **[UNSPEC]** synthesis and implementation strategy/directives; whether retiming or physopt was
    enabled; the MYIR board-file source.

- **Required files:** `finn_build/step_04_synth.py`, `finn_build/platform/mys-7z020/*`,
  `tools/parse_vivado_reports.py`
- **Required dependencies:** **Vivado + Vitis HLS 2022.1**, FINN 0.10, MYIR board files
- **Required hardware:** CPU-heavy, memory-heavy. ⚠ **See "Host risks".**
- **Expected measurements:** Table 7, Figs. 12, 13, 16, 17 equivalents

---

## T1.10 — RTL simulation throughput

1. **Research objective.** Obtain a cycle-accurate throughput figure that sits between the
   analytical estimate and the hardware measurement, isolating fabric behaviour from driver overhead.
2. **Input.** Stitched IP from T1.9.
3. **Dataset.** Random tensors of the correct dtype/shape (and, for T1.12, real images).
4. **Model.** As built.
5. **Model configuration.** N/A.
6. **Quantization method.** N/A.
7. **Software pipeline.** FINN `rtlsim` on the stitched IP via **Verilator** **[PAPER ref 72]**;
   measure cycles per frame in a steady-state stream.
8. **Hardware pipeline.** Simulated.
9. **Configuration parameters.** Batch/stream length: **[UNSPEC]**. `rtlsim_trace` off (VCD tracing
   would be prohibitive at 4.5 M cycles on this host) — **[OURS]**.
10. **Expected output.** Cycles/frame, derived FPS at 100 MHz.
11. **Evaluation metrics.** **Table 8 RTLSim column [PAPER]**: max node 741,670 cycles,
    latency 7.41 ms, **134.83 FPS**. Note the paper's own framing: the analytical 462 FPS
    "overlooks … memory transactions, pipeline boundaries, and partial stalling", and RTLSim at
    134.83 FPS is what a bare-metal C driver later achieved on real hardware **[PAPER §4.2.5]**.
    ⚠ Latency 7.41 ms at 100 MHz = 741 k cycles, i.e. the RTLSim *latency* equals the *max-node*
    cycle count — dimensionally odd for a pipelined design whose estimated latency was 4.59 M
    cycles. We will report our own two figures separately (per-frame throughput cycles, and
    first-frame latency) rather than attempt to match this row.
12. **[PAPER]** the methodology, the tool, the three-way estimate/sim/actual comparison and its
    explanation.
13. **[REPO]** nothing.
14. **[UNSPEC]** stream length, warm-up handling, how the paper's 7.41 ms latency was defined.

- **Required files:** `finn_build/step_05_rtlsim.py`
- **Required dependencies:** FINN 0.10, Verilator, pyverilator
- **Required hardware:** CPU. ⚠ **The most memory- and time-hungry Tier-1 step.**
- **Expected measurements:** Table 8 RTLSim column equivalent

---

## T1.11 — Vivado power estimation

1. **Research objective.** Produce the tool-side power model that the measured board power (T2.3) is
   compared against.
2. **Input.** T1.9 implemented design.
3. **Dataset.** N/A.
4. **Model.** As built.
5. **Model configuration.** N/A.
6. **Quantization method.** N/A.
7. **Software pipeline.** Vivado `report_power` on the routed design; parse to tables.
8. **Hardware pipeline.** Implemented design + the PS7 block.
9. **Configuration parameters.** Switching activity source (vectorless vs. SAIF from simulation):
   **[UNSPEC]** — the paper says only "the Vivado power analysis tool" and that estimates are
   "generated under worst-case or synthetic conditions" **[PAPER §4.3.1]**, implying vectorless.
   **Our decision: vectorless default, and record the confidence level Vivado reports.**
   Ambient/thermal settings: **[UNSPEC]** — the paper mentions a Vivado "maximum ambient estimated
   temperature" of **46.3 °C** **[PAPER §4.3.2]**.
10. **Expected output.** **Table 11** equivalent (per-module power) and **Table 12** equivalent
    (per-resource-type breakdown).
11. **Evaluation metrics.** Paper reference **[PAPER]**: `top_wrapper` 3.129 W, of which
    `StreamingDataflowPartition_1` 1.554 W and `zynq_ps` 1.497 W; total on-chip **3.358 W** =
    dynamic + **0.229 W static**; Block RAM 0.262 W, DSP 0.207 W, Signals 0.612 W,
    Slice Logic 0.428 W, Clocks 0.124 W, PS7 1.496 W. The paper notes measured power is
    "considerably lower" than this estimate.
12. **[PAPER]** the tool, both tables in full, the worst-case-estimate caveat, 46.3 °C ambient.
13. **[REPO]** nothing.
14. **[UNSPEC]** switching-activity methodology, ambient/airflow/heatsink settings in the power
    constraints.

- **Required files:** `tools/parse_power_report.py`
- **Required dependencies:** Vivado 2022.1
- **Required hardware:** CPU
- **Expected measurements:** Tables 11, 12 equivalents

---

## T1.12 — Functional equivalence PyTorch ↔ FINN  **[OURS — closes a gap in the paper]**

1. **Research objective.** Prove the hardware model computes the same function as the trained
   network. **The paper never does this**: §4.2.2 states only that RTL simulation "confirmed the
   correct working of the implemented system", with no quantitative evidence, and the repository's
   `validate.py` is unmodified FINN MNIST/CIFAR-10 boilerplate
   (`paper_vs_repository.md` §5). Without this, every accuracy number is unanchored from the hardware.
2. **Input.** T1.3 `best.pt`; T1.5 FINN graph; T1.9 stitched IP; a fixed set of val images.
3. **Dataset.** A frozen subset of `visdrone_car/val` (size **[UNSPEC]** — our choice, e.g. a few
   hundred images; recorded).
4. **Model.** T1.3 network at four abstraction levels: Brevitas QAT, QONNX, FINN `cppsim`,
   FINN `rtlsim`.
5. **Model configuration.** As T1.3.
6. **Quantization method.** As T1.3; this experiment verifies the quantisation survived compilation.
7. **Software pipeline.** For each image: Brevitas forward → QONNX execute → `cppsim` → `rtlsim`;
   assert tensor equality at each hop; then apply the **[REPO]** dequant (`×1/127`), the **[REPO]**
   Detect decode and NMS, and compute detection metrics at each level.
8. **Hardware pipeline.** cppsim / rtlsim only (Tier 1); the same harness is reused on-board in T2.1.
9. **Configuration parameters.** Tolerance: **exact integer equality** for QONNX/cppsim/rtlsim;
   Brevitas→QONNX may differ only by export rounding — threshold **[UNSPEC/OURS]**.
   NMS `conf_thres=0.10`, `iou_thres=0.10`, `max_det=20` **[REPO `infer_client_v3.py:342`]**.
10. **Expected output.** An equivalence report: per-hop max abs difference, % exactly-equal output
    tensors, and mAP@0.5 computed at each level.
11. **Evaluation metrics.** Bit-exactness rate; mAP@0.5 delta between Brevitas and rtlsim (should be
    ~0); the same P/R/mAP as T1.3 reproduced through the hardware path.
12. **[PAPER]** only the unquantified claim in §4.2.2, plus Table 5's metric definitions.
13. **[REPO]** dequant scale, decode formulas, NMS thresholds, I/O dtypes and layouts.
14. **[UNSPEC]** everything about the methodology — this experiment is ours.

- **Required files:** `tools/equivalence_check.py`, `utils/postprocess.py`, `utils/metrics.py`
- **Required dependencies:** FINN 0.10, torch 1.13.1, brevitas 0.9.0, Verilator
- **Required hardware:** CPU + GPU (for the Brevitas reference pass)
- **Expected measurements:** equivalence report (no paper counterpart)

---

## T1.13 — Coral path: FP32 training → TFLite INT8 PTQ → Edge TPU compile

1. **Research objective.** Build the comparison artifact for T3.1 — the same task on a commercial
   ASIC accelerator.
2. **Input.** T1.1 dataset.
3. **Dataset.** `visdrone_car` (+ a representative calibration subset for PTQ).
4. **Model.** **[UNSPEC — important.]** §3.3 says only "another training run was performed, this
   time with full FP32 precision", without stating whether it used the *pruned* Table-1 topology or
   stock YOLOv3-Tiny. This materially affects the fairness of Table 15. **Our decision: use the
   pruned Table-1 topology** so the Coral and FPGA run the same network; record the deviation risk.
5. **Model configuration.** As chosen above; 416×416; `nc=1`.
6. **Quantization method.** **INT8 post-training quantisation (PTQ)** **[PAPER §3.3]** — explicitly
   PTQ, not QAT, and therefore *not* the same quantisation as the FPGA path. Calibration dataset
   size/selection: **[UNSPEC]**.
7. **Software pipeline.** Train FP32 → export TFLite → PTQ INT8 → `edgetpu_compiler`. **[PAPER §3.3]**
   notes a newer **community-maintained `libedgetpu`** is required **[PAPER ref 68:
   `github.com/feranick/libedgetpu`]**.
8. **Hardware pipeline.** Edge TPU compiler (host); execution in T3.1.
9. **Configuration parameters.** PTQ scheme (full-integer vs. dynamic range), calibration count,
   compiler version: all **[UNSPEC]**.
10. **Expected output.** `model_int8.tflite`, `model_int8_edgetpu.tflite`, compiler log (ops mapped
    to TPU vs. fallen back to CPU — this strongly affects T3.1 and the paper never reports it).
11. **Evaluation metrics.** Compile success; **fraction of ops mapped to the Edge TPU** (ours);
    FP32 and INT8 mAP@0.5 on `val`.
12. **[PAPER]** that an FP32 run was done, that INT8 PTQ was used, that the compiler was applied,
    and that the community `libedgetpu` is needed.
13. **[REPO]** nothing — no TFLite or Coral code exists upstream.
14. **[UNSPEC]** topology used, PyTorch→TFLite conversion route, PTQ scheme, calibration set,
    compiler version, op-mapping outcome.

- **Required files:** `tools/export_tflite.py`, `tools/compile_edgetpu.sh`
- **Required dependencies:** tensorflow / tflite converter, `edgetpu_compiler`, `libedgetpu` (fork)
- **Required hardware:** GPU for the FP32 run; compilation is CPU-only (**no Coral needed yet**)
- **Expected measurements:** TFLite/Edge TPU artifacts + mAP; op-mapping report

---

# TIER 2 — Requires the FPGA board

**Blocking hardware:** MYiR **Z-turn Board V2** with `XC7Z020CLG400` SoC, 1 GiB DDR3, micro-SD,
HDMI, Gigabit Ethernet **[PAPER §3]**. Board file `myir.com:mys-7z020:part0:2.1` **[REPO]**.
None of this is present on or attached to the host. Every experiment below is **designed now,
executed later.**

---

## T2.1 — On-board throughput sweep (frequency × batch size)

1. **Research objective.** Measure real accelerator throughput versus PL clock and batch size, and
   quantify the gap to the analytical (T1.8) and simulated (T1.10) predictions.
2. **Input.** T1.9 bitstream + generated PYNQ driver; random tensors resident in DRAM.
3. **Dataset.** None for the throughput figure — **[PAPER §4.2.6]** states results were obtained "by
   directly feeding random data directly via DRAM transfers, avoiding unnecessary data reshaping and
   repacking". Real-data runs are T2.2.
4. **Model.** As built.
5. **Model configuration.** N/A.
6. **Quantization method.** N/A.
7. **Software pipeline.** PYNQ overlay load → `throughput_test()`.
   **Critical methodological detail [REPO `driver_base.py:421-448`]:** `throughput_test()` times
   **only `execute_on_buffers()`**, with buffers already in DRAM:
   ```python
   start = time(); self.execute_on_buffers(); end = time()
   res["throughput[images/s]"] = self.batch_size / (end - start)
   ```
   `fold_input`, `pack_input`, `copy_input_data_to_device`, `copy_output_data_from_device`,
   `unpack_output`, `unfold_output` are timed **separately and excluded**. So **Table 9's FPS is a
   fabric+DMA number, not end-to-end.** Our harness must use this identical definition or the
   comparison is invalid. This is the single most important detail recovered from the repository.
8. **Hardware pipeline.** PS (Linux + PYNQ) → DMA → `idma0` → StreamingDataflowPartition → `odma0`.
9. **Configuration parameters.** PL clock ∈ **{50, 100, 200} MHz** **[PAPER §4.2.6]**;
   batch ∈ **{1, 100}** **[PAPER Table 9]**; clock set at runtime via the PYNQ driver
   **[PAPER §3.10.2]**; `clock_mode: aggressive`, `max_clock_freq: 200.0` **[REPO `model_info.json`]**.
   Repetition count and warm-up: **[UNSPEC]** — the paper reports single values with no dispersion
   for Table 9. **Our decision: ≥10 repetitions, report mean ± σ** (an improvement on the paper).
10. **Expected output.** A **Table 9** equivalent plus the *Actual* column of **Table 8**.
11. **Evaluation metrics.** Runtime (ms), throughput (FPS), DRAM in/out bandwidth (Mb/s).
    Paper reference **[PAPER Table 9]**:
    | Clock | Batch | Runtime (ms) | FPS | DRAM in (Mb/s) | DRAM out (Mb/s) |
    |---|---|---|---|---|---|
    | 50 | 1 | 22.774 | 43.909 | 22.796 | 0.134 |
    | 50 | 100 | 1915.531 | 52.205 | 27.103 | 0.159 |
    | 100 | 1 | 11.564 | 86.473 | 44.894 | 0.263 |
    | 100 | 100 | 957.986 | 104.386 | 54.194 | 0.318 |
    | 200 | 1 | 6.068 | 164.786 | 85.552 | 0.501 |
    | 200 | 100 | **479.189** | **208.686** | 108.343 | 0.635 |
    Expected qualitative behaviours to test **[PAPER]**: roughly **linear scaling with clock**;
    batch-100 beats batch-1 by **~17 %** because a full pipeline avoids flush stalls.
    ⚠ **Corroboration warning:** the repository's only committed measurement
    (`nw_metrics.txt`: **23.49 FPS @ 100 MHz, batch 10**) is ~4.4× below Table 9, and was produced
    with a **different bitstream** (`resizer.bit`, distinct MD5 from `visdrone.bit`). See
    `paper_vs_repository.md` §4. We therefore have **no independent corroboration of Table 9** and
    should treat it as a single-source claim.
12. **[PAPER]** the three clock points, both batch sizes, the random-data-via-DRAM methodology, all
    of Table 9, the linear-scaling and 17 % batch claims.
13. **[REPO]** the exact `throughput_test()` semantics; `max_clock_freq`/`clock_mode`.
14. **[UNSPEC]** repetition count, warm-up, thermal state during measurement, PS clock/governor
    settings, whether DVFS was active.

- **Required files:** `bench/throughput_sweep.py`, `deploy/` package (bitstream + our driver)
- **Required dependencies:** PYNQ on the board; numpy
- **Required hardware:** **Z-turn Board V2**, micro-SD with PYNQ image, Ethernet
- **Expected measurements:** Table 9 + Table 8 (actual) equivalents, with dispersion

---

## T2.2 — Per-stage latency breakdown (real data, Python driver)

1. **Research objective.** Quantify where end-to-end time goes when the standard FINN Python driver
   is used on real video frames, and so quantify the driver's cost.
2. **Input.** A video stream / image sequence.
3. **Dataset.** Real frames (val images or drone footage).
4. **Model.** As built.
5. **Model configuration.** N/A.
6. **Quantization method.** N/A.
7. **Software pipeline.** Instrumented host/PS pipeline with timers around each stage:
   preprocessing → driver exec → rescale → detect → NMS → box processing.
8. **Hardware pipeline.** As T2.1, batch size **1** **[PAPER §4.2.6]**.
9. **Configuration parameters.** Batch 1; clocks {50, 100, 200} MHz **[PAPER Table 10]**;
   letterbox 416×416 + BGR→RGB **[REPO]**; NMS `conf=0.10`, `iou=0.10`, `max_det=20` **[REPO]**.
   Number of frames averaged: **[UNSPEC]** (the paper reports mean ± σ, so n > 1, but n is not given).
10. **Expected output.** A **Table 10** equivalent.
11. **Evaluation metrics.** Mean ± σ per stage. Paper reference **[PAPER Table 10]** (ms):
    | Stage | 50 MHz | 100 MHz | 200 MHz |
    |---|---|---|---|
    | Preprocessing | 32.907 ± 1.040 | 34.043 ± 1.212 | 32.191 ± 1.490 |
    | Driver Exec | 28.913 ± 0.700 | 17.891 ± 0.788 | 12.121 ± 0.326 |
    | Rescale | 0.511 ± 0.213 | 0.500 ± 0.176 | 0.481 ± 0.024 |
    | Detect | 4.988 ± 1.137 | 4.670 ± 0.237 | 4.746 ± 4.502 |
    | NMS | 10.399 ± 2.644 | 9.335 ± 1.051 | 9.232 ± 2.383 |
    | Box Process | 11.373 ± 2.778 | 11.211 ± 3.170 | 10.749 ± 2.950 |
    | Total Postprocess † | 27.980 ± 5.046 | 26.362 ± 3.643 | 25.819 ± 8.754 |

    † sum of detect + NMS + box process **[PAPER]**.
    Derived claim to test **[PAPER §4.2.6]**: the Python driver costs **21 % / 35 % / 50 %** of
    throughput at 50 / 100 / 200 MHz, because repack time is fixed while compute time shrinks.
    Note that only *Driver Exec* scales with clock — all other stages are PS-bound, which is the
    paper's own explanation for the 30 FPS end-to-end ceiling in T2.4.
12. **[PAPER]** the stage decomposition, all of Table 10, the 21/35/50 % driver-penalty analysis.
13. **[REPO]** the preprocessing and post-processing definitions that make the stages reproducible.
14. **[UNSPEC]** frame count, which images, whether the PS was otherwise idle, PS clock governor.

- **Required files:** `bench/latency_breakdown.py`
- **Required dependencies:** PYNQ, numpy, opencv (on PS), torch *or* a numpy decode path
- **Required hardware:** Z-turn board
- **Expected measurements:** Table 10 equivalent

---

## T2.3 — Measured power and thermal

1. **Research objective.** Measure real board power under continuous inference load and derive FPS/W;
   compare with the Vivado estimate from T1.11.
2. **Input.** Continuous inference load from T2.1.
3. **Dataset.** N/A (synthetic load).
4. **Model.** As built.
5. **Model configuration.** N/A.
6. **Quantization method.** N/A.
7. **Software pipeline.** Run `throughput_test()` in a loop; log power and die temperature.
8. **Hardware pipeline.** Whole board powered from a **5 V DC bench supply with current readout**
   **[PAPER §4.3, Fig. 18a]**; **IR thermal camera** aimed at the SoC **[PAPER Fig. 18b]**; **no
   heatsink — static air only** **[PAPER §4.3.2]**.
9. **Configuration parameters.** 5 V rail; clocks {idle, 50, 100, 200} MHz; measurement at
   steady-state load. Sampling rate, averaging window, ambient temperature: **[UNSPEC]**.
10. **Expected output.** **Table 13** (power) and **Table 14** (FPS/W) equivalents + thermal images.
11. **Evaluation metrics.** Paper reference **[PAPER]**:
    - Table 13: idle (no bitstream) **1.9 W**; 50 MHz **2.12 W**; 100 MHz **2.38 W**; 200 MHz **2.55 W**.
    - **Derived [PAPER §4.3.2]:** since ~1.9 W is PS + DDR, the **PL draws only 0.22–0.65 W**.
    - Table 14 (FPS/W, whole-package power): 20.71 / 24.63 (50 MHz, batch 1/100);
      36.32 / 43.86 (100 MHz); **64.63 / 81.88** (200 MHz).
    - Thermal: die **53.1 °C** measured vs. Vivado's 46.3 °C max ambient estimate.
    - The paper's own caveat: idle power is high because the board firmware keeps the ARM PS at
      maximum frequency with no load-based scaling; it states that fixing this is out of scope.
12. **[PAPER]** the measurement setup (5 V bench, IR camera, no heatsink), all of Tables 13 and 14,
    the PS/PL attribution, the 53.1 °C figure, the unoptimised-firmware caveat.
13. **[REPO]** nothing.
14. **[UNSPEC]** instrument models and accuracy, sampling/averaging, ambient conditions, emissivity
    setting for the IR camera, whether DDR was included in "PS".
    **Our additions:** report instrument uncertainty, and measure PL-only power by differencing
    bitstream-loaded vs. not — the paper derives this rather than measuring it.

- **Required files:** `bench/power_log.py`, `bench/README_bench_setup.md`
- **Required dependencies:** instrument driver (depends on supply model)
- **Required hardware:** Z-turn board, **5 V bench supply with current measurement**, **IR thermal
  camera**. ⚠ Both absent; without them T2.3 cannot run and Tables 13/14 and all FPS/W figures,
  including the paper's headline efficiency claim, are unreproducible.
- **Expected measurements:** Tables 13, 14, Fig. 18 equivalents

---

## T2.4 — End-to-end WebSocket streaming ("FPGA as a TPU")

1. **Research objective.** Evaluate the accelerator as a network-attached inference device, and
   measure the end-to-end throughput/latency that an external client actually observes.
2. **Input.** Live camera or video file on the client.
3. **Dataset.** Live frames.
4. **Model.** Deployed via a **`.fnn`** package.
5. **Model configuration.** As built.
6. **Quantization method.** N/A.
7. **Software pipeline.** **[PAPER §3.11]**, reimplemented by us:
   - **Server (`InferServer`, on PS):** WebSocket; JSON commands **`load_model`** and **`status`**;
     binary frames; batches of **up to 100** frames; MD5 integrity check of the `.fnn`
     **[PAPER §3.11.4/§3.11.9]**, all corroborated **[REPO `infer_server.py`]**.
   - **Client (`InferClient`):** 4 threads — **Capture**, **Frame Dispatcher/Sender**,
     **Post-processing**, **Display** **[PAPER §3.11.2]**; thread-safe bounded queues.
   - **Binary header, 16 bytes [PAPER §3.11.5]:** `frame_id` u32, `width` u32, `height` u32,
     `channels` u16, `dtype_code` u16, then raw payload. Confirmed **[REPO]**.
   - **`.fnn` = zip of `model.bit`, `model.hwh`, `model.pkl`, `model_info.json`** **[PAPER §3.11.12]**,
     confirmed byte-for-byte **[REPO `yolov3-tiny-4w4a-visdrone.fnn`]**; `model_info.json` must carry
     `max_clock_freq`, `model_name`, `trained_on_dataset` **[REPO]**.
   - **Load sequence [PAPER §3.11.13]:** extract (base64) → verify all 4 files → parse metadata →
     configure FPGA via PYNQ → initialise queue + worker thread.
8. **Hardware pipeline.** Client ⇄ WiFi/Ethernet ⇄ PS ⇄ DMA ⇄ PL.
9. **Configuration parameters.** Port **8765** **[REPO]**; client queue size **30** **[REPO
   `infer_client_v3.py:423`]**; server batch cap **100** **[PAPER + REPO]**; PL clock 200 MHz
   **[PAPER §4.4]**; server max WebSocket queue "increased to a high value" **[PAPER §4.4]** —
   actual value **[UNSPEC]**.
10. **Expected output.** End-to-end FPS, stream latency over WiFi and wired Ethernet, annotated video.
11. **Evaluation metrics.** Paper reference **[PAPER §4.4]**: ~**12 ms** per inference on the server
    at 200 MHz (consistent with Table 10's driver cost); max stream latency **~130 ms over WiFi**,
    **~13 ms over wired Ethernet**; **30 FPS** achieved end-to-end with a wired link and a macOS
    client. The paper attributes the shortfall from 208 to 30 FPS to "many unnecessary data-move
    operations, copying and conversion between raw buffer arrays and the numpy array", and notes
    WebSocket performance "varies greatly between different operating system implementations".
    Context given **[PAPER]**: 30 FPS still exceeds LPYOLO's 18 FPS [63] on the same board and network.
12. **[PAPER]** the whole architecture, thread roles, protocol split (JSON vs. binary), the 16-byte
    header layout, the `.fnn` structure and load sequence, batch cap of 100, all §4.4 measurements
    and their explanation.
13. **[REPO]** port, queue sizes, concrete server/client structure, and a **working reference `.fnn`**
    to validate our packager against. ⚠ But the reference client is **incomplete — its `utils`
    module is missing from the repository**, so it cannot be run as shipped
    (`paper_vs_repository.md` §5).
14. **[UNSPEC]** server queue cap, client OS for our runs, network topology, whether the 30 FPS was
    camera- or pipeline-limited (the camera may itself cap at 30 FPS — **we must test with a >30 FPS
    source to separate these**, which the paper does not do).

- **Required files:** `streamer/server.py`, `streamer/client.py`, `streamer/protocol.py`,
  `streamer/fnn_package.py`, `utils/postprocess.py`
- **Required dependencies:** websockets, asyncio, numpy, opencv-python, PYNQ (board side)
- **Required hardware:** Z-turn board, host with camera, Ethernet and WiFi paths
- **Expected measurements:** §4.4 equivalents, with the camera-limit confound resolved

---

## T2.5 — Field footage inference

1. **Research objective.** Demonstrate on-field behaviour and test the paper's claim that real
   fixed-altitude drone imagery yields far better recall than the Table 5 validation figure suggests.
2. **Input.** Drone video over roads and parking lots.
3. **Dataset.** **[PAPER §4.5]** — the authors' own captures. **Not published; unobtainable.**
   **Our decision required:** either capture our own footage, or substitute a public aerial
   car-detection sequence (e.g. a held-out VisDrone *video* sequence). Substitution changes the
   experiment and must be labelled as a deviation.
4. **Model.** As built.
5. **Model configuration.** As built.
6. **Quantization method.** N/A.
7. **Software pipeline.** Offline video → preprocess → infer → decode → NMS → annotate.
8. **Hardware pipeline.** Board (or rtlsim as a Tier-1 fallback, which would still demonstrate the
   function without the performance claim).
9. **Configuration parameters.** NMS `conf=0.10`, `iou=0.10`, `max_det=20` **[REPO]** — note these
   are permissive and inflate apparent recall relative to a standard 0.25/0.45; flight altitude,
   camera, resolution, frame count all **[UNSPEC]**.
10. **Expected output.** Annotated frames — a **Fig. 19** equivalent.
11. **Evaluation metrics.** **[PAPER §4.5]** offers only a qualitative claim: "almost all of the cars
    were correctly detected and localized". There is **no quantitative field metric in the paper**.
    **Our addition:** hand-label a small frame subset and report actual P/R, so the claim is testable.
12. **[PAPER]** the intent, the qualitative outcome, and the fixed-altitude/fixed-scale argument
    (§4.1: "if the intended usage is onboard a drone that can be flown at a standard altitude the
    scale of the objects to be recognized can be assumed to be essentially fixed").
13. **[REPO]** NMS thresholds and the decode path.
14. **[UNSPEC]** the footage itself, capture parameters, and any quantitative criterion.

- **Required files:** `tools/run_video.py`
- **Required dependencies:** opencv-python
- **Required hardware:** Z-turn board + footage (or rtlsim fallback)
- **Expected measurements:** Fig. 19 equivalent + our own labelled P/R

---

# TIER 3 — Requires additional hardware

---

## T3.1 — Coral Edge TPU benchmark

1. **Research objective.** Benchmark the same detection task on a commercial ASIC edge accelerator,
   for the FPGA-vs-ASIC comparison of Table 15.
2. **Input.** T1.13 `model_int8_edgetpu.tflite`; images.
3. **Dataset.** `visdrone_car` val (for accuracy) + synthetic/real frames (for latency).
4. **Model.** The INT8 PTQ TFLite model.
5. **Model configuration.** 416×416; `nc=1`.
6. **Quantization method.** INT8 PTQ **[PAPER §3.3]** — note this differs from the FPGA path's 4W4A
   QAT, so Table 15 compares two different quantisations of the same architecture. The paper does not
   flag this; **we will.**
7. **Software pipeline.** `libedgetpu` **[PAPER ref 68 — community fork]** + tflite runtime;
   two driver configurations: **standard mode** and **max mode** **[PAPER §5.1]**.
8. **Hardware pipeline.** Coral Edge TPU (USB accelerator or dev board) at **250 MHz** and
   **500 MHz** internal frequency **[PAPER Table 15]**.
9. **Configuration parameters.** Driver mode ∈ {standard, max}; batch size **1** — **[PAPER §5.1]**
   explicitly uses batch 1 for both sides "to allow for a fair comparison, at the cost of degraded
   performance for this work". Repetitions: **[UNSPEC]**.
10. **Expected output.** A **Table 15** equivalent.
11. **Evaluation metrics.** Paper reference **[PAPER Table 15]**:
    | Metric | Coral 250 MHz | Coral 500 MHz | Ours 50 | Ours 100 | Ours 200 |
    |---|---|---|---|---|---|
    | Latency (ms) | 9.69 | 6.73 | 22.77 | 11.56 | 6.07 |
    | FPS | 103.2 | 148.7 | 43.91 | 86.47 | 164.8 |
    | Power (W) | 1.2 † | 2.4 † | 2.12 | 2.38 | 2.55 |
    | FPS/W | 86.0 | 62.0 | 20.7 | 36.3 | 64.5 |

    † **The Coral power figures are datasheet estimates, not measurements** — the paper states this
    plainly in its own footnote ("estimated from the manufacturer's datasheet"), and derives the
    1.2 W figure by simply halving 2.4 W for half-frequency. **Our improvement: measure Coral power
    directly** on the same 5 V bench used in T2.3, making the comparison symmetric.
    Note also that on the paper's own numbers **the Coral at 250 MHz is more efficient
    (86.0 vs 64.5 FPS/W)** than this work; the paper's efficiency claim is specifically against
    *FPGA* literature (Table 16), not against the ASIC.
12. **[PAPER]** both driver modes, both frequencies, batch-1 fairness rule, all of Table 15, and the
    explicit admission that Coral power is estimated.
13. **[REPO]** nothing.
14. **[UNSPEC]** Coral variant (USB vs. M.2 vs. dev board — thermally very different), host machine,
    repetition count, whether latency includes host↔USB transfer, real Coral power.

- **Required files:** `bench/coral_bench.py`
- **Required dependencies:** `libedgetpu` (feranick fork), tflite-runtime / pycoral
- **Required hardware:** **Coral Edge TPU** (absent) + ideally the 5 V bench for real power
- **Expected measurements:** Table 15 equivalent, with *measured* rather than estimated Coral power

---

## T3.2 — Literature comparison (desk work, no hardware)

1. **Research objective.** Place our reconstruction against published YOLOv3-Tiny-class FPGA
   implementations, i.e. rebuild Table 16 with our own row.
2. **Input.** Our T1.9/T2.1/T2.3 results + the cited papers.
3. **Dataset.** N/A — **[PAPER §5.2]** notes mAP is deliberately excluded "because the datasets that
   the networks were trained upon are different".
4.–9. N/A (no implementation).
10. **Expected output.** A **Table 16** equivalent.
11. **Evaluation metrics.** Paper reference **[PAPER Table 16]**:
    | Reference | Board | Precision | Clock | LUT | DSP | BRAM | Latency | FPS | Power | FPS/W |
    |---|---|---|---|---|---|---|---|---|---|---|
    | [12] FINN-R | PYNQ-Z1 | 1W3A | 100 | 46.5 K | — | 280 (18 K) | 33.44 | 29.9 | 2.5 | 11.96 |
    | [48] SATAY | VCU110 | 8W16A | 220 | 127 K | 1780 | 2090.5 (36 K) | 14.3 | 69.93 | 15.4 | 4.54 |
    | [48] SATAY | VCU118 | 8W16A | 255 | 431 K | 6687 | 2148 (36 K) | 6.8 | 147.06 | 42.9 | 3.43 |
    | [63] LPYOLO | Z7020 | 4W4A | 100 | 39.4 K | 203 | 91 (18 K) | 52.3 | 19.12 | 2.4 | 7.97 |
    | This work | Z7020 | 4W4A | 200 | 37.9 K | 204 | 138 (18 K) | 4.79 | 208.77 | 2.55 | 81.85 |
    Caveats the paper itself records **[PAPER §5.2]**: FINN-R's network is Tincy-YOLO, not
    YOLOv3-Tiny; SATAY uses 8W16A (a heavier network); **LPYOLO [63] has the exact same network
    definition as this work**, making it the only apples-to-apples row — 10× FPS and 7.3× FPS/W.
    Cost context: VCU118 > €15,000 vs. ~€150 for the Z-turn board.
    ⚠ The "This work" latency of 4.79 ms is the **batch-100** figure (479.189/100 from Table 9) while
    the Coral comparison in Table 15 used batch 1 (6.07 ms). Mixing batch regimes across the two
    comparison tables flatters Table 16; **our reconstruction will report both consistently.**
12. **[PAPER]** the entire table, the precision/architecture caveats, the cost argument, and the
    reason mAP is omitted.
13. **[REPO]** nothing.
14. **[UNSPEC]** how the cited works' power figures were normalised (board-level vs. chip-level
    differs across these papers and is not stated) — a real weakness in the comparison we should note.

- **Required files:** `docs/comparison_table.md`, `tools/report_tables.py`
- **Required dependencies:** none
- **Required hardware:** none
- **Expected measurements:** Table 16 equivalent, with batch regime stated per row

---

# Host risks and constraints

Measured on this machine, not assumed:

| Resource | Observed | Implication |
|---|---|---|
| GPU | 1× TITAN RTX, 24 GB, driver 535.247.01 | ✅ Sufficient; exceeds the paper's P100 16 GB |
| **RAM** | **15 GB total, ~10 GB available** | ⚠ **The binding constraint.** Vivado P&R on a 7z020 fits, but Verilator RTLSim of the full stitched design (T1.7, T1.10) over millions of cycles is memory-hungry. **Mitigation: `NUM_DEFAULT_WORKERS=1`, no VCD tracing, and validate T1.10 feasibility on a trimmed stream before committing.** This is the first thing to pressure-test. |
| CPU | 6 cores | ⚠ Serialised HLS synthesis will be slow; budget hours per build |
| Disk `/` | 468 GB, **74 GB free (84 % used)** | ⚠ Too tight for Vivado. Nothing large goes here |
| Disk `/mnt/data10tb` | 9.1 TB, **2.9 TB free** | ✅ Install Vivado, FINN build dirs, dataset and checkpoints here |
| Docker | 28.1.1 | ✅ FINN runs containerised |
| Xilinx tools | **none installed** | Must install **2022.1** (per `visdrone.hwh`), to `/mnt/data10tb` |
| VisDrone data | **absent** | T1.1 blocked until downloaded |
| brevitas / FINN / qonnx | **absent in every conda env** | Fresh env needed; no existing env has torch 1.13.1 |
| Z-turn board | **absent** | All of Tier 2 blocked |
| 5 V bench supply, IR camera | **absent** | T2.3 blocked |
| Coral Edge TPU | **absent** | T3.1 blocked |

---

# Open decisions requiring your approval

No value below has been invented or silently defaulted. Each needs a call before implementation.

| # | Decision | Options | Our recommendation |
|---|---|---|---|
| **D1** | Speed grade (**P5**) | `-2` (paper §3) vs. `-1` (shipped `visdrone.hwh`) | **`-1`** — matches the only hard evidence; timing results then correspond to the authors' build |
| **D2** | Table 1 row 12 MaxPool stride (**P2**) | 1 (shapes consistent) vs. 2 (as printed, impossible) | **1 with padding** |
| **D3** | Evaluation split (**P6**) | train (per §4.1 text) vs. val (per Table 5 caption) | **val**, labelled explicitly |
| **D4** | Hyper-parameters | Ultralytics `hyp.scratch-low.yaml` vs. author-tuned (unavailable) | **`hyp.scratch-low.yaml`**, versioned in-repo |
| **D5** | Coral topology (T1.13 field 4) | pruned Table-1 net vs. stock YOLOv3-Tiny | **pruned Table-1**, so both platforms run the same network |
| **D6** | VisDrone class merging | `car` only vs. merge van/truck/bus | **`car` only** |
| **D7** | Folding source | re-derive via the verified cycle model (T1.6) vs. reverse-engineer the bitstream | **re-derive** — it is the reconstruction, and the bitstream is opaque anyway |
| **D8** | T2.5 footage | capture our own vs. substitute public aerial video vs. defer | **defer** until Tier 2 is unblocked |
| **D9** | Tier-2/3 hardware | procure Z-turn V2 + bench supply + IR camera + Coral, vs. scope to Tier 1 | **your call** — tell us what is obtainable and we will re-scope |
| **D10** | FINN version | 0.10 (**[PAPER §3]**) with Vivado 2022.1 (**[REPO]**) | **FINN 0.10 + 2022.1**; verify FINN 0.10 supports 2022.1 before installing |

---

# Execution order (once approved)

```
T1.1 ─> T1.2 ─┐
       └> T1.3 ─> T1.4 ─> T1.5 ─> T1.6 ─> T1.7 ─> T1.9 ─> T1.11
                            │        │        │        └─> T2.1 ─> T2.2 ─> T2.3
                            │        └> T1.8  └> T1.10
                            └> T1.12 (after T1.9/T1.10)                 └─> T2.4 ─> T2.5
       └> T1.13 ──────────────────────────────────────────────────> T3.1
                                                                     └─> T3.2
```

**Suggested first move after approval:** a *feasibility spike* rather than T1.1 — install FINN 0.10 +
Vivado 2022.1 on `/mnt/data10tb` and push a trivial network (e.g. FINN's own TFC/CNV example) all the
way to a 7z020 bitstream and rtlsim. That validates the 15 GB RAM constraint, the 2022.1 pairing and
the MYIR board file **before** we spend GPU-days on training. Failing fast there is much cheaper than
discovering it after T1.3.
