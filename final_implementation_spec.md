# Final Implementation Specification

**Single technical specification** for our reconstruction of Calì, Falaschetti & Biagetti,
*"Optimized Implementation of YOLOv3-Tiny for Real-Time Image and Video Recognition on FPGA"*,
Electronics **2025**, 14, 3993.

**Supersedes nothing; consolidates:** `experiment_reconstruction_plan.md` (per-experiment design),
`paper_vs_repository.md` (evidence base), `required_implementation.md` (build list),
`implementation_decisions.md` (D1–D10).

**Status.** Specification frozen pending your approval. Nothing installed, downloaded, trained,
synthesised, simulated, or built.

### Provenance tags used throughout

| Tag | Meaning |
|---|---|
| **[P]** | Paper-specified. Section/table cited. |
| **[R]** | Repository-recovered. File:line cited. |
| **[F]** | Arithmetically forced — only one value is self-consistent. |
| **[A]** | **OUR REIMPLEMENTATION ASSUMPTION** — undeterminable from either source. |
| **[D]** | Settled by decision D1–D10. |
| **[O]** | Our addition; no counterpart in the paper. |

---

## 1. Model architecture

### 1.1 Resolved layer table

Authoritative source: **paper Table 1 [P]**, with row 12 corrected per **D2 [F]**. Channel widths are
taken from Table 1, *not* from §3.1's "reduced by a factor of 5" prose (see §18, **P1**).

| # | Layer | In | Out | K | S | Pad | Activation | MVU instance | Style |
|---|---|---|---|---|---|---|---|---|---|
| 1 | QuantConv | 416×416×3 | 416×416×8 | 3×3 | 1 | 1 | QuantReLU | `MVAU_rtl_0` | **RTL** |
| 2 | MaxPool | 416×416×8 | 208×208×8 | 2×2 | 2 | 0 | — | — | HLS |
| 3 | QuantConv | 208×208×8 | 208×208×8 | 3×3 | 1 | 1 | QuantReLU | `MVAU_rtl_1` | **RTL** |
| 4 | MaxPool | 208×208×8 | 104×104×8 | 2×2 | 2 | 0 | — | — | HLS |
| 5 | QuantConv | 104×104×8 | 104×104×16 | 3×3 | 1 | 1 | QuantReLU | `MVAU_hls_0` | **HLS** |
| 6 | MaxPool | 104×104×16 | 52×52×16 | 2×2 | 2 | 0 | — | — | HLS |
| 7 | QuantConv | 52×52×16 | 52×52×32 | 3×3 | 1 | 1 | QuantReLU | `MVAU_rtl_2` | **RTL** |
| 8 | MaxPool | 52×52×32 | 26×26×32 | 2×2 | 2 | 0 | — | — | HLS |
| 9 | QuantConv | 26×26×32 | 26×26×56 | 3×3 | 1 | 1 | QuantReLU | `MVAU_hls_1` | **HLS** |
| 10 | MaxPool | 26×26×56 | 13×13×56 | 2×2 | 2 | 0 | — | — | HLS |
| 11 | QuantConv | 13×13×56 | 13×13×104 | 3×3 | 1 | 1 | QuantReLU | `MVAU_hls_2` | **HLS** |
| 12 | MaxPool | 13×13×104 | 13×13×104 | 2×2 | **1** **[F/D2]** | same | — | — | HLS |
| 13 | QuantConv | 13×13×104 | 13×13×208 | 3×3 | 1 | 1 | QuantReLU | `MVAU_hls_3` | **HLS** |
| 14 | QuantConv | 13×13×208 | 13×13×56 | 1×1 | 1 | 0 | QuantReLU | `MVAU_hls_4` | **HLS** |
| 15 | QuantConv | 13×13×56 | 13×13×104 | 3×3 | 1 | 1 | QuantReLU | `MVAU_rtl_3` | **RTL** |
| 16 | QuantConv | 13×13×104 | 13×13×18 | 3×3 | 1 | 1 | QuantHardTanh | `MVAU_rtl_4` | **RTL** |

Structural removals relative to stock YOLOv3-Tiny, all **[P §3.1]**:
- **Detection head detached at export** (coordinate scaling runs on the PS in float).
- **All upsample and concat layers removed** → a single 13×13 detection branch. FINN cannot map
  branched topologies.
- Channel widths reduced so all weights fit in on-chip BRAM.

### 1.2 Architecture self-check — fully closed against Table 6

Before any training or synthesis, the implemented model **must** reproduce paper Table 6 exactly.
We verified that Table 1 (as resolved above) does so, for all ten convolutions, using
`weights = k²·C_in·C_out` and `MAC = H_out·W_out·k²·C_in·C_out`:

| MVU | Weights (ours) | Table 6 **[P]** | MAC (ours) | Table 6 **[P]** | Type |
|---|---|---|---|---|---|
| `MVAU_rtl_0` | 3·3·3·8 = 216 | 216 ✓ | 416²·216 = 37,380,096 | 37,380,096 ✓ | 8b–8b |
| `MVAU_rtl_1` | 3·3·8·8 = 576 | 576 ✓ | 208²·576 = 24,920,064 | 24,920,064 ✓ | 4b–4b |
| `MVAU_hls_0` | 3·3·8·16 = 1,152 | 1152 ✓ | 104²·1152 = 12,460,032 | 12,460,032 ✓ | 4b–4b |
| `MVAU_rtl_2` | 3·3·16·32 = 4,608 | 4608 ✓ | 52²·4608 = 12,460,032 | 12,460,032 ✓ | 4b–4b |
| `MVAU_hls_1` | 3·3·32·56 = 16,128 | 16,128 ✓ | 26²·16128 = 10,902,528 | 10,902,528 ✓ | 4b–4b |
| `MVAU_hls_2` | 3·3·56·104 = 52,416 | 52,416 ✓ | 13²·52416 = 8,858,304 | 8,858,304 ✓ | 4b–4b |
| `MVAU_hls_3` | 3·3·104·208 = 194,688 | 194,688 ✓ | 13²·194688 = 32,902,272 | 32,902,272 ✓ | 4b–4b |
| `MVAU_hls_4` | 1·1·208·56 = 11,648 | 11,648 ✓ | 13²·11648 = 1,968,512 | 1,968,512 ✓ | 4b–4b |
| `MVAU_rtl_3` | 3·3·56·104 = 52,416 | 52,416 ✓ | 13²·52416 = 8,858,304 | 8,858,304 ✓ | 4b–4b |
| `MVAU_rtl_4` | 3·3·104·18 = 16,848 | 16,848 ✓ | 13²·16848 = 2,847,312 | 2,847,312 ✓ | 4b–8b |

**Totals — all exact:**
- 4b–4b MAC: **113,330,048** = Table 6's stated 113,330,048 ✓
- 8b–8b MAC: **37,380,096** ✓ · 4b–8b MAC: **2,847,312** ✓ · grand total 153,557,456
- Weights: **350,696** total; **17,064** at 8b (layers 1 + 16) ✓; **333,632** at 4b ✓

**This is the cheapest and strongest correctness gate in the whole project.** If our model's weight
and MAC counts do not match this table exactly, the architecture is wrong and nothing downstream is
worth running. It is also folding-independent, so it can be checked the moment the model file exists.

### 1.3 Detection head

Not part of the accelerator — detached at export **[P §3.2]** and executed on the PS/host in float.

- **Anchors [R `client/infer_client_v3.py:25`]:** `(10,14), (23,27), (37,58)`, divided by stride.
- **Stride [R same]:** `32`, single detection level (`nl = 1`).
- `nc = 1` **[P §3.1.2 / R:24]**, `no = nc + 5 = 6`, `na = 3` → `3 × 6 = 18` output channels,
  matching `oshape_normal = (1,13,13,18)` **[R `model.pkl`]**.
- **Decode [R `client/models.py`]** — non-standard, matched to the HardTanh output range:
  ```
  xy      = (y[..., 0:2] + 0.5 + grid) * stride
  wh      = (y[..., 2:4] + 1.0) ** 2  * anchor_grid
  obj,cls =  y[..., 4:] / 2 + 0.5
  ```
  Stock YOLOv3 uses `sigmoid(xy)+grid`, `exp(wh)·anchor`, `sigmoid(obj)`. **These substitutions are
  mandatory and must be used at training time as well as inference**, or the loss optimises a
  different parameterisation than inference decodes. The paper describes this layer only as
  "basically only contains scaling operations" (§3.1).

---

## 2. Dataset and preprocessing

### 2.1 Dataset

| Property | Value | Tag |
|---|---|---|
| Source | VisDrone2019-DET, ref [65] | **[P §3.1.2]** |
| Task | Task 1 only — object detection in images | **[P §3.1.2]** |
| Scale | 10,209 images, 2.6 M boxes, 10 categories | **[P §3.1.2]** |
| Classes used | **`car` only**, no merging of van/truck/bus | **[P §3.1.2 / D6]** |
| Label format | YOLO **XYWH**, normalised | **[P §3.1.2]** |
| `nc` / `names` | `1` / `['car']` | **[P] / [R:24]** |
| `car` category index | **verify empirically** against annotation files; documented order is `0 ignored, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor` | **[A1]** |
| `score=0` (ignored regions) | **dropped** | **[A6]** |
| `truncation` / `occlusion` | **all levels kept** | **[A7]** |
| Reporting split | **`val`**, explicitly labelled; `train` metrics also computed for diagnosis | **[A2 / D3]** |
| Storage | `/mnt/data10tb/visdrone_car/` | **[O]** |

Emitted artifacts: `data/visdrone_car.yaml`, `manifest.json` (per-split image/box counts, category
index used, content hash), plus a box-size histogram **[O]** that documents whether the `(10,14)…`
anchor set is appropriate for this data.

### 2.2 Preprocessing

| Stage | Value | Tag |
|---|---|---|
| Resize | **letterbox** to 416×416, aspect-preserving, `auto=False` | **[R `infer_client_v3.py:241`]** |
| Channel order | **BGR → RGB** | **[R:242]** |
| Layout | **NHWC** | **[R `model.pkl` `ishape_normal`]** |
| dtype into accelerator | **UINT8**, un-normalised | **[R `driver.py` `idt`]** |
| Normalisation | performed **on-chip** by a `Div` node inserted before conv 1 | **[P §3.4.1]** |
| `Div` divisor | **255** | **[A3]** — §3.4.1 says only "normalized to values between 0 and 1" |
| Training-time resize | **letterbox**, same as inference | **[A4]** — the paper says only "resizing and color channel inversion" (§3.11.7) |

**Rationale for A4:** train/inference preprocessing mismatch is a classic silent accuracy loss.
Since inference letterboxes **[R]**, training letterboxes too.

---

## 3. Training configuration

Two runs produce Table 5; a third produces the Coral artifact.

| Parameter | Value | Tag |
|---|---|---|
| Epochs | **300** | **[P Table 3]** |
| Batch size | **64** | **[P Table 3]** |
| Image size | **416 × 416** | **[P Table 3]** |
| Optimizer | **SGD** | **[P Table 3]** |
| Early-stopping patience | **30** | **[P Table 3]** |
| Initialisation | **from scratch** (no COCO pretrain) | **[P §3.1.4]** "low-scratch" |
| Hyper-parameter file | Ultralytics **`hyp.scratch-low.yaml`**, vendored and frozen; Table-3 values override | **[A5 / D4]** |
| Augmentation | as in that file | **[A5]** |
| Seed | `0`, logged | **[A8]** |
| mAP implementation | one shared implementation across all runs | **[O]** |
| GPU | 1 × TITAN RTX 24 GB (paper used Tesla P100 16 GB) | **[P §3.1.3]** |
| Package versions | **paper Table 2, verbatim** — torch 1.13.1, torchvision 0.14.1, brevitas 0.9.0, numpy 1.26.3, opencv-python 4.10.0.84, pillow 11.1.0, PyYAML 6.0.2, requests 2.32.3, scipy 1.15.0, tqdm 4.67.1, matplotlib 3.5.1, tensorboard 2.15.2, pandas 2.2.3, seaborn 0.13.2, thop 0.1.1.post2209072238 | **[P Table 2]** |

**Runs:**
1. **T1.2** — unpruned stock YOLOv3-Tiny, FP32, two branches → Table 5 left column.
2. **T1.3** — pruned Table-1 topology, 4W4A QAT → Table 5 right column + Fig. 15.
3. **T1.13** — pruned Table-1 topology, FP32, for the Coral path **[A9 / D5]**.

**Loss:** must be consistent with the §1.3 decode. The quadratic `wh` and affine `obj` terms change
the gradients relative to stock YOLOv5 loss, so the loss implementation is **[O]** and cannot be
copied unmodified. Loss gains come from A5.

---

## 4. Quantization configuration

Quantisation-aware training with **Brevitas 0.9.0** **[P Table 2]**, in a modified Ultralytics YOLOv5
harness (`common.py` + `train.py` modified, `quant_common.py` added) **[P §3.1.1]**.

### 4.1 Bit-width allocation

| Tensor | Width | Tag |
|---|---|---|
| Conv 1 weights | **INT8** | **[P §3.1.1]** "first and last convolution weights were INT8" |
| Conv 16 weights | **INT8** | **[P §3.1.1]** |
| Conv 2–15 weights | **INT4** | **[P §3.1.1]** |
| Activations, conv 1 input | **UINT8** (the image) | **[P Table 6 "8b–8b"] / [R `idt=UINT8`]** |
| Activations, conv 2–15 | **INT4 / UINT4** | **[P §3.1.1]** |
| Activation into conv 16 | **4b** | **[P Table 6 "4b–8b"]** |
| **Network output activation** | **INT8**, `QuantHardTanh` | **[F]** — see below |
| Output dequant scale | **1/127 = 0.00787402** | **[R `scale.npy`]** |

**Why the output is 8-bit [F].** §3.1.1 says "all the remaining weights and activations were quantized
as INT4", which read literally implies a 4-bit output. But the repository pins `odt = INT8`
**[R `driver.py`]** and `scale.npy = 0.00787402`, and `1/127` is exactly the scale mapping a
`HardTanh` range of `[-1, 1]` onto signed 8-bit `[-127, 127]`. A 4-bit output would require a scale of
`1/7`. The output activation is therefore **8-bit**, which also matches the paper's own stated reason
for 8 bits at the end — "to produce more detailed outputs at the end" (§3.1.1). This is forced, not
chosen.

### 4.2 Quantiser details — all `[A]`, all recorded in `configs/quant.yaml`

The paper specifies widths only. It mentions "configurable bit-widths, rounding modes, and
quantization regimes" (§3.1.1) without giving values. Every item below is an
**OUR REIMPLEMENTATION ASSUMPTION**:

| Item | Our choice | ID |
|---|---|---|
| Weight scaling granularity | **per-tensor** | **[A10]** |
| Weight scaling parameterisation | float scale, learned (Brevitas default family) | **[A11]** |
| Activation quantiser signedness | unsigned after `ReLU`, signed after `HardTanh` | **[A12]** |
| Rounding mode | Brevitas default (round-to-nearest-even) | **[A13]** |
| Input quantiser bit width | 8 (matches UINT8 input) | **[A14]** |
| BatchNorm | **none**, per Table 1, which lists no BN layers | **[A15]** |
| Bias quantisation | none (no bias, or absorbed into thresholds) | **[A16]** |
| QAT start | from scratch, not FP32 warm start | **[A17]** (consistent with [P §3.1.4]) |

**Note on A15.** Table 1 lists no BatchNorm, yet the paper's abbreviation list defines "BN — Batch
Normalization". Stock YOLOv3-Tiny has BN after every conv. If BN is present, FINN absorbs it into
`MultiThreshold` during streamlining, so the *hardware* is unaffected either way — but training
dynamics and final accuracy are affected substantially. **Flagged as a leading candidate explanation
if our Table 5 reconstruction diverges**, and as the first ablation to run if so.

---

## 5. QONNX export

| Item | Value | Tag |
|---|---|---|
| Command | `python export.py --weights <best.pt> --img 416 --include finn_onnx` | **[P §3.2]**, verbatim |
| Detection head | **detached** before export | **[P §3.2, §3.1]** |
| Output graph | QONNX (`Quant` nodes feeding `Conv`/`MaxPool`) | **[P Fig. 3]** |
| Output tensor | `(1,13,13,18)` INT8 (or `(1,18,13,13)` pre-transpose) | **[R `model.pkl`]** |
| Export API | `brevitas.export.export_qonnx` | **[A18]** |
| ONNX opset | to be fixed and recorded at implementation time | **[A19]** |

**Acceptance gates:** `qonnx` checker passes; node-type inventory matches Fig. 3; output shape and
dtype match the **[R]** contract exactly.

---

## 6. FINN conversion

FINN **0.10** **[P §3]**. Transform sequence, all names **[P §3.4]**:

| Step | Transform | Effect | Figure |
|---|---|---|---|
| 1 | `ConvertQONNXtoFINN` | folds `Quant` into `Conv`/`MaxPool`; activations → **`MultiThreshold`** | — |
| 2 | tidy-up passes | infer shapes + datatypes; assign unique node names (needed for folding/specialisation) | — |
| 3 | **insert `Div`** before conv 1 | normalise UINT8 → [0,1]; divisor **255** **[A3]** | **Fig. 4** |
| 4 | `Streamline` | eliminate float ops into `MultiThreshold`; **lower `Conv` → `Im2Col` + `MatMul`** | **Fig. 5** |
| 5 | HW mapping | `MatMul` → **MVU**; `Im2Col` → **SWG**; two `Transpose` nodes remain (YOLO NCHW vs. FINN NHWC), handled in the driver | **Fig. 6** |
| 6 | `CreateDataflowPartition` | **exactly one** partition; only a trailing `Mul` (dequant, ×1/127) left outside, run on the PS | **Fig. 7** |

**Hard assertions [O]** — §3.4.4 makes these correctness requirements, so they fail the build rather
than warn:
1. `len(dataflow_partitions) == 1` (FINN cannot map multiple partitions to the PL).
2. Zero floating-point ops inside the partition.
3. Exactly 2 residual `Transpose` + 1 residual `Mul` outside.
4. Partition input dtype `UINT8`, shape `(1,416,416,3)`; output `INT8`, `(1,13,13,18)`.

**Not specified:** the exact tidy-up list — §3.4.1 declines to enumerate it ("the FINN source code can
be consulted"). We will use FINN 0.10's standard tidy-up set and record which passes ran **[A20]**.

---

## 7. Layer specialization

**PAPER-DETERMINED**, decoded from Table 6's MVU instance names and verified against Table 1 by weight
count (§1.2). Emitted as `configs/specialize.json`.

| Convolution | MVU instance | Style | Implied capability |
|---|---|---|---|
| 1 (3→8, 8b–8b) | `MVAU_rtl_0` | **RTL** | DSP48E1 packing, 2 MAC/DSP at 8b **[P Table 4]** |
| 2 (8→8) | `MVAU_rtl_1` | **RTL** | 4 MAC/DSP at 4b–4b |
| 3 (8→16) | `MVAU_hls_0` | **HLS** | LUT style available |
| 4 (16→32) | `MVAU_rtl_2` | **RTL** | 4 MAC/DSP |
| 5 (32→56) | `MVAU_hls_1` | **HLS** | LUT style |
| 6 (56→104) | `MVAU_hls_2` | **HLS** | LUT style |
| 7 (104→208) | `MVAU_hls_3` | **HLS** | LUT style |
| 8 (208→56, 1×1) | `MVAU_hls_4` | **HLS** | LUT style |
| 9 (56→104) | `MVAU_rtl_3` | **RTL** | 4 MAC/DSP |
| 10 (104→18, 4b–8b) | `MVAU_rtl_4` | **RTL** | 2 MAC/DSP at 8b weights |

**5 RTL + 5 HLS.** Capability constraints, all **[P]**:
- RTL MVU uses DSP slices with packing; **no LUT style available** (§3.5.2).
- HLS MVU supports **both** DSP and LUT styles but packs neither efficiently; it is "discouraged"
  except as the escape hatch when DSPs are exhausted (§3.5.3, §3.6.2).
- RTL SWG offers **parallel-window mode**, which "reduces the number of clock cycles … at the expense
  of necessitating a **full SIMD unfolding**" (§3.5.4).
- **DSP packing, Table 4 [P]:** DSP48E1 → **4 MAC/DSP** at 4b act / 4b weight; **2 MAC/DSP** at
  (4,8]b; HLS DSP → **1 MAC/DSP**.

`SpecializeLayers(fpga_part)` defaults to RTL where available, so HLS choices must be applied
explicitly via `ApplyConfig` **[P §3.5]**.

---

## 8. Folding strategy

**The paper's core contribution, and the part it delegates to a repository that does not contain it.**
PE/SIMD values are therefore **re-derived** (D7), never copied. `configs/folding.json` is a
**generated artifact**, not hand-written.

### 8.1 Verified cost model

Adopted only after checking it reproduces the paper's own Table 8 arithmetic exactly:

```
cycles(MVU)  =  (C_out / PE) · (k² · C_in / SIMD) · (H_out · W_out)
latency      =  total_cycles / f_clk
throughput   =  f_clk / max_node_cycles
```

| Check | Computation | Paper **[P Table 8]** |
|---|---|---|
| Latency | 4,586,895 ÷ 100 MHz = **45.87 ms** | 45.87 ms ✓ exact |
| Throughput | 100 MHz ÷ 216,320 = **462.3 FPS** | 462.28 FPS ✓ exact |
| Bottleneck decomposition | 416² + 208² = 173,056 + 43,264 = **216,320** | 216,320 (`StreamingMaxPool_hls_0`) ✓ exact |
| Conv 1 at `SIMD=27, PE=8` | 1 · 1 · 416² = **173,056** < 216,320 ✓ | consistent; and `SIMD=27` over a 3-channel stream is what **[R]** `ishape_folded=(…,1,3)` encodes |

### 8.2 Procedure **[P §3.6.1-3.6.2]**, implemented by us

1. Start **fully folded** (every PE = SIMD = 1).
2. Estimate cycles per node; **unfold the slowest node**; re-estimate resources.
3. Repeat until the slowest node is **fully unfolded** (the bottleneck).
4. Keep small tail layers (MaxPool, SWG, MultiThreshold on 13×13 maps) **fully folded** — "there is no
   advantage in accelerating an already fast layer".
5. When **DSPs are exhausted**, switch MVUs to **HLS-LUT** style to keep unfolding using the abundant
   LUT fabric (§3.6.2). This is the move that produces the paper's speed-up.

### 8.3 Constraints

| Constraint | Value | Tag |
|---|---|---|
| Balance target | **≈216,320 cycles/frame** | **[F]** from §8.1 |
| BRAM_18K budget | 140 | **[P §3]** |
| DSP48E1 budget | 220 | **[P §3]** |
| LUT budget | 53,200 | **[P §3]** |
| Input SIMD (conv 1) | **3** on the input stream (`SIMD = k²C_in = 27` at the MVU) | **[R `ishape_folded`]** |
| Output PE (conv 16) | **1** | **[R `oshape_folded`]** |
| Weights | **all internal (BRAM)**, `mem_mode` internal | **[P §3.8] / [R `number_of_external_weights=0`]** |
| SIMD must divide `k²C_in`; PE must divide `C_out` | FINN requirement | — |
| RTL SWG parallel-window ⇒ `SIMD = k²C_in` | **[P §3.5.4]** | — |
| `ram_style` / `resType` per layer | **[A21]** | — |
| Tie-break when two nodes are equally slow | **[A22]** — unfold the cheaper one first | — |
| Stopping criterion | **[A23]** — §3.6.1's "reasonable resource limits" is unquantified; we stop at ≤95 % of each budget pre-synthesis | — |

### 8.4 Reference targets (context, not objectives)

Paper **[P Table 7]**: BRAM_18K estimated 200 / actual 138 of 140 (98.57 %); LUT estimated 26,694 /
actual 41,605 of 53,200 (78.20 %); DSP estimated 202 / actual 204 of 220 (92.73 %). The paper's
explanation for the gap (§4.2.4): estimated BRAM exceeded the 140 available, so Vivado converted some
to **LUTRAM**, inflating LUT use; Vitis HLS blocks inflate it further.

---

## 9. FIFO strategy

| Item | Value | Tag |
|---|---|---|
| Purpose | absorb per-layer rate mismatch; avoid stalls without wasting BRAM | **[P §3.7]** |
| Method | FIFO depth characterisation by **RTL simulation with `pyverilator`** | **[P §3.7.1]** |
| Depths | **re-derived** — §3.7.2 delegates them to the repository, which does not contain them | **[A24]** |
| Sizing approach | FINN's automated characterisation first, then hand-tune only nodes that still stall | **[A25]** |
| Stall acceptance threshold | to be fixed at implementation time and recorded | **[A26]** |
| Output | `configs/fifo_depths.json` (**generated**) + per-node stall report | **[O]** |

### 9.1 The StreamingMaxPool anomaly — a claim to test, not to assume

**[P §3.7.2]:** the HLS `StreamingMaxPool` was found to require **≈570,000 cycles** against a predicted
**216,320** (≈2.6×), because "Vitis implementations on large feature maps fail to optimize loop
unrollings". **[P §4.2.5]** attributes a **3.43×** effective throughput drop to this single bottleneck,
and notes an RTL MaxPool would fix it but was out of scope.

⚠ **[P Table 8]** separately reports the RTLSim max node at **741,670** cycles. 570,000 and 741,670 are
two different figures for the same layer in the same paper. We will report whichever our simulation
yields and flag the ambiguity rather than reconcile it. See §18, **P8**.

---

## 10. Hardware target

| Item | Value | Tag |
|---|---|---|
| Board | MYiR **Z-turn Board V2** | **[P §3]** |
| Board definition | `myir.com:mys-7z020:part0:2.1` | **[R `visdrone.hwh`]** |
| SoC | Zynq-7020, `xc7z020clg400` | **[P §3]** |
| **Speed grade** | **`-1`** (paper prose says `-2`) | **[R / D1]** |
| PS | dual-core ARM Cortex-A9 @ 750 MHz; 2×32 KiB L1 per core; 512 KiB L2; 256 MiB on-chip SRAM | **[P §3]** |
| PL | 53,200 LUT; ~106,400 FF; 140 × 36 Kb BRAM; 220 DSP48E1 (48-bit MAC) | **[P §3]** |
| DRAM | 1 GiB DDR3 @ 533.33 MHz | **[P §3] / [R `PCW_UIPARAM_DDR_FREQ_MHZ`]** |
| AXI | 4 × 64-bit HP (PL↔memory) + 4 × 32-bit GP | **[P §3] / [R]** |
| PL clock target | **200 MHz**, `clock_mode: aggressive` | **[R `model_info.json`]** |
| PL clock sweep | **50 / 100 / 200 MHz**, set at runtime | **[P §4.2.6, §3.10.2]** |
| Toolchain | **Vivado + Vitis HLS 2022.1** | **[R `visdrone.hwh`] / D10** |
| FINN | **0.10** | **[P §3]** |
| Cost context | board ≈ €100–150 | **[P §3, §5.2]** |

---

## 11. RTLSim configuration

| Item | Value | Tag |
|---|---|---|
| Simulator | **Verilator** | **[P §4.2.2, ref 72]** |
| Harness | `pyverilator` | **[P §3.7.1]** |
| Scope | stitched IP (whole dataflow partition) | **[P §4.2.5]** |
| Uses | (a) FIFO depth characterisation (§9); (b) throughput measurement (T1.10); (c) functional equivalence (T1.12) | **[P] + [O]** |
| Waveform tracing | **disabled** | **[A27]** — VCD over millions of cycles is infeasible on 15 GB RAM |
| Stream length | to be fixed and recorded; steady-state measured after warm-up | **[A28]** — the paper does not state it |
| Metrics reported | cycles/frame (throughput) **and** first-frame latency, **reported separately** | **[O]** |
| Clock assumed for FPS conversion | 100 MHz, matching Table 8 | **[P Table 8]** |

**Reference [P Table 8]:** max node 741,670 cycles; latency 7.41 ms; **134.83 FPS**. The paper notes
that replacing the auto-generated Python driver with bare-metal C allowed this full 134 FPS to be
reached on real hardware (§4.2.5).

⚠ Table 8's RTLSim latency (7.41 ms = 741 k cycles at 100 MHz) **equals its max-node count**, which is
dimensionally odd for a pipelined design whose estimated latency was 4.59 M cycles. We therefore
report our two figures separately rather than trying to match this row. See §18, **P9**.

---

## 12. FPGA synthesis configuration

| Step | Transform | Tag |
|---|---|---|
| 1 | `PrepareIP` — export RTL to the build directory | **[P §3.9]** |
| 2 | `HLSSynthIP` — Vitis HLS C→RTL (slow) | **[P §3.9]** |
| 3 | `CreateStitchedIP` — stitch IPs; adds **`idma0`** and **`odma0`** for DRAM in/out | **[P §3.10]** |
| 4 | `ZynqBuild` — Vivado synthesis + place & route | **[P §3.10.1]** |
| 5 | PYNQ driver generation (MMIO class, packing helpers, bitfile loader with runtime clock + batch size) | **[P §3.10.2]** |

| Setting | Value | Tag |
|---|---|---|
| Part | `xc7z020clg400-1` | **[D1]** |
| Board files | `myir.com:mys-7z020:part0:2.1` — **to obtain** from MYIR, or author a minimal custom platform | **[R]** |
| Target clock | 200 MHz | **[R `model_info.json`]** |
| Clock mode | `aggressive` | **[R]** |
| Synthesis / implementation strategy | Vivado defaults unless timing fails | **[A29]** |
| Parallel HLS workers | **`NUM_DEFAULT_WORKERS=1`** | **[O]** — forced by 15 GB RAM |
| Build directory | `/mnt/data10tb/finn_build/` | **[O]** — `/` has only 74 GB free |

**Outputs:** `model.bit`, `model.hwh`, generated PYNQ driver; stitched block design (**Fig. 12**);
device floorplan (**Fig. 13**); utilisation report (**Fig. 16**); timing report with WNS/TNS
(**Fig. 17**); resource table (**Table 7**).

**Power estimation (T1.11):** Vivado `report_power` on the routed design. Switching activity
**vectorless** **[A30]** — §4.3.1 says only "the Vivado power analysis tool" and that estimates are
"generated under worst-case or synthetic conditions", which implies vectorless. Record the confidence
level Vivado reports. Reference **[P Tables 11, 12]**: `top_wrapper` 3.129 W
(`StreamingDataflowPartition_1` 1.554 W, `zynq_ps` 1.497 W); total on-chip 3.358 W including 0.229 W
static; BRAM 0.262 W, DSP 0.207 W, Signals 0.612 W, Slice Logic 0.428 W, Clocks 0.124 W, PS7 1.496 W.
Vivado max ambient estimate 46.3 °C **[P §4.3.2]**.

---

## 13. Evaluation metrics

### 13.1 Accuracy (T1.2, T1.3, T1.12, T1.13)

| Metric | Definition | Tag |
|---|---|---|
| Precision (P) | single-class, `car` | **[P Table 5]** |
| Recall (R) | single-class | **[P Table 5]** |
| mAP@0.5 | IoU = 0.5 | **[P Table 5]** |
| Box / Object / Classification loss | final-epoch values (cls ≡ 0 for one class) | **[P Table 5]** |
| Split | **`val`**, labelled; `train` also computed | **[A2 / D3]** |
| NMS | `conf_thres=0.10`, `iou_thres=0.10`, `max_det=20` | **[R `infer_client_v3.py:342`]** |

One shared mAP implementation across all runs **[O]**, so T1.2 ↔ T1.3 is internally valid even if
neither matches the paper's absolute numbers.

### 13.2 Hardware correctness (T1.12 — ours, no paper counterpart)

| Metric | Acceptance |
|---|---|
| QONNX vs. cppsim vs. rtlsim output | **exact integer equality** |
| Brevitas vs. QONNX | ≤ export rounding tolerance **[A31]** |
| mAP@0.5 delta, Brevitas → rtlsim | ≈ 0 |
| Fraction of outputs bit-exact | reported |

**Why this exists:** §4.2.2 claims only that RTL simulation "confirmed the correct working of the
implemented system", with no quantitative evidence, and the repository's `validate.py` is unmodified
FINN MNIST/CIFAR-10 boilerplate. Nothing in either source demonstrates that the FPGA computes the
trained function.

### 13.3 Performance

| Metric | Where | Tag |
|---|---|---|
| Total cycles, max-node cycles | T1.8 | **[P Table 8]** |
| Latency (ms) | T1.8, T1.10, T2.1 | **[P Table 8]** |
| Throughput (FPS) | T1.8, T1.10, T2.1 | **[P Tables 8, 9]** |
| DRAM in/out bandwidth (Mb/s) | T2.1 | **[P Table 9]** |
| Per-stage latency: preprocess, driver exec, rescale, detect, NMS, box process | T2.2 | **[P Table 10]** |
| End-to-end stream FPS, network latency | T2.4 | **[P §4.4]** |

**Throughput definition — binding [R `driver_base.py:421-448`].** `throughput_test()` times **only**
`execute_on_buffers()`, with buffers already resident in DRAM:
```python
start = time(); self.execute_on_buffers(); end = time()
res["throughput[images/s]"] = self.batch_size / (end - start)
```
`fold_input`, `pack_input`, `copy_input_data_to_device`, `copy_output_data_from_device`,
`unpack_output` and `unfold_output` are timed **separately and excluded**. Therefore
**paper Table 9's FPS is a fabric+DMA figure, not end-to-end**, while Table 10 is the end-to-end
Python-driver breakdown. **Our T2.1 must use this identical definition** or the numbers are not
comparable. This is the single most important methodological detail recovered from the repository.

### 13.4 Power and efficiency

| Metric | Where | Tag |
|---|---|---|
| Vivado estimated power, per module and per resource type | T1.11 | **[P Tables 11, 12]** |
| Measured board power (5 V rail) at idle / 50 / 100 / 200 MHz | T2.3 | **[P Table 13]** |
| **Measured** PL-only power (loaded − unloaded difference) | T2.3 | **[O]** — the paper derives this arithmetically |
| Efficiency FPS/W (whole-package power) | T2.3 | **[P Table 14]** |
| Die temperature (IR) | T2.3 | **[P §4.3.2]** |
| **Measured** Coral power | T3.1 | **[O]** — the paper uses a datasheet estimate |

### 13.5 Resource utilisation

LUT, DSP, BRAM_18K: estimated vs. actual vs. available vs. % **[P Table 7]**; plus WNS/TNS timing
closure **[P Fig. 17]**.

---

## 14. Experiment list

| ID | Paper § | Experiment | Tier |
|---|---|---|---|
| **T1.1** | §3.1.2 | VisDrone → YOLO `car`-only dataset | 1 |
| **T1.2** | §4.1 Table 5 (left) | Unpruned YOLOv3-Tiny FP32 baseline | 1 |
| **T1.3** | §3.1, §4.1 Table 5 (right), Fig. 15 | Pruned + 4W4A QAT training | 1 |
| **T1.4** | §3.2, Fig. 3 | QONNX export, head detached | 1 |
| **T1.5** | §3.4, Figs. 4–7 | FINN front-end transformations | 1 |
| **T1.6** | §3.5–3.6, Figs. 8, 10, 11 | Specialisation + folding derivation | 1 |
| **T1.7** | §3.7 | FIFO sizing + MaxPool anomaly | 1 |
| **T1.8** | §4.2.3 Table 6, §4.2.5 Table 8 (est.) | Analytical MAC / cycle / throughput model | 1 |
| **T1.9** | §3.9–3.10, §4.2.1/4, Figs. 12, 13, 16, 17, Table 7 | HLS synth → stitch → ZynqBuild | 1 |
| **T1.10** | §4.2.5 Table 8 (RTLSim) | RTL simulation throughput | 1 |
| **T1.11** | §4.3.1 Tables 11, 12 | Vivado power estimation | 1 |
| **T1.12** | — **[O]** | Functional equivalence PyTorch ↔ FINN | 1 |
| **T1.13** | §3.3 | FP32 → TFLite INT8 PTQ → Edge TPU compile | 1 |
| **T2.1** | §4.2.5 Table 8 (actual), §4.2.6 Table 9 | On-board throughput sweep (freq × batch) | 2 |
| **T2.2** | §4.2.6 Table 10 | Per-stage latency breakdown | 2 |
| **T2.3** | §4.3.2 Tables 13, 14, Fig. 18 | Measured power + thermal | 2 |
| **T2.4** | §3.11, §4.4 | End-to-end WebSocket streaming | 2 |
| **T2.5** | §4.5 Fig. 19 | Field footage inference | 2 |
| **T3.1** | §5.1 Table 15 | Coral Edge TPU benchmark | 3 |
| **T3.2** | §5.2 Table 16 | Literature comparison (desk work) | 3 |

Full 14-field design for each is in `experiment_reconstruction_plan.md`.

---

## 15. Expected outputs

| Paper artifact | Our output | Produced by |
|---|---|---|
| Table 1 | implemented model; weight/MAC self-check vs. Table 6 | T1.3 (checkable at T1.8) |
| Table 5 | `results/accuracy.md` — both columns, split labelled | T1.2, T1.3 |
| Table 6 | `results/mac_table.md` | T1.8 |
| Table 7 | `results/resources.md` — estimated vs. actual | T1.9 |
| Table 8 | `results/perf_three_way.md` — estimated / RTLSim / actual | T1.8, T1.10, T2.1 |
| Table 9 | `results/throughput_sweep.md` — **with mean ± σ** | T2.1 |
| Table 10 | `results/latency_breakdown.md` | T2.2 |
| Tables 11, 12 | `results/power_estimated.md` | T1.11 |
| Tables 13, 14 | `results/power_measured.md` | T2.3 |
| Table 15 | `results/coral_comparison.md` — **measured** Coral power | T1.13 + T3.1 |
| Table 16 | `results/literature_comparison.md` — **batch regime per row** | T3.2 |
| Fig. 3 | `results/figs/qonnx_graph.*` | T1.4 |
| Figs. 4–7 | `results/figs/finn_stage_*.*` | T1.5 |
| Figs. 8, 10, 11 | `results/figs/folding_*.*` | T1.6 |
| Fig. 15 | `results/figs/training_curves.*`, `pr_curve.*` | T1.3 |
| Figs. 12, 13, 16, 17 | `results/figs/vivado_*.*` | T1.9 |
| Fig. 18 | `results/figs/bench_setup.*` | T2.3 |
| Fig. 19 | `results/figs/field_frames.*` + **quantified P/R** | T2.5 |
| — **[O]** | `results/equivalence_report.md` | T1.12 |
| — **[O]** | `results/dataset_stats.md` | T1.1 |

Generated configs (artifacts, not hand-written): `configs/folding.json`, `configs/fifo_depths.json`.
Every `results/` entry carries a manifest recording the config hash, tool versions, and git revision.

---

## 16. Paper values versus values we must determine ourselves

### 16.1 Fixed by the paper — we implement, we do not choose

Architecture (Table 1, with the two documented corrections) · bit-width allocation INT8/INT4 ·
activation types `QuantReLU`/`QuantHardTanh` · structural removals (head, upsample, concat) · dataset,
task, single `car` class, XYWH format · Table 3 training parameters (300 / 64 / 416² / SGD / 30) ·
Table 2 package versions · Brevitas as the QAT tool · FINN 0.10 · the full transform sequence of §3.4
· the single-partition constraint · RTL/HLS specialisation per layer (Table 6) · Table 4 DSP packing ·
RTL/HLS capability matrix · parallel-window constraint · the folding *procedure* of §3.6 · the FIFO
*methodology* of §3.7 · the synthesis sequence of §3.9–3.10 · device resource limits · the clock sweep
points · the metric set · the streaming architecture, 16-byte header, `.fnn` layout, batch cap 100.

### 16.2 Recovered from the repository — facts we adopt into code we write

Anchors `(10,14),(23,27),(37,58)` at stride 32 · the non-standard decode formulas · dequant scale
1/127 · I/O dtypes and shapes · input SIMD 3 / output PE 1 · `number_of_external_weights = 0` ·
letterbox + BGR→RGB preprocessing · NMS thresholds · Vivado **2022.1** · board
`myir.com:mys-7z020:part0:2.1` · speed grade **`-1`** · 200 MHz / `aggressive` · `throughput_test()`
semantics · WebSocket port 8765 / queue 30.

### 16.3 Values we must determine ourselves

**Derived (not free choices) —** PE/SIMD per layer, via the §8.1 cost model against the ≈216,320-cycle
target · FIFO depths, via pyverilator characterisation · the `car` category index, by inspecting the
annotation files.

**Genuine assumptions —** all 36 items in §17.

### 16.4 Paper reference values recorded as context, **not** as targets

Per your instruction not to optimise toward the reported numbers, these are recorded so divergence is
*interpretable*, not so it is minimised:

| Quantity | Paper value |
|---|---|
| Pruned P / R / mAP@0.5 | 48.16 % / 17.16 % / 17.68 % |
| Baseline P / R / mAP@0.5 | 67.06 % / 38.44 % / 41.76 % |
| Resources (actual) | 138/140 BRAM, 41,605/53,200 LUT, 204/220 DSP |
| Estimated / RTLSim / actual FPS @100 MHz | 462.28 / 134.83 / 104.39 |
| Throughput @200 MHz, batch 100 | **208.69 FPS** |
| Measured power @200 MHz | **2.55 W** |
| Efficiency @200 MHz, batch 100 | **81.88 FPS/W** |
| End-to-end streaming | 30 FPS wired; ~13 ms Ethernet / ~130 ms WiFi |
| Coral | 103.2 FPS @250 MHz / 148.7 FPS @500 MHz |

---

## 17. All assumptions

Every item below is an **OUR REIMPLEMENTATION ASSUMPTION** unless its class says otherwise. Each is
recorded in `DECISIONS.md` with a date and rationale at implementation time.

| ID | Assumption | Our value | Affects | Linked |
|---|---|---|---|---|
| **A1** | `car` category index in VisDrone annotations | **verify empirically**; do not assume | dataset | D6 |
| **A2** | Reporting split | `val`, labelled; `train` also computed | accuracy | **D3** |
| **A3** | `Div` divisor for input normalisation | `255` | accuracy, graph | — |
| **A4** | Training-time resize | letterbox, matching inference | accuracy | — |
| **A5** | Hyper-parameter file | Ultralytics `hyp.scratch-low.yaml`, vendored + frozen | accuracy | **D4** |
| **A6** | VisDrone `score=0` ignored regions | dropped | dataset | D6 |
| **A7** | `truncation` / `occlusion` attributes | all levels kept | dataset | D6 |
| **A8** | Random seed | `0`, logged | reproducibility | — |
| **A9** | Coral comparison topology | pruned Table-1 topology | Table 15 validity | **D5** |
| **A10** | Weight scaling granularity | per-tensor | accuracy | — |
| **A11** | Weight scale parameterisation | learned float (Brevitas default family) | accuracy | — |
| **A12** | Activation signedness | unsigned after ReLU, signed after HardTanh | accuracy | — |
| **A13** | Rounding mode | Brevitas default (round-half-even) | accuracy | — |
| **A14** | Input quantiser width | 8 bit | accuracy | — |
| **A15** | **BatchNorm presence** | **none**, per Table 1 | **accuracy — high risk** | §4.2 note |
| **A16** | Bias quantisation | no bias / absorbed into thresholds | accuracy | — |
| **A17** | QAT initialisation | from scratch, no FP32 warm start | accuracy | — |
| **A18** | Brevitas export API | `export_qonnx` | export | — |
| **A19** | ONNX opset | fix and record at implementation | export | — |
| **A20** | FINN tidy-up pass list | FINN 0.10 standard set, recorded | graph | §3.4.1 declines to enumerate |
| **A21** | `ram_style` / `resType` per layer | FINN defaults unless resources force otherwise | resources | — |
| **A22** | Folding tie-break | unfold the cheaper of two equally slow nodes | folding | — |
| **A23** | Folding stopping criterion | ≤95 % of each budget pre-synthesis | folding | §3.6.1 unquantified |
| **A24** | FIFO depths | re-derived by characterisation | throughput | §3.7.2 delegates to a repo that lacks them |
| **A25** | FIFO sizing approach | automated characterisation, then hand-tune stalling nodes | throughput | — |
| **A26** | Stall acceptance threshold | fix and record | throughput | — |
| **A27** | RTLSim waveform tracing | **disabled** | feasibility (15 GB RAM) | — |
| **A28** | RTLSim stream length / warm-up | fix and record | throughput | — |
| **A29** | Vivado synth/impl strategy | defaults unless timing fails | resources, timing | — |
| **A30** | Power-estimation switching activity | vectorless | Tables 11, 12 | §4.3.1 implies it |
| **A31** | Brevitas→QONNX equality tolerance | export rounding only | T1.12 | **[O]** |
| **A32** | Throughput repetition count | ≥10, report mean ± σ | T2.1 | **[O]** improvement |
| **A33** | Latency-breakdown frame count | fix and record | T2.2 | paper gives n>1 only |
| **A34** | Power sampling / averaging window | fix and record; report instrument uncertainty | T2.3 | **[O]** |
| **A35** | T2.5 footage | **deferred**; substitute only with a labelled deviation | scope | **D8** |
| **A36** | Server WebSocket queue cap | fix and record | T2.4 | §4.4 says only "a high value" |

Plus the two cross-cutting decisions that are *not* assumptions: **D1** speed grade `-1`
(repository-determined), **D2** MaxPool stride 1 (arithmetically forced), **D7** folding re-derivation
(project mandate), **D10** FINN 0.10 + Vivado 2022.1 (paper + repository, pairing to verify).

---

## 18. All known inconsistencies in the paper

| # | Inconsistency | Evidence | Our resolution | Impact |
|---|---|---|---|---|
| **P1** | §3.1 says channel widths were "reduced by a factor of 5"; Table 1 shows factors of 2–4.92 (16→8, 32→8, 64→16, 128→32, 256→56, 512→104, 1024→208) | §3.1 line 458 vs. Table 1 | **Table 1 is authoritative** — independently corroborated by Table 6 weight counts (§1.2). Treat "factor of 5" as loose prose | architecture |
| **P2** | Table 1 row 12: `MaxPool 13×13×104 → 13×13×104, 2×2, stride 2` — impossible | Table 1 line 507 | **stride 1, 'same' padding** — forced by the printed shapes, by stock YOLOv3-Tiny, and by Table 6's layer-13 MAC count (`13²·9·104·208 = 32,902,272` exact; at 6×6 it would be 7,008,768) | **critical — architecture** |
| **P3** | Final detection conv is **3×3** (Tables 1 and 6 agree: 16,848 = 3·3·104·18); stock YOLOv3-Tiny uses 1×1 | Tables 1, 6 | **3×3** — two independent tables agree, so this is a deliberate deviation, not an error | architecture |
| **P4** | Abstract and §6 claim "**three to seven**×" FPS/W gain; §5.2 claims "**7 to 24 times**" and 7.3× over LPYOLO | Abstract vs. §5.2 vs. Table 16 | Report both; target neither | reporting only |
| **P5** | §3 specifies `XC7Z020CLG400-**2**`; the shipped `visdrone.hwh` says `SPEEDGRADE="-1"` | §3 line 418 vs. `.hwh` | **`-1`** (D1) — the only machine-readable evidence, and the conservative direction | timing, folding |
| **P6** | Table 5 is captioned "**Validation** results"; §4.1 body says the figures are "on the **training** set" | line 1144 vs. lines 1122–1127 | **report `val`**, labelled; also compute `train` for diagnosis (D3) | accuracy comparability |
| **P7** | §3 claims the repo holds "the complete code base … and the Jupyter notebooks used for training"; §3.7.2 claims it holds the folding config and all FIFO depths | repository inspection | **Both false.** The repo holds only the streaming driver. The two delegated artifacts are absent, so both must be re-derived (D7, A24) | **the whole reconstruction** |
| **P8** | §3.7.2 reports the HLS MaxPool at **≈570,000** cycles; Table 8 reports the RTLSim max node at **741,670** | §3.7.2 vs. Table 8 | Report what our simulation yields; flag the ambiguity rather than reconcile it | FIFO, throughput |
| **P9** | Table 8's RTLSim latency (7.41 ms = 741 k cycles @100 MHz) **equals** its max-node cycle count — dimensionally odd for a pipeline whose estimated latency was 4.59 M cycles | Table 8 | Report per-frame throughput cycles and first-frame latency **separately**; do not attempt to match this row | performance reporting |
| **P10** | Table 16 uses the **batch-100** latency (4.79 ms = 479.189/100) while Table 15 uses **batch-1** (6.07 ms). Mixing regimes across the two comparison tables flatters Table 16 | Tables 9, 15, 16 | State the batch regime **per row** in both tables | comparison fairness |
| **P11** | §3.1.1 says "all the remaining weights and activations were quantized as INT4", implying a 4-bit output; but `odt = INT8` and `scale = 1/127` require an **8-bit** output | §3.1.1 vs. `driver.py`, `scale.npy` | **Output activation is 8-bit** — arithmetically forced (1/127 maps HardTanh `[-1,1]` to `[-127,127]`; 4-bit would need 1/7), and it matches §3.1.1's own stated reason for 8 bits at the end | quantisation correctness |
| **P12** | The repo's only committed measurement (`nw_metrics.txt`: **23.49 FPS @100 MHz, batch 10**) is ~4.4× below Table 9's 104.39 FPS, and was produced with `resizer.bit` — a **different MD5** from `visdrone.bit` | `nw_metrics.txt`, `md5sum` | Treat Tables 9/10/13/14 as **single-source, uncorroborated**. Our T2.1 measurements will be the only independent evidence | expectation setting |

---

## 19. Hardware-dependent experiments

### Requires the Zynq board (MYiR Z-turn V2) — absent

| ID | Experiment | Paper artifacts blocked |
|---|---|---|
| **T2.1** | On-board throughput sweep | Table 8 (actual column), **Table 9** |
| **T2.2** | Per-stage latency breakdown | **Table 10** |
| **T2.3** | Measured power + thermal | **Tables 13, 14**, Fig. 18, the 53.1 °C figure |
| **T2.4** | End-to-end WebSocket streaming | §4.4 (30 FPS, 13/130 ms) |
| **T2.5** | Field footage inference | Fig. 19 |

### Additionally requires instruments — absent

**T2.3** needs a **5 V DC bench supply with current readout** and an **IR thermal camera**
**[P §4.3, Fig. 18]**. Without these, **every FPS/W number in the paper is unreproducible even with
the board**, including the headline 81.88 FPS/W — power cannot be inferred from throughput.

### Requires a Coral Edge TPU — absent

| ID | Experiment | Blocked |
|---|---|---|
| **T3.1** | Coral benchmark (standard + max driver modes, 250/500 MHz) | **Table 15** |

**T1.13 is not blocked** — TFLite conversion, INT8 PTQ and `edgetpu_compiler` all run host-only. Only
execution needs the device.

### Net consequence

The paper's headline claim — **208 FPS at 2.55 W, 81.88 FPS/W** — depends on T2.1 **and** T2.3, i.e.
on the board **and** both instruments. All three are absent. **D9 is awaiting your answer.**

---

## 20. Experiments that can run without the physical FPGA board

**13 of 20 experiments**, covering the entire design-and-compile flow. Nothing here needs the board,
the instruments, or the Coral.

| ID | Experiment | Needs | Paper artifacts produced |
|---|---|---|---|
| **T1.1** | Dataset preparation | CPU, ~3 GB disk | dataset stats **[O]** |
| **T1.2** | FP32 baseline training | GPU | Table 5 left |
| **T1.3** | 4W4A QAT training | GPU | Table 5 right, Fig. 15 |
| **T1.4** | QONNX export | CPU | Fig. 3 |
| **T1.5** | FINN front-end | CPU + Docker | Figs. 4–7 |
| **T1.6** | Specialisation + folding | CPU | Figs. 8, 10, 11, `folding.json` |
| **T1.7** | FIFO sizing | CPU ⚠ RAM-heavy | `fifo_depths.json`, MaxPool anomaly |
| **T1.8** | Analytical model | CPU | **Table 6**, Table 8 (estimated) |
| **T1.9** | Synthesis → bitstream | Vivado 2022.1 ⚠ heavy | **Table 7**, Figs. 12, 13, 16, 17 |
| **T1.10** | RTLSim throughput | Verilator ⚠ heaviest | Table 8 (RTLSim) |
| **T1.11** | Vivado power estimation | Vivado | **Tables 11, 12** |
| **T1.12** | Functional equivalence | CPU + GPU | equivalence report **[O]** |
| **T1.13** | TFLite + Edge TPU compile | GPU + compiler | TFLite artifacts, op-mapping report **[O]** |

**Reproducible without the board:** the architecture (Table 1 ⇄ Table 6 cross-check), both accuracy
columns (Table 5), the folding and resource outcome (Table 7), the analytical and simulated throughput
(Table 8, two of three columns), and the estimated power (Tables 11, 12). **Plus a result the paper
does not have at all:** quantitative proof that the hardware computes the trained function (T1.12).

**Not reproducible without it:** every measured number — Tables 9, 10, 13, 14, 15, 16 and the
abstract's claims.

### Host constraints (measured, not assumed)

| Resource | Observed | Implication |
|---|---|---|
| GPU | TITAN RTX 24 GB | ✅ exceeds the paper's P100 16 GB |
| **RAM** | **15 GB total, ~10 GB free** | ⚠ **binding constraint** — Verilator (T1.7, T1.10) and Vitis HLS are the risks. Mitigations: `NUM_DEFAULT_WORKERS=1` **[O]**, tracing disabled **[A27]** |
| CPU | 6 cores | ⚠ serialised HLS; hours per build |
| Disk `/` | 74 GB free (84 % used) | ⚠ nothing large here |
| Disk `/mnt/data10tb` | 2.9 TB free | ✅ Vivado, FINN build dir, dataset, checkpoints |
| Docker | 28.1.1 | ✅ |
| Xilinx tools | **none** | install 2022.1 to `/mnt/data10tb` |
| VisDrone, brevitas, FINN | **absent** | T1.1 / T1.3 / T1.5 blocked until installed |

---

## 21. Recommended implementation order

Ordered so that **each step is validated by the cheapest available check before the next step depends
on it**, and so that the two highest-risk unknowns — the 15 GB RAM ceiling and the FINN 0.10 ⇄ Vivado
2022.1 pairing — are tested **before** any GPU time is spent.

### Phase 0 — Toolchain feasibility spike (no paper experiment; ~1 day)

**Start here, not at T1.1.** Install FINN 0.10 + Vivado/Vitis 2022.1 on `/mnt/data10tb` and push one
of FINN's **own** example networks (TFC or CNV) end-to-end: front-end → folding → HLS synth → stitch →
`ZynqBuild` for `xc7z020clg400-1` → rtlsim.

Validates: (a) the **15 GB RAM** ceiling against Verilator and Vitis HLS — the top technical risk;
(b) whether **FINN 0.10 actually pairs with Vivado 2022.1** (**D10**'s open question); (c) whether the
**MYIR board file** works or a custom platform is needed; (d) the real wall-clock cost of one
synthesis iteration on 6 cores.

> **Failing here costs a day. Discovering it after T1.3 costs GPU-days of training first.**
> If RAM proves insufficient, we re-scope before spending anything.

### Phase 1 — Architecture, validated on paper arithmetic alone (hours, no training)

1. **Write the model** (`models/quant_yolov3_tiny.py`) to §1.1.
2. **Run the Table 6 self-check** (§1.2): assert all 10 weight counts and all 10 MAC counts, and the
   three totals. **This is the smallest experiment that validates the methodology** — pure arithmetic,
   no data, no GPU, no tools, and it conclusively confirms or refutes our reading of Table 1
   (including the **D2** stride correction and the **P1**/**P3** resolutions).
3. **Build the analytical cost model** (`tools/cycle_model.py`, part of **T1.8**) and reproduce
   Table 8's estimated column: 4,586,895 cycles, 45.87 ms, 462.28 FPS, bottleneck 216,320.

No step here can fail expensively, and all three gate everything downstream.

### Phase 2 — Data and training (GPU-days)

4. **T1.1** dataset conversion + stats; verify the `car` index (**A1**) and freeze the manifest.
5. **T1.3** QAT training — the pruned 4W4A model. Run this **before** T1.2: the hardware flow depends
   only on T1.3, so it unblocks Phases 3–4 while the baseline is still training.
6. **T1.2** FP32 baseline → completes Table 5.
7. **T1.4** QONNX export; verify shape/dtype against the **[R]** I/O contract.

### Phase 3 — FINN compilation, cheapest-first (days)

8. **T1.5** front-end transforms + the four hard assertions (§6). Graph-level, fast, fails loudly.
9. **T1.6** specialisation (fixed by Table 6) + folding derivation → `folding.json`, Figs. 8/10/11.
   Estimation only, no synthesis — so iterate here freely before spending synthesis hours.
10. **T1.8** complete: emit the Table 6 and Table 8-estimated reports from the real graph.
11. **T1.7** FIFO sizing. **First RAM-heavy step** — Phase 0 will already have told us whether this is
    viable. Independently test the MaxPool anomaly (**P8**).

### Phase 4 — Synthesis and simulation (days per iteration)

12. **T1.9** HLS synth → stitch → `ZynqBuild` → Table 7, Figs. 12/13/16/17. Expect iteration back to
    T1.6 if resources or timing miss.
13. **T1.11** Vivado power estimation → Tables 11, 12. Cheap once T1.9 exists.
14. **T1.10** RTLSim throughput → Table 8 RTLSim column. **Heaviest step**; do it after T1.9 so a
    failure here does not block the bitstream.
15. **T1.12** functional equivalence PyTorch ↔ QONNX ↔ cppsim ↔ rtlsim. **Do not proceed to Tier 2
    without this passing** — deploying a model we have not proven correct would make every Tier-2
    measurement uninterpretable. This is the gate the paper never had.

### Phase 5 — Coral host-side path (parallelisable with Phases 3–4)

16. **T1.13** FP32 run (**A9**) → TFLite INT8 PTQ → `edgetpu_compiler` + op-mapping report. Needs no
    Coral. Can run on the GPU whenever Phase 2 frees it.

### Phase 6 — Hardware, **gated on D9**

17. **T2.1** throughput sweep — using the exact `throughput_test()` semantics of §13.3, ≥10 reps.
18. **T2.2** per-stage latency breakdown.
19. **T2.3** measured power + thermal — **additionally gated on the bench supply and IR camera**.
20. **T2.4** end-to-end streaming (with a >30 FPS source, to separate camera limit from pipeline limit).
21. **T2.5** field footage — **gated on D8**.

### Phase 7 — Comparison (desk work)

22. **T3.1** Coral benchmark — gated on the device; with **measured** power.
23. **T3.2** literature comparison — batch regime stated per row (**P10**).

### Critical path summary

```
Phase 0 spike ─► Table 6 self-check ─► T1.3 ─► T1.4 ─► T1.5 ─► T1.6 ─► T1.7 ─► T1.9 ─► T1.12 ─► Tier 2
   (1 day)         (hours, free)      (GPU-days)                   (iterate)   (days)   (gate)   (D9)
```

Three gates, each cheaper than what follows it: **the toolchain spike** (before any GPU time), **the
Table 6 arithmetic check** (before any training), and **T1.12 equivalence** (before any hardware
measurement).
