# Decision log

Running record of every choice that is not a `PAPER FACT`. Append-only: when a
decision changes, add a dated entry rather than editing history.

The full six-field treatment of D1–D10 lives in `implementation_decisions.md`;
the assumption catalogue A1–A36 lives in `final_implementation_spec.md` §17. This
file records **what the code actually does**, with the file that does it.

---

## Status at implementation time (2026-10-06)

### Decisions D1–D10

| # | Decision | Value in code | Class | Where |
|---|---|---|---|---|
| D1 | FPGA speed grade | `xc7z020clg400-1` | REPOSITORY-DETERMINED | `models/arch_spec.py:FPGA_PART` |
| D2 | L12 MaxPool stride | 2×2, **stride 1**, `same` pad | ARITHMETICALLY FORCED | `arch_spec.LAYERS[11]`, `models/quant_yolov3_tiny.SamePadMaxPool` |
| D3 | Reporting split | `val`, labelled; `train` also computed | **OUR ASSUMPTION** | `val.py --split`, `utils/metrics.py` |
| D4 | Hyper-parameters | `configs/hyp.scratch-low.yaml`, vendored + frozen | **OUR ASSUMPTION** | `configs/hyp.scratch-low.yaml` |
| D5 | Coral topology | pruned Table-1 | **OUR ASSUMPTION** | `tools/export_tflite.py` |
| D6 | VisDrone classes | `car` only, no merging | PAPER-DETERMINED | `tools/visdrone_to_yolo.py` |
| D7 | PE/SIMD folding | **derived**, never copied | PROJECT MANDATE | `tools/folding_search.py` → `configs/folding.json` |
| D8 | T2.5 footage | deferred | **OUR ASSUMPTION** | `tools/run_video.py --status` |
| D9 | Tier-2/3 hardware | **PENDING** | USER DECISION | every `bench/` script blocks on it |
| D10 | FINN / Vivado | FINN 0.10 + Vivado 2022.1, **pairing UNVERIFIED** | PAPER + REPOSITORY | `requirements/finn.env.sh.example`, `step_04_synth.preflight` |

### Assumptions A1–A36 — where each one lives

| ID | Assumption | Implemented in |
|---|---|---|
| A1 | `car` category index **verified, not assumed** | `tools/visdrone_to_yolo.py --verify-index` |
| A2 | reporting split `val` | `val.py`, `train.py` |
| A3 | `Div` divisor 255 | `configs/quant.yaml`, `finn_build/custom_steps.insert_input_div` |
| A4 | training letterbox | `utils/datasets.py`, `utils/preprocess.py` |
| A5 | hyp preset | `configs/hyp.scratch-low.yaml` |
| A6 | drop `score=0` | `tools/visdrone_to_yolo.py` |
| A7 | keep all truncation/occlusion | `tools/visdrone_to_yolo.py` |
| A8 | seed 0 | `train.py --seed` |
| A9 | Coral uses pruned topology | `tools/export_tflite.py` |
| A10–A14, A16 | quantiser details | `configs/quant.yaml`, `models/quant_common.py` |
| **A15** | **BatchNorm ON since 2026-10-16 (was OFF) — highest risk** | `configs/quant.yaml`, `train.py --no-batchnorm` ablation flag |
| A17 | QAT from scratch | `configs/quant.yaml` |
| A18, A19 | export API / opset | `export.py`, `configs/quant.yaml` |
| A20 | tidy-up pass list | `finn_build/step_01_frontend.py` manifest |
| A21 | `ram_style`/`resType` | `finn_build/step_02_folding.build_finn_config` |
| A22, A23 | folding tie-break / stop | `tools/folding_search.py` |
| A24–A26 | FIFO depths / approach / threshold | `finn_build/step_03_fifos.py` |
| A27 | no waveform tracing | `step_03_fifos.py`, `step_05_rtlsim.py`, `finn.env.sh.example` |
| A28 | rtlsim stream length | `step_05_rtlsim.py` |
| A29 | Vivado strategy defaults | `step_04_synth.py` |
| A30 | vectorless power | `tools/parse_power_report.py` |
| A31 | Brevitas→QONNX tolerance | `tools/equivalence_check.py` |
| A32 | ≥10 reps, mean ± σ | `bench/throughput_sweep.py` |
| A33 | latency frame count | `bench/latency_breakdown.py` |
| A34 | power sampling window | `bench/power_log.py`, `bench/README_bench_setup.md` |
| A35 | T2.5 deferred | `tools/run_video.py` |
| A36 | server queue cap | `streamer/protocol.py` |

### Improvements I1–I10 — where each one lives

| ID | Improvement | Implemented in |
|---|---|---|
| I1 | equivalence gate (no paper counterpart) | `tools/equivalence_check.py` |
| I2 | mean ± σ over ≥10 reps | `bench/throughput_sweep.py` |
| I3 | **measured** Coral power | `bench/coral_bench.py --measured-watts` |
| I4 | **measured** PL-only power | `bench/power_log.py` |
| I5 | batch regime stated per row | `tools/report_tables.py` |
| I6 | quantified field P/R | `tools/run_video.py --labels` |
| I7 | >30 FPS source | `streamer/client.py --synthetic-fps` |
| I8 | Edge TPU op-mapping fraction | `tools/compile_edgetpu.sh` |
| I9 | hard build assertions | `finn_build/custom_steps.py` |
| I10 | Table 6 gate before synthesis | `tools/table6_selfcheck.py`, `tests/test_table6.py` |

---

## Entries

### 2026-10-06 — initial source implementation

- Architecture encoded in `models/arch_spec.py` as the **single source of truth**,
  deliberately stdlib-only so GATE 1 runs before any toolchain exists.
  `tests/test_configs.py` asserts the YAML documentation cannot drift from it.
- **GATE 1 PASSES: 60 checks, 0 failures.** All ten Table 6 weight counts, all ten
  MAC counts, and all six totals reproduce exactly (113,330,048 / 37,380,096 /
  2,847,312 / 350,696 / 17,064 / 333,632).
- Cycle model verified against paper Table 8's own arithmetic: 4,586,895 ÷
  100 MHz = 45.87 ms exactly; 100 MHz ÷ 216,320 = 462.28 FPS exactly;
  416² + 208² = 216,320 exactly.
- `configs/folding.json` **generated** by `tools/folding_search.py` and marked
  `PROVISIONAL`. It converges on the paper's own bottleneck
  (`StreamingMaxPool_L2` @ 216,320) and exercises the §3.6.2 HLS-LUT escape hatch.
  **The LUT and BRAM figures in it are OUR ASSUMPTION and must be calibrated
  against T1.9's real Vivado report, after which the search must be re-run.**
- Resource-model correction during implementation: the first BRAM estimator
  assumed a fixed 512×36 geometry and over-counted narrow deep weight memories by
  ~9× (381 tiles for a 4-bit × 194,688 memory instead of ~43). Replaced with an
  aspect-ratio-aware model; `tests/test_cycle_model.py` pins the fix.
- `configs/fifo_depths.json` is **not** generated: it requires RTL simulation
  (A24), which is not run in this phase. `finn_build/step_03_fifos.py` produces it.

#### Still open

- **D9 PENDING** — board, 5 V bench supply, IR camera, Coral all absent.
- **D10 UNVERIFIED** — whether FINN 0.10 supports Vivado 2022.1. Phase 0 tests it.
- **A5** — the upstream `hyp.scratch-low.yaml` release tag must be recorded the
  first time the file is fetched and diffed.
- **A1** — the VisDrone `car` index must be verified against the real annotation
  files (`--verify-index`) before the first training run.
- **A19** — ONNX opset is pinned to 13 in `configs/quant.yaml`; revisit if the
  brevitas 0.9.0 / FINN 0.10 pair disagrees.

### 2026-10-15 — evaluation fixes before the first training run

- **Single-class confidence.** `utils/postprocess.non_max_suppression` now uses
  objectness alone when `nc == 1`. The class channel receives no gradient
  (`utils/loss.py`), so `obj * cls` scaled every score by an untrained value
  and could drop a perfect detection entirely.
- **Evaluation thresholds separated from deployment thresholds.** `train.py`
  evaluates with `EVAL_CONF_THRES=0.001`, `EVAL_IOU_THRES=0.6`,
  `EVAL_MAX_DET=300` (**OUR ASSUMPTION**: Ultralytics `val.py` defaults). The
  REPOSITORY FACT values (0.10 / 0.10 / 20) stay the defaults for deployment,
  where the 20-box cap is real; applied to mAP they truncated recall.
- **P/R reported at max F1** in `utils/metrics.evaluate`, as Ultralytics
  `val.py` does (unsmoothed here), instead of at the lowest confidence kept.
  `tests/test_postprocess.py` pins all three.
- **Final activation uses a constant 1/127 scale.** `models/quant_common.ActQuantSigned`
  was brevitas `Int8ActPerTensorFloat`, whose scale is learned from runtime
  statistics, so the PS-side dequant `Mul` would have been whatever training
  chose rather than the REPOSITORY FACT 1/127 that `utils/postprocess.py`
  hard-codes. It is now `Int8ActPerTensorFloatMinMaxInit` with
  `scaling_impl_type=CONST`, `narrow_range=True`, `max_val=1.0`; brevitas 0.9.0
  forms a narrow signed scale as `max_val / 127`, giving exactly 1/127.
  Recorded in `configs/quant.yaml` as `act_hardtanh_scaling`.

### 2026-10-16 — dataset converted, A1 verified, A15 flipped to BatchNorm ON

- **Dataset (T1.1).** VisDrone2019-DET train + val converted by
  `tools/visdrone_to_yolo.py --verify-index` into `/mnt/data10tb/visdrone_car`
  (images are symlinks into `/mnt/data10tb/visdrone_raw`; manifest in
  `manifest.json` there). train: 6,471 images, 144,866 car boxes; val: 548
  images, 14,064 car boxes. test-dev not downloaded.
- **A1 VERIFIED.** Category id 4 is `car` in both splits (144,867 / 14,064
  boxes before filtering). The annotations also contain id 11 (`others`,
  1,532 / 32 boxes), which `VISDRONE_CATEGORIES` does not name; it is dropped
  with the other non-car classes.
- **A15 flipped: BatchNorm is now ON** (`configs/quant.yaml`
  `structure.batchnorm: true`). Measured before the first training run: with
  BN off, the activations are small enough at the last conv that the
  constant-1/127 `QuantHardTanh` rounds every output to exactly 0 (all ReLU
  layers are alive in train mode; only conv10 collapses). An overfit probe on
  8 fixed train images, 150 SGD steps at `lr0`, seed 0:

  | | loss @ 0 | loss @ 150 | final output |
  |---|---|---|---|
  | BN off | 1.0104 | 1.0104 (unchanged) | all zeros |
  | BN on  | 1.5213 | 0.0502 | std ≈ 0.49 |

  The BN-free network cannot leave the all-zero state, so a full run would
  early-stop at mAP ≈ 0. The hardware is unaffected: FINN absorbs BN into
  MultiThreshold during streamlining. `train.py --no-batchnorm` remains as the
  ablation. `tools/export_tflite.py` now infers BN from the checkpoint's keys
  instead of hard-coding it off.
- **Environment.** brevitas 0.9.0 imports `pkg_resources`, which setuptools 84
  no longer ships; `.env` pinned to `setuptools==69.5.1`, recorded as
  `setuptools<81` in `requirements/train.requirements.txt`.

#### Still open

- `train.py --fp32` only changes the printed title; it still builds the
  quantised model, so T1.2 has no real FP32 baseline yet.

### 2026-10-16 — `train.py --resume`

- `train.py` writes `last.pt` every epoch (model, optimizer, OneCycleLR state,
  epoch, best mAP, patience counter, history, RNG states; tmp + rename so a
  kill mid-write cannot truncate it) and rewrites `results.json` every epoch.
  `--resume runs/<name>/last.pt` continues from the next epoch and refuses to
  run if `--epochs`/`--batch-size`/`--img` differ from the original, since
  they fix the LR schedule. Starting a fresh run into a directory that already
  holds `last.pt` is refused. `best.pt` keeps its previous format.
- Verified on a 256/64-image subset, 4 epochs: a run killed with SIGINT during
  epoch 2 and resumed tracks the uninterrupted run to within the GPU
  run-to-run noise (box loss 1.170 vs 1.179, 1.062 vs 1.039; two fresh runs
  differ by about as much at epoch 0, so this is not bit-exact).
- Measured at batch 64 / 416: about 4.3 GB of GPU memory per `nvidia-smi`
  (3.3 GB allocated by PyTorch), about 2.6 min of training per epoch.

### 2026-10-16 — first T1.3 run diagnosed; loss replaced by YOLOv5 ComputeLoss

- **Run `runs/pruned_4w4a`** (MSE loss) early-stopped at epoch 274, best
  mAP@0.5 **0.0173** at epoch 244 (paper CONTEXT 0.1768). Re-scoring
  `best.pt` on `val` at lower IoU thresholds shows the network finds cars but
  mislocates them:

  | IoU | P | R | mAP |
  |---|---|---|---|
  | 0.5 | 0.119 | 0.076 | 0.017 |
  | 0.3 | 0.275 | 0.191 | 0.087 |
  | 0.1 | 0.460 | 0.327 | 0.238 |

  Median best IoU of any detection per GT box is 0.077; only 9 % of GT boxes
  have a detection above IoU 0.5. Val cars are 11.9 × 8.6 px median (33 %
  under 8 px wide), so a 0.2-cell centre error already destroys IoU. Cell
  collisions are not the limit: 1.86 GT boxes per occupied cell, 3 anchors.
- **DERIVED FACT.** The REPOSITORY FACT decode is YOLOv5's with `sigmoid(x)`
  replaced by `p = y/2 + 0.5` (`2p - 0.5 = y + 0.5`, `(2p)² = (y + 1)²`). With
  paper §3.1.4's "standard low-scratch hyper-parameters", the author's loss
  was almost certainly YOLOv5 `ComputeLoss` under that substitution, and
  `configs/hyp.scratch-low.yaml`'s gains (box 0.05, obj 1.0) are calibrated
  for it, not for MSE. Paper Table 5's final losses (box 0.153, obj 0.305) are
  on that scale.
- **`utils/loss.py` is now that loss** (A5): 1 − CIoU on the decoded box;
  3-cell assignment (own cell + two nearest neighbours, `anchor_t` 4); BCE on
  `p(y_obj)` against target = detached CIoU; gains scaled as YOLOv5 `train.py`
  does (`box·3/nl` = 0.15, `obj·(416/640)²·3/nl` = 1.27, obj balance 4.0 for
  one layer); total × batch size. Logged `box`/`obj` are gain-scaled per-batch
  means, as YOLOv5 logs them, so they are **not comparable with the MSE
  run's columns**. Pinned by `tests/test_loss.py`, including that the loss
  scores exactly the box `models/detect.Detect` returns.
- **OUR ASSUMPTIONS on top of YOLOv5:**
  - `p` is squeezed into `[1e-3, 1 − 1e-3]` before the log. With the affine
    `p` the BCE gradient is ~1/p (with sigmoid it is bounded by 1); a clamp
    instead would add a zero-gradient region to HardTanh's own (measured:
    `QuantHardTanh` passes 0 gradient beyond ±1).
  - Objectness prior as YOLOv5 `_initialize_biases`: `p = 8/(416/32)²`, put
    into the final BatchNorm shift as `y = 2p − 1` (the conv has no bias);
    skipped when BN is off. On one train batch at init it lowers obj loss
    from 8.05 to 2.62.
  - `train.py` clips the gradient norm at 10, as YOLOv5 v7.0 does; needed
    because the loss is now batch-size scaled (grad norm ≈ 3,300 at init,
    ≈ 1,070 at the old `best.pt`).
- **Not changed:** augmentation (mosaic/scale/translate stay off, A4),
  `OneCycleLR` (its 30 % warmup and momentum cycling differ from YOLOv5's
  3-epoch warmup + linear decay; `warmup_*` in the hyp file remain unused),
  patience 30.
- **Not yet trained.** `runs/pruned_4w4a` holds the MSE run and is kept as
  evidence; the next run needs a new `--name`, not `--resume`.
