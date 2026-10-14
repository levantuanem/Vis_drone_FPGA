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
| **A15** | **BatchNorm OFF — highest risk** | `configs/quant.yaml`, `train.py --batchnorm` ablation flag |
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
