# Reconstruction: YOLOv3-Tiny on a Zynq-7020 via FINN

Independent reimplementation of Calì, Falaschetti & Biagetti, *"Optimized
Implementation of YOLOv3-Tiny for Real-Time Image and Video Recognition on
FPGA"*, **Electronics 2025, 14, 3993**.

**This is a reconstruction, not a re-run.** The paper is the primary
specification. The author repository (`sn0wst0rm/FINN-VisDrone-YOLO`) is consulted
only for implementation details the paper omits. No upstream code is vendored or
executed.

**We do not optimise toward the paper's reported numbers.** Every paper value in
this repo is labelled CONTEXT and exists so that divergence is *interpretable*.

---

## Status

| | |
|---|---|
| Specification | **frozen** — `final_implementation_spec.md` |
| Source | **written** — Phases 1–4 complete |
| GATE 1 (Table 6 arithmetic) | ✅ **PASSES — 60 checks, 0 failures** |
| Cycle model vs paper Table 8 | ✅ **verified exactly** |
| Folding derivation | ✅ generated, **marked PROVISIONAL** |
| Unit tests | ✅ **104 passing** |
| Training | ⬜ not run (dataset not downloaded) |
| FINN / Vivado / RTLSim / synthesis | ⬜ not run (toolchains not installed) |
| Tier 2 / Tier 3 hardware | ⛔ **D9 PENDING** |

## What runs right now, with nothing installed

```bash
python3 tools/table6_selfcheck.py       # GATE 1: the architecture gate
python3 tools/cycle_model.py --verify   # cost model vs the paper's own Table 8
python3 tools/folding_search.py         # derive configs/folding.json
python3 tools/report_tables.py          # results/TABLES.md
python3 -m pytest tests/ -q             # 104 tests, ~0.4 s
```

All four are stdlib-only (plus numpy for the post-processing tests). No dataset,
GPU, FINN, Vivado, board or network required.

---

## Provenance discipline

Every fact in this codebase carries its source. This is the point of the project.

| Tag | Meaning |
|---|---|
| `PAPER FACT` | stated in the paper; section/table cited inline |
| `REPOSITORY FACT` | recovered from the author repo; file:line cited inline |
| `DERIVED FACT` | only one value is self-consistent; the constraint is shown |
| `OUR ASSUMPTION` | undeterminable from either source; carries an A-id |

`models/arch_spec.py` is the **single source of truth** for the architecture and is
deliberately stdlib-only, so the arithmetic gate runs before any toolchain exists.
`tests/test_configs.py` asserts the YAML documentation cannot drift from it.

### The three findings that shape this reconstruction

1. **The repository does not contain what the paper says it does.** Paper §3 claims
   it holds "the complete code base … and the Jupyter notebooks used for training";
   §3.7.2 claims it holds the folding configuration and all FIFO depths. It holds
   neither — only the `finn_streamer` deployment driver (3 commits, 47 files, 20 of
   them `.pyc`). The two artifacts the paper explicitly delegates to it are the two
   things it lacks, so both are **re-derived** here (D7, A24).
2. **Table 6 reconstructs exactly from Table 1 — all ten layers.** This is the
   cheapest and strongest correctness gate available, and it settles three
   ambiguities in the paper by arithmetic rather than preference.
3. **The repository supplies five facts the paper omits**, all load-bearing:
   the anchors, a non-standard decode head, the 1/127 dequant scale, the I/O
   contract, and `throughput_test()`'s exact semantics.

---

## GATE 1 — the architecture gate

`python3 tools/table6_selfcheck.py` confirms, from Table 1 alone:

| Check | Result |
|---|---|
| 10/10 weight counts vs paper Table 6 | ✅ |
| 10/10 MAC counts vs paper Table 6 | ✅ |
| MAC @ 4b–4b = **113,330,048** | ✅ exact |
| MAC @ 8b–8b = **37,380,096** | ✅ exact |
| MAC @ 4b–8b = **2,847,312** | ✅ exact |
| total weights = **350,696** | ✅ exact |
| weights @ 8b = **17,064** | ✅ exact |
| weights @ 4b = **333,632** | ✅ exact |
| layer 15 MAC = **8,858,304** | ✅ exact |

Folding-independent and tool-independent. **If this fails, the architecture is
wrong and nothing downstream is worth running.**

### Three paper ambiguities it settles

- **Layer 12 MaxPool is stride 1, not 2** (`D2`). Table 1 prints `2×2 / stride 2`
  with 13×13 in *and* out — impossible. Stride 2 would make layer 13 cost
  7,008,768 MAC; Table 6 says 32,902,272, which is exactly `13²·9·104·208`.
  **Arithmetically forced.**
- **Layer 16 is 3×3**, not stock YOLOv3-Tiny's 1×1 — Tables 1 and 6 agree
  (`16,848 = 3·3·104·18`).
- **Layer 16's output activation is INT8**, not INT4. §3.1.1 says "all the
  remaining … INT4", but `odt = INT8` with a dequant scale of `1/127` is exactly
  the HardTanh `[-1,1] → [-127,127]` mapping; 4-bit would need `1/7`.

---

## Folding — derived, never copied

The cost model was adopted only after it reproduced the paper's own Table 8
arithmetic exactly:

```
MVU cycles = (C_out / PE) · (k²·C_in / SIMD) · (H_out · W_out)
latency    = total_cycles / f_clk
throughput = f_clk / max_node_cycles

4,586,895 ÷ 100 MHz   = 45.87 ms   == Table 8   ✅
100 MHz ÷ 216,320     = 462.28 FPS == Table 8   ✅
416² + 208² = 173,056 + 43,264 = 216,320        ✅  (StreamingMaxPool_hls_0)
```

`tools/folding_search.py` then implements paper §3.6.1–3.6.2 verbatim — start
fully folded, unfold the slowest layer, keep tail layers folded, switch HLS MVUs
to the **LUT** style once DSPs are exhausted — and converges on the paper's own
bottleneck. Output: `configs/folding.json`, marked **PROVISIONAL**.

> ⚠ The DSP counts follow paper Table 4 exactly and are trustworthy. The **LUT and
> BRAM figures are `OUR ASSUMPTION`** and must be calibrated against T1.9's real
> Vivado report (`tools/parse_vivado_reports.py` emits the factors), after which
> the search must be re-run and the design re-synthesised.

---

## Layout

```
models/arch_spec.py          single source of truth (stdlib-only)
models/quant_common.py       Brevitas quantised blocks        [needs brevitas]
models/quant_yolov3_tiny.py  Table 1 network                  [needs torch]
models/detect.py             REPOSITORY FACT decode head      [needs torch]
utils/                       pre/post-processing, boxes, metrics  (numpy)
utils/loss.py                loss matching the non-standard decode  [needs torch]
tools/table6_selfcheck.py    GATE 1                           [stdlib]
tools/cycle_model.py         analytical cost model            [stdlib]
tools/folding_search.py      DECISION D7 derivation           [stdlib]
tools/visdrone_to_yolo.py    dataset conversion (A1 verified) [stdlib]
tools/equivalence_check.py   GATE 2 — T1.12                   [needs FINN]
finn_build/step_01..05       FINN front-end → folding → FIFO → synth → rtlsim
streamer/                    protocol, .fnn packaging, server, client
bench/                       throughput, latency, power, Coral
tests/                       104 tests, no heavy dependencies
configs/                     quant, hyp, specialize (paper), folding (generated)
```

Modules needing an absent dependency raise a **clear, documented ImportError**
rather than failing obscurely, and every CLI has an offline mode
(`--describe`, `--plan`, `--show-reference`, `--preflight`, `--self-test`,
`--dry-run`) so the whole pipeline is inspectable with nothing installed.

---

## Implementation order and gates

```
GATE 0  Phase 0 spike       install FINN 0.10 + Vivado 2022.1, push a FINN
        (~1 day)            example to a 7z020 bitstream + rtlsim.
                            Tests the 15 GB RAM ceiling and the UNVERIFIED
                            FINN 0.10 ⇄ Vivado 2022.1 pairing (D10).
GATE 1  Table 6 check       ✅ ALREADY PASSING — hours, free, no deps
        T1.1 → T1.3 → T1.4  dataset, QAT training, QONNX export
        T1.5 → T1.6 → T1.7  FINN front-end, folding, FIFO sizing
        T1.9 → T1.11        synthesis, power estimation
        T1.10               RTLSim throughput
GATE 2  T1.12 equivalence   Tier 2 must NOT begin until this passes
        T2.1 … T2.5         ⛔ D9 PENDING
        T3.1, T3.2          ⛔ D9 PENDING (Coral)
```

`T1.3` is deliberately sequenced before `T1.2`: the hardware flow depends only on
the pruned model, so Phases 3–4 can start while the FP32 baseline still trains.

---

## Blockers

**D9 is PENDING.** Verified absent on this host: Z-turn Board V2, 5 V bench supply
with current readout, IR thermal camera, Coral Edge TPU.

Consequence, stated plainly: the paper's headline result — **208.69 FPS at 2.55 W
= 81.88 FPS/W** — needs T2.1 **and** T2.3, i.e. the board **and both instruments**.
On present evidence **it is not reproducible here.** 13 of 20 experiments proceed
regardless.

## Reference documents

| File | Contents |
|---|---|
| `final_implementation_spec.md` | the 20-section technical specification |
| `implementation_decisions.md` | D1–D10, six fields each |
| `paper_vs_repository.md` | evidence base and discrepancy log |
| `experiment_reconstruction_plan.md` | all 20 experiments, 14 fields each |
| `required_implementation.md` | file-by-file build list |
| `DECISIONS.md` | running log: where every assumption lives in code |
