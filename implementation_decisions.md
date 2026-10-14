# Implementation Decisions D1–D10

Resolution of the ten open decisions raised in `experiment_reconstruction_plan.md`.
Nothing here is chosen silently: each decision records the paper's exact words, what the repository
confirms, what is missing, the choice, the reason, and whether it affects **reproducibility** or only
**implementation detail**.

### Determination classes

| Class | Meaning |
|---|---|
| **PAPER-DETERMINED** | The paper settles it. No assumption needed. |
| **REPOSITORY-DETERMINED** | The paper is silent or wrong; the repository provides hard evidence. |
| **ARITHMETICALLY FORCED** | Neither states it, but only one value is self-consistent. Not a free choice. |
| **OUR REIMPLEMENTATION ASSUMPTION** | Genuinely undeterminable from either source. A judgement we own. |
| **PROJECT MANDATE** | Settled by your instruction to reconstruct rather than re-run. |
| **USER DECISION** | Yours to make; we cannot. |

---

## Concise decision table

| # | Decision | Choice | Class | Impact |
|---|---|---|---|---|
| **D1** | FPGA speed grade | **`xc7z020clg400-1`** | REPOSITORY-DETERMINED | **Reproducibility** |
| **D2** | Table 1 row-12 MaxPool stride | **stride 1, padding 'same'** | ARITHMETICALLY FORCED | **Reproducibility (critical)** |
| **D3** | Reporting split for Table 5 | **`val`**, labelled explicitly | **OUR REIMPLEMENTATION ASSUMPTION** | **Reproducibility** |
| **D4** | Training hyper-parameters | **Ultralytics `hyp.scratch-low.yaml`**, vendored and frozen | **OUR REIMPLEMENTATION ASSUMPTION** | **Reproducibility** |
| **D5** | Coral comparison topology | **pruned Table-1 topology** | **OUR REIMPLEMENTATION ASSUMPTION** | **Reproducibility** |
| **D6** | VisDrone class policy | **`car` only, no merging** | PAPER-DETERMINED | **Reproducibility** |
| **D7** | Source of PE/SIMD folding | **re-derive from the verified cycle model** | PROJECT MANDATE | **Reproducibility** |
| **D8** | T2.5 field footage | **defer**; substitute only with a labelled deviation | **OUR REIMPLEMENTATION ASSUMPTION** | Scope only |
| **D9** | Tier-2/3 hardware procurement | **awaiting your answer** | USER DECISION | Scope only |
| **D10** | FINN / Vivado versions | **FINN 0.10 + Vivado/Vitis 2022.1**, pairing to be verified | PAPER + REPOSITORY | **Reproducibility** |

---

## D1 — FPGA part and speed grade

**1. What the paper says.** §3, verbatim: the Z-turn Board V2 "is equipped, amongst other things,
with an **XC7Z020CLG400-2** SoC". The `-2` speed grade is stated once, in prose, and never used again;
no timing report in the paper names a part.

**2. What the repository confirms.** It **contradicts** the paper. The shipped hardware handoff
(`finn_streamer/fpga/bitfile/visdrone.hwh`, identical by MD5 to `client/test_package/model.hwh`)
carries:
```xml
<SYSTEMINFO ARCH="zynq" BOARD="myir.com:mys-7z020:part0:2.1"
            DEVICE="7z020" PACKAGE="clg400" SPEEDGRADE="-1"/>
```
The design that produced the paper's bitstream was therefore closed on the **`-1`** speed grade.

**3. What is missing.** No explanation of the discrepancy. Possible readings: the board carries a `-2`
device but the MYIR board file declares `-1`; or the prose is wrong. We cannot distinguish without
the physical board (the device marking would settle it).

**4. Recommended choice.** Build for **`xc7z020clg400-1`**.

**5. Why.** It is the only machine-readable evidence, and it is the conservative direction: a design
that closes timing at 200 MHz on `-1` also closes on `-2`, but not the reverse. Choosing `-2` risks
producing a bitstream that cannot be reproduced on a `-1` part, and would make our timing results
optimistic relative to the authors'. If the board arrives and its marking reads `-2`, we rebuild and
report both — the cost is one synthesis run.

**6. Impact. Reproducibility.** Speed grade changes achievable clock, timing slack (Fig. 17), and
potentially the folding that fits. It is not cosmetic.

---

## D2 — MaxPool stride at Table 1 row 12

**1. What the paper says.** Table 1, row 12, verbatim:
```
MaxPooling    13 × 13 × 104    13 × 13 × 104    2×2    2    -
```
i.e. kernel 2×2, **stride 2**, with input and output both 13×13×104.

**2. What the repository confirms.** Nothing directly — no model definition exists upstream. But it
confirms the *consequence*: the accelerator's output is `(1,13,13,18)` (`driver.py`, `model.pkl`),
so the spatial resolution must still be 13×13 after this layer.

**3. What is missing.** The row as printed is impossible: a 2×2/stride-2 pool on a 13×13 map yields
6×6, not 13×13. One of the two columns is wrong and the paper does not say which.

**4. Recommended choice.** **Kernel 2×2, stride 1, padding 'same'** (right/bottom pad), preserving
13×13×104.

**5. Why.** This is **arithmetically forced**, not preferred. Three independent constraints agree:
- the printed input/output shapes are only consistent with stride 1;
- stock YOLOv3-Tiny uses exactly a 2×2/stride-1 pool at this position (the sixth pool), so the paper's
  table is reporting a layer that genuinely exists with stride 1;
- the downstream MAC counts in **Table 6** are only reproducible at 13×13. Layer 13
  (104→208, 3×3) is listed at 32,902,272 MAC; `13·13·3·3·104·208 = 32,902,272` **exactly**. At 6×6 it
  would be 7,008,768. Table 6 therefore independently rules out stride 2.

So "stride 2" is a typo in Table 1, and we can say so with certainty rather than preference.

**6. Impact. Reproducibility — critical.** Getting this wrong changes every subsequent feature-map
size, every MAC count, the resource budget, the folding solution, and the output shape. It is the
single most consequential typo in the paper.

---

## D3 — Which split Table 5 reports

**1. What the paper says.** It says both things, in the same breath.
- Table 5 caption (line 1144): "**Validation results** for the original and final pruned model,
  trained on the VisDrone dataset limited to the single class 'car'."
- §4.1 body (lines 1122–1127): "From the final epoch results, the single-class ('car') precision (P)
  on the **training set** settled at around 48%, while recall (R) remained near 17%."

**2. What the repository confirms.** Nothing — no training or evaluation code exists upstream.

**3. What is missing.** Which split produced P = 48.16 %, R = 17.16 %, mAP@0.5 = 17.68 %. Also which
split produced the baseline column (P 67.06 / R 38.44 / mAP 41.76). VisDrone's `test-dev` labels are
withheld in the original challenge, so in practice the choice is train vs. val.

**4. Recommended choice.** **OUR REIMPLEMENTATION ASSUMPTION:** evaluate and report on the
**`val`** split, and label every table unambiguously (`split=val`). We will additionally compute the
same metrics on `train` so that, if our val numbers diverge sharply from the paper's, we can tell
whether the paper was reporting train-set figures.

**5. Why.** "Validation results" in the table caption is the stronger signal: it is the formal label
on the data, whereas the body text reads like loose prose about the final training epoch's logged
metrics (Ultralytics prints both each epoch, which is a very easy thing to conflate). Reporting val
is also the only defensible choice on its own merits — train-set detection metrics are not evidence
of generalisation. Computing both costs one extra evaluation pass and removes the ambiguity entirely,
which is better than picking a side.

**6. Impact. Reproducibility.** Our accuracy numbers are only comparable to the paper's if the split
matches. Train-set mAP is typically much higher than val, so a mismatch here could look like a
reconstruction failure when it is only a labelling difference.

---

## D4 — Training hyper-parameters

**1. What the paper says.** Table 3 gives five values: epochs **300**, batch size **64**, image size
**416 × 416**, optimizer **SGD**, patience **30**. §3.1.4 (line 565) adds only: "**Standard
low-scratch hyper-parameters** were used in order to start the training process."

**2. What the repository confirms.** Nothing — no `hyp*.yaml`, no `train.py`. The repository does
confirm the lineage that makes the phrase interpretable: §3.1.1 cites a quantised **Ultralytics
YOLOv5** fork (ref [64], Okcu), and the leftover `names = ['face']` in
`client/infer_client_v3.py:23` confirms the code descends from LPYOLO [63], itself an Ultralytics
derivative.

**3. What is missing.** Everything beyond those five values: `lr0`, `lrf`, schedule shape, momentum,
`weight_decay`, warmup epochs/momentum/bias, loss gains (`box`, `obj`, `cls`), `anchor_t`, and the
full augmentation block (`hsv_*`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`,
`fliplr`, `mosaic`, `mixup`, `copy_paste`). Also the random seed, and which Ultralytics release the
"low-scratch" file came from.

**4. Recommended choice.** **OUR REIMPLEMENTATION ASSUMPTION:** adopt Ultralytics'
`data/hyps/hyp.scratch-low.yaml`, **vendored into `configs/hyp.scratch-low.yaml` and frozen**, with
the five Table-3 values overriding anything in it. Record the upstream release tag we copied from.
Fix `seed=0` and log it.

**5. Why.** "low-scratch" is not generic English — it is the literal filename stem of Ultralytics'
low-augmentation from-scratch hyper-parameter preset. Given the paper trains a YOLOv5-harness model
and says "standard", that file is the intended referent with high confidence. Vendoring rather than
importing it matters because the file's contents have changed across YOLOv5 releases; a floating
reference would make our own results irreproducible even to ourselves. Treating it as an assumption
rather than a fact is the honest framing, because the paper never names the file.

**6. Impact. Reproducibility.** These values dominate the accuracy result. Mosaic and scale
augmentation in particular interact strongly with small-object aerial detection, and loss gains
interact with the non-standard decode head (see the spec, §4). Accuracy divergence from Table 5
should be attributed here before anywhere else.

---

## D5 — Which topology the Coral comparison uses

**1. What the paper says.** §3.3 in full: "In order to effectively compare the inference results and
performance later obtained with this work against a Coral TPU device, another training run was
performed, this time **with full FP32 precision**. Then, a tflite model was exported with **int8 PTQ**
precision. This tflite model was then compiled by the edgetpu compiler." Table 15 then compares Coral
latency/FPS/FPS-per-watt against the Zynq result.

**2. What the repository confirms.** Nothing — no TFLite, Coral, or FP32 export code exists upstream.

**3. What is missing.** Whether that FP32 run used the **pruned, single-branch Table-1 topology** or
**stock YOLOv3-Tiny**. The paper never says. This matters a great deal: stock YOLOv3-Tiny has roughly
5× the channel widths and two detection branches, so a Coral running stock YOLOv3-Tiny is doing far
more work per frame than the FPGA, and Table 15's latency comparison would be measuring two different
networks. Also missing: the PyTorch→TFLite conversion route, the PTQ scheme (full-integer vs.
dynamic-range), and the calibration set.

**4. Recommended choice.** **OUR REIMPLEMENTATION ASSUMPTION:** use the **pruned Table-1 topology**,
so both platforms execute the identical network. Report the Edge TPU compiler's op-mapping summary
alongside the result, and state the assumption prominently in any Table-15 equivalent.

**5. Why.** The stated purpose is "to effectively compare" against this work; a comparison is only
meaningful if the workload is held constant. If instead the authors used stock YOLOv3-Tiny, then
their Table 15 flatters the FPGA on work-per-frame while flattering the Coral on accuracy, and the
row is not interpretable either way. Holding the network fixed is the only choice that yields a
defensible number. We will note that this may differ from what the authors did, which is itself a
finding worth reporting.

**6. Impact. Reproducibility.** It determines whether our Table 15 equivalent is comparable to
theirs at all. Flagged, not hidden.

---

## D6 — VisDrone class policy

**1. What the paper says.** §3.1.2, explicitly: the dataset "comprises 10,209 images annotated with
2.6 million bounding boxes across 10 object categories (pedestrian, people, bicycle, car, van, truck,
tricycle, awning-tricycle, bus, motor)" and "only one of the 10 categories was used: **car objects**".
Also: "only the first one (object detection in images)" of the five VisDrone tasks, and that a format
conversion to YOLO **XYWH** notation was performed.

**2. What the repository confirms.** `nc = 1` in `client/infer_client_v3.py:24`, and an 18-channel
head (`3 anchors × (1 class + 5) = 18`) in `model.pkl`'s `oshape_normal = (1,13,13,18)`. A single
class is confirmed end to end.

**3. What is missing.** Only two small things: (a) the numeric category index for `car` in the
annotation files — VisDrone's documented ordering is `0 ignored-regions, 1 pedestrian, 2 people,
3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor`, which we will
**verify against the downloaded files rather than assume**; and (b) the policy for boxes flagged
`score=0` (ignored regions) and for the `truncation` / `occlusion` attributes.

**4. Recommended choice.** **`car` only, no merging** of `van`/`truck`/`bus`. Drop `score=0` boxes.
Keep all truncation/occlusion levels. Verify the category index empirically and record it in the
dataset manifest. The merge policy is PAPER-DETERMINED; the ignored-region and attribute policies are
minor **OUR REIMPLEMENTATION ASSUMPTIONS** (recorded as A6/A7 in the spec).

**5. Why.** "only one of the 10 categories was used: car objects" is unambiguous — `van`, `truck` and
`bus` are separate entries in the same enumeration the paper quotes, so merging them would contradict
the text. `score=0` marks VisDrone's ignored regions, which are not objects; including them as
positives would inject label noise.

**6. Impact. Reproducibility.** Dataset content directly determines every accuracy number. Merging
vehicle classes would raise object density substantially and make Table 5 incomparable.

---

## D7 — Where the PE/SIMD folding comes from

**1. What the paper says.** §3.6.1 gives the *method*: start fully folded (every SIMD and PE = 1),
apply a per-layer JSON via `ApplyConfig`, then "an iterative adjustment procedure was followed, each
time unfolding the slowest layer and attempting to balance the cycle-per-layer metric while
maintaining the network's size within reasonable resource limits"; keep small tail layers fully
folded; iterate "until a bottleneck — i.e. the slowest layer being fully unfolded — was encountered".
§3.6.2 adds the key move: use the HLS **LUT** implementation style for MVUs once DSPs are exhausted.
§3.7.2 then states the resulting values "can be found in the repository [62] and are not included in
this document due to their length".

**2. What the repository confirms.** **It does not contain them.** This is the central finding of
`paper_vs_repository.md` §1. The repository holds a compiled bitstream whose
`StreamingDataflowPartition_1` is a black box — the `.hwh` exposes only 7 top-level modules
(`StreamingDataflowPartition_1`, `idma0`, `odma0`, `axi_interconnect_0`, `smartconnect_0`,
`rst_zynq_ps_200M`, `zynq_ps`) with no per-layer parameters. What it *does* pin are two endpoints:
input SIMD = 3 and output PE = 1, from `ishape_folded=(1,416,416,1,3)` and `oshape_folded=(1,13,13,18,1)`.

**3. What is missing.** All interior PE and SIMD values, plus `ram_style`/`resType`/`mem_mode` per
layer, the tie-break rule when two layers are equally slow, and the quantitative meaning of
"reasonable resource limits".

**4. Recommended choice.** **PROJECT MANDATE:** re-derive them, using the cost model below, and emit
`configs/folding.json` as a *generated artifact* with its derivation logged. Do not reverse-engineer
the bitstream (it is opaque) and do not copy any value.

**5. Why.** This is the reconstruction. Beyond your instruction, the derivation is tractable and
anchored: the paper's own Table 8 is exactly consistent with a closed-form FINN cost model, which we
verified arithmetically before adopting it —

```
cycles(MVU) = (C_out / PE) · (k²·C_in / SIMD) · (H_out · W_out)
latency     = total_cycles / f_clk
throughput  = f_clk / max_node_cycles
```

- `4,586,895 cycles ÷ 100 MHz = 45.87 ms` → Table 8's stated latency, exactly.
- `100 MHz ÷ 216,320 cycles = 462.3 FPS` → Table 8's stated 462.28 FPS, exactly.
- The bottleneck `StreamingMaxPool_hls_0` at 216,320 cycles decomposes as `416² + 208² = 173,056 +
  43,264 = 216,320` — one cycle per input pixel plus one per output pixel.
- Layer 1 at full input unfolding (`SIMD = k²C_in = 27`, as RTL parallel-window mode requires per
  §3.5.4, and `PE = C_out = 8`) gives `1 · 1 · 416² = 173,056 < 216,320` ✓ — and `SIMD = 27` over a
  3-channel stream is precisely what the repository's `ishape_folded` trailing `3` encodes.

So the balance target is **≈216,320 cycles/frame**, and for each MVU we solve for the smallest
`(PE, SIMD)` satisfying the inequality under the 140 BRAM / 220 DSP / 53,200 LUT budget, with
DSP packing from Table 4 and RTL/HLS assignment fixed by Table 6. The paper's *procedure* and its
*stated targets* are both reproducible without copying a single value.

**6. Impact. Reproducibility.** This determines throughput, resource utilisation, and whether the
design fits at all. It is the paper's actual contribution and the hardest part to reproduce.

---

## D8 — Field footage for T2.5 (Fig. 19)

**1. What the paper says.** §4.5: "to assess the performance of the complete system on a realistic
use case, we captured footage from a drone flown over roads and parking lots", and the qualitative
result: "despite having a nominally low recall on the validation set, almost all of the cars were
correctly detected and localized, demonstrating that the system can actually be used on the field."
§4.1 supplies the supporting argument: at a standard flight altitude "the scale of the objects to be
recognized can be assumed to be essentially fixed", which is why a single-scale head suffices.

**2. What the repository confirms.** Nothing — the footage is not published, and the Data Availability
Statement says only that contributions "are included in the article".

**3. What is missing.** The footage itself, the capture parameters (altitude, camera, resolution,
frame rate), the number of frames assessed, and any quantitative criterion. "Almost all" is the only
figure of merit given.

**4. Recommended choice.** **OUR REIMPLEMENTATION ASSUMPTION:** **defer** T2.5 until Tier 2 is
unblocked. If it proceeds, substitute a held-out VisDrone *video* sequence (Task 2/3 data, which is
public and from the same domain) and label the substitution as a deviation. Additionally hand-label a
small frame subset and report real P/R, replacing "almost all" with a number.

**5. Why.** The experiment is unobtainable as specified, so there is no way to reproduce it faithfully;
the honest options are to skip it or to substitute transparently. Deferring costs nothing because it
is blocked on hardware anyway. Adding a quantitative metric is worth doing because the repository's
NMS thresholds (`conf=0.10`, `iou=0.10` — far more permissive than a typical 0.25/0.45) would inflate
apparent recall, which is plausibly part of why the field result looks so much better than Table 5.

**6. Impact. Scope only.** T2.5 is a qualitative demonstration. No quantitative claim in the paper
depends on it.

---

## D9 — Tier-2 / Tier-3 hardware

**1. What the paper says.** §3 specifies the **MYiR "Z-turn Board V2"** (XC7Z020CLG400 SoC, 1 GiB
DDR3, micro-SD, HDMI, Gigabit Ethernet), ~€100–150. §4.3 specifies a **5 V DC bench supply with
current measurement** and an **IR thermal camera**, with no heatsink. §5.1 requires a **Coral Edge
TPU** with the `libedgetpu` community fork, at 250 MHz and 500 MHz.

**2. What the repository confirms.** The board identity, via `BOARD="myir.com:mys-7z020:part0:2.1"`
in `visdrone.hwh`. It confirms nothing about instruments.

**3. What is missing.** Nothing in the specification — this is purely a question of what you can
obtain. Verified absent on this host: no board, no bench supply, no IR camera, no Coral.

**4. Recommended choice.** **USER DECISION — awaiting your answer.** Our default, absent one, is to
execute all 13 Tier-1 experiments and leave Tier 2/3 fully specified but unrun.

**5. Why.** Not ours to decide. For planning: without the board, Tables 8(actual)/9/10/13/14/15/16 and
Figs. 18/19 are unreproducible — including the paper's headline 208 FPS at 2.55 W. Without the bench
supply and IR camera specifically, **every FPS/W figure in the paper is unreproducible even with the
board**, since power cannot be measured. Note that T1.13 (TFLite + Edge TPU compilation) needs **no**
Coral and stays in Tier 1.

**6. Impact. Scope only** — it bounds which experiments run, not how correctly the ones we do run are
implemented.

---

## D10 — FINN and Vivado versions

**1. What the paper says.** §3, verbatim: "the **FINN framework version 0.10** was chosen, as it
appeared to be the most mature framework". §3.9–3.10 name **Vitis HLS** and **Vivado** as the
back-end tools but **never state a version** for either.

**2. What the repository confirms.** The Vivado version, which the paper omits:
`visdrone.hwh` header reads `VIVADOVERSION="2022.1"`, timestamped `Sat Feb 15 16:53:39 2025`.
`model_info.json` additionally pins the PL target: `"max_clock_freq": 200.0`,
`"clock_mode": "aggressive"`.

**3. What is missing.** Whether FINN 0.10 officially supports Vivado/Vitis **2022.1** — FINN pins its
expected tool version per release, and a mismatch typically fails during HLS synthesis or IP
packaging. We cannot check this without consulting FINN's release notes, which we have not done
(no installation, per your instruction).

**4. Recommended choice.** **FINN 0.10 + Vivado/Vitis 2022.1**, with the pairing **verified against
FINN 0.10's documented requirement before any download**. If FINN 0.10 requires a different version,
that conflict gets escalated to you rather than silently resolved, because either choice then
contradicts one of the two sources.

**5. Why.** FINN 0.10 is PAPER-DETERMINED and not negotiable — the whole transformation flow, the RTL
MVU with DSP packing (§3.5.2, which the paper credits to a specific FINN feature), and the RTL SWG
parallel-window mode (§3.5.4) are version-specific. 2022.1 is the only hard evidence for the Vivado
side, and it beats a guess. Flagging the unverified pairing now is better than discovering it
mid-build, which is exactly what the feasibility spike at the end of this document is for.

**6. Impact. Reproducibility.** Tool version changes HLS scheduling, resource estimates, and
place-and-route outcomes. It is also the most likely single cause of an unbuildable flow, which is
why it is the first thing the recommended implementation order tests.

---

## Decisions that were *not* needed

Two items from the original plan turned out to be determinable rather than open, and are recorded here
so they are not re-litigated:

- **RTL vs. HLS specialisation per layer** — initially thought lost with the folding config. It is
  **PAPER-DETERMINED**, decodable from Table 6's MVU instance names and verified against Table 1 by
  weight count: conv 1, 2, 4, 9, 10 → **RTL**; conv 3, 5, 6, 7, 8 → **HLS**.
- **Final activation bit width** — §3.1.1 says "all the remaining weights and activations were
  quantized as INT4", which would imply a 4-bit output. But the repository pins `odt = INT8` and
  `scale.npy = 0.00787402 = 1/127`, and `1/127` is exactly the scale that maps a `HardTanh` range of
  `[-1, 1]` onto `INT8` `[-127, 127]`. The output activation is therefore **8-bit**, which is
  **ARITHMETICALLY FORCED** and also matches the paper's own stated rationale for 8 bits at the end
  ("to produce more detailed outputs at the end", §3.1.1).
