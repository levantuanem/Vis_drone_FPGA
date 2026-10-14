# Environment and dependencies

**Nothing in this directory has been installed.** Every file documents what is
needed, in the order it is needed. Install order matters: the Phase 0 feasibility
spike comes before any GPU time.

## What is present on this host (verified, not assumed)

| Resource | Observed | Verdict |
|---|---|---|
| GPU | 1× NVIDIA TITAN RTX, 24 GB, driver 535.247.01 | ✅ exceeds the paper's P100 16 GB |
| **RAM** | **15 GB total, ~10 GB free** | ⚠ **the binding constraint** |
| CPU | 6 cores | ⚠ serialised HLS; hours per build |
| Disk `/` | 468 GB, **74 GB free (84 % used)** | ⚠ too tight for Vivado |
| Disk `/mnt/data10tb` | 9.1 TB, **2.9 TB free** | ✅ put everything large here |
| Docker | 28.1.1 | ✅ FINN runs containerised |
| Python (system) | 3.8.10 + numpy 1.24.4, PyYAML 5.3.1, pytest 4.6.9, matplotlib | ✅ enough for the stdlib-only gates |
| Xilinx tools | **none** | install 2022.1 to `/mnt/data10tb` |
| VisDrone dataset | **absent** | T1.1 blocked |
| brevitas / FINN / qonnx | **absent in every conda env** | fresh env needed |
| Z-turn Board V2 | **absent** | all Tier 2 blocked — **D9 PENDING** |
| 5 V bench supply, IR camera | **absent** | T2.3 blocked — **D9 PENDING** |
| Coral Edge TPU | **absent** | T3.1 blocked — **D9 PENDING** |

## Install order

| # | What | Size | For | File |
|---|---|---|---|---|
| 0 | **Nothing — run the stdlib gates first** | — | GATE 1 | see root `README.md` |
| 1 | FINN 0.10 + Docker image | ~10 GB | T1.5–T1.10 | `finn.env.sh.example` |
| 2 | Vivado + Vitis HLS **2022.1** | ~40–60 GB targeted | T1.9, T1.11 | `finn.env.sh.example` |
| 3 | MYIR `mys-7z020` board files | small | T1.9 | `../finn_build/platform/mys-7z020/README.md` |
| 4 | Training env (paper Table 2) | ~5 GB | T1.2–T1.4, T1.12 | `train.requirements.txt` |
| 5 | VisDrone2019-DET train/val/test-dev | ~2–3 GB | T1.1 | → `/mnt/data10tb` |
| 6 | TFLite + Edge TPU compiler | ~2 GB | T1.13 | `coral.requirements.txt` |
| 7 | PYNQ image for the board | ~7 GB | Tier 2 | `board.requirements.txt` |

Steps 1–3 before step 4: the **Phase 0 spike** must confirm the 15 GB RAM ceiling
and the FINN 0.10 ⇄ Vivado 2022.1 pairing (**D10**, still `UNKNOWN`) before any
GPU-days are spent on training.

## Licensing note

XC7Z020 is in the free Vivado ML Standard / WebPACK device tier, so no paid
licence should be required. **Confirm this for 2022.1 specifically** before
committing to the download.

## What runs with no installation at all

```bash
python3 tools/table6_selfcheck.py        # GATE 1 — 60 checks
python3 tools/cycle_model.py --verify    # model vs paper Table 8
python3 tools/folding_search.py          # derives configs/folding.json
python3 tools/report_tables.py           # results/TABLES.md
python3 -m pytest tests/ -q              # 104 tests
```
