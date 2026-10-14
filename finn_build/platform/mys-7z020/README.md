# MYIR `mys-7z020` board files — MUST BE OBTAINED

**Status: ABSENT. Nothing is downloaded by this repository.**

## What is needed

`REPOSITORY FACT` — the author's shipped hardware handoff (`visdrone.hwh`) declares:

```xml
<SYSTEMINFO ARCH="zynq" BOARD="myir.com:mys-7z020:part0:2.1"
            DEVICE="7z020" PACKAGE="clg400" SPEEDGRADE="-1"/>
```

So `ZynqBuild` needs a Vivado board definition registered as
`myir.com:mys-7z020:part0:2.1`. This is a **vendor** board file, not a stock
Xilinx/Digilent platform, so it does not ship with Vivado.

Expected contents once obtained:

```
board.xml
part0_pins.xml
preset.xml          # PS7 configuration: DDR3, clocks, peripherals
board_files/...     # as packaged by MYIR
```

## Two routes

1. **Obtain from MYIR** — the Z-turn Board V2 / MYS-7Z020 resource package.
   Preferred, because it carries the correct PS7 preset (DDR3 timings, MIO
   assignment) that the reference build used.
2. **Author a minimal custom platform** — if the vendor files are unavailable,
   FINN can target the bare part `xc7z020clg400-1` with a hand-written PS7
   preset. This changes the DDR configuration away from the reference build and
   must be recorded in `DECISIONS.md` as a deviation.

## Decisions that touch this directory

- **D1** — speed grade `-1`, taken from this very `.hwh`, not from the paper's
  `-2` prose. If the physical board's device marking reads `-2`, rebuild and
  report both.
- **D10** — Vivado `2022.1`, also from this `.hwh`. The paper never states a
  Vivado version.
- **D9 — PENDING** — the board itself is absent, so every Tier-2 experiment is
  blocked regardless of these files.

## Installation (when the files exist)

```bash
# NOT RUN in this phase
cp -r <myir_board_files>/mys-7z020 "$FINN_XILINX_PATH/Vivado/2022.1/data/boards/board_files/"
# or point Vivado at a custom repo:
#   set_param board.repoPaths {<this directory>}
```

`finn_build/step_04_synth.py --preflight` checks for these files and refuses to
start a synthesis run without them.
