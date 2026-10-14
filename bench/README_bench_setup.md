# Bench setup protocol (T2.3, T3.1)

**Everything here is BLOCKED: D9 PENDING.** Verified absent on this host: the
Z-turn Board V2, a 5 V bench supply with current readout, an IR thermal camera,
and a Coral Edge TPU.

## What the paper specifies — `PAPER FACT`

- §4.3 / Fig. 18(a): the **whole board** is powered from a **5 V DC bench supply**
  and the **current drawn** is measured.
- §4.3 / Fig. 18(b): **IR thermal camera** pointed at the SoC.
- §4.3.2: **no thermal dissipation apart from static air radiation** — no heatsink,
  no fan.
- §4.3.2: idle is measured **with the bitstream not yet loaded**.
- §5.1: Coral power is **estimated from the manufacturer's datasheet**, not
  measured, and 1.2 W is 2.4 W halved for half frequency.

## What the paper does not specify — `OUR ASSUMPTION A34`

Instrument models and accuracy, sampling rate, averaging window, ambient
temperature, IR emissivity setting, and whether "PS" in the 1.9 W attribution
includes DDR. All of these go in the results manifest, with the supply's stated
accuracy recorded as an uncertainty.

Fixed in `bench/power_log.py` and recorded: **2 Hz sampling over 30 s windows**.

## Protocol

1. **Ambient** — record room temperature and confirm still air. Photograph the
   setup (the paper's Fig. 18 equivalent).
2. **Idle, no bitstream** — power the board, boot Linux, do **not** load the
   overlay. Average for 30 s. This is the paper's 1.9 W figure, and it is also
   the baseline for improvement **I4**.
3. **Per clock (50 / 100 / 200 MHz)** — load the overlay at that clock, start a
   continuous `throughput_test()` loop, let the die reach thermal steady state,
   then average for 30 s.
4. **PL-only power — improvement I4.** Subtract step 2 from step 3 to obtain a
   **measured** PL figure. The paper *derives* 0.22–0.65 W arithmetically; we
   measure it. Report both.
5. **Thermal** — IR reading of the die at steady state under 200 MHz load.
   Record the emissivity used. The paper reports **53.1 °C** against Vivado's
   **46.3 °C** max-ambient estimate.
6. **FPS/W** — combine with T2.1's throughput, using **whole-package** power, as
   the paper does ("it is not possible to exclude the PS from a real-world
   context", §4.3.2). Report batch 1 and batch 100 separately.
7. **Coral — improvement I3.** Measure the Coral on the *same* 5 V bench, so
   Table 15 becomes symmetric instead of measured-vs-datasheet.

## Why this cannot be worked around

Power is not inferable from throughput. Without the supply **and** the IR camera,
Tables 13 and 14 and every FPS/W number — including the paper's headline
**81.88 FPS/W** — are unreproducible **even once the board arrives**. Record that
as a limitation rather than estimating.
