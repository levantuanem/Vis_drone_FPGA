#!/usr/bin/env bash
# T1.13 (second half) -- compile an INT8 TFLite model for the Edge TPU.
#
# PAPER FACT 3.3: "This tflite model was then compiled by the edgetpu compiler.
# In order to perform the export, a newer, community-maintained, version of the
# libedgetpu module is required" (PAPER FACT ref 68: github.com/feranick/libedgetpu).
#
# OUR ADDITION I8: the compiler log reports which ops map to the Edge TPU and
# which fall back to the CPU. CPU fallback would materially change T3.1, and the
# paper never reports it -- so we capture the split into a sidecar JSON.
#
# DEPENDENCY NOT INSTALLED: edgetpu_compiler, libedgetpu (community fork).
# Requires NO Coral device -- compilation is host-only.
# NOT RUN in this phase.

set -euo pipefail

MODEL="${1:-build/model_int8.tflite}"
OUTDIR="${2:-build}"

if ! command -v edgetpu_compiler >/dev/null 2>&1; then
  cat <<'MSG'
BLOCKED: edgetpu_compiler is not installed (by design in this phase).

To install later (NOT run now):
  # Debian/Ubuntu, Google's coral repo
  echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo apt-key add -
  sudo apt-get update && sudo apt-get install -y edgetpu-compiler

  # Runtime for T3.1 (needs the device): community libedgetpu, PAPER ref 68
  #   https://github.com/feranick/libedgetpu
MSG
  exit 1
fi

mkdir -p "$OUTDIR"
LOG="$OUTDIR/edgetpu_compile.log"
edgetpu_compiler -s -o "$OUTDIR" "$MODEL" | tee "$LOG"

# I8: extract the TPU/CPU op split from the compiler log
python3 - "$LOG" "$OUTDIR" <<'PY'
import json, re, sys
log, outdir = sys.argv[1], sys.argv[2]
text = open(log).read()
tpu = re.search(r"Number of operations that will run on Edge TPU:\s*(\d+)", text)
cpu = re.search(r"Number of operations that will run on CPU:\s*(\d+)", text)
t = int(tpu.group(1)) if tpu else None
c = int(cpu.group(1)) if cpu else None
doc = {
    "_provenance": "I8 -- op-mapping split; the paper never reports this",
    "ops_on_edgetpu": t, "ops_on_cpu": c,
    "fraction_on_edgetpu": (t / float(t + c)) if (t is not None and c) else None,
    "note": "CPU fallback inflates T3.1 latency and would make Table 15 "
            "incomparable; record it with the benchmark result.",
}
path = outdir + "/edgetpu_op_mapping.json"
json.dump(doc, open(path, "w"), indent=2)
print("wrote " + path)
print(json.dumps(doc, indent=2))
PY
