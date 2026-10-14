"""Plots: Fig. 15 training curves / PR curve, Figs. 8/10/11 folding bars.

matplotlib is an optional dependency (PAPER FACT Table 2 pins 3.5.1); every
function degrades to a no-op with a clear message if it is absent, so importing
this module never breaks a lightweight run.
"""

import json
import os


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:  # pragma: no cover
        return None


def _save(fig, path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def training_curves(history, out_path):
    """Fig. 15(a) equivalent. history: list of per-epoch metric dicts."""
    plt = _plt()
    if plt is None:
        print("matplotlib absent; skipped {}".format(out_path))
        return None
    keys = [k for k in ("box", "obj", "cls", "precision", "recall", "map50")
            if any(k in h for h in history)]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 2.8))
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        ax.plot([h.get("epoch", i) for i, h in enumerate(history)],
                [h.get(k) for h in history], linewidth=1.4)
        ax.set_title(k)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    _save(fig, out_path)
    plt.close(fig)
    return out_path


def pr_curve(recall, precision, out_path, label="car"):
    """Fig. 15(b) equivalent."""
    plt = _plt()
    if plt is None:
        print("matplotlib absent; skipped {}".format(out_path))
        return None
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.plot(recall, precision, linewidth=1.6, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out_path)
    plt.close(fig)
    return out_path


def cycles_per_layer(nodes, out_path, floor=None, title="Cycles per layer"):
    """Fig. 10 equivalent -- the flat cycle profile the folding search targets."""
    plt = _plt()
    if plt is None:
        print("matplotlib absent; skipped {}".format(out_path))
        return None
    names = [n["name"] for n in nodes]
    cycles = [n["cycles"] for n in nodes]
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(names)), 3.6))
    ax.bar(range(len(names)), cycles)
    if floor:
        ax.axhline(floor, linestyle="--", linewidth=1.2,
                   label="balance floor = {:,}".format(floor))
        ax.legend()
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("clock cycles / frame")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, out_path)
    plt.close(fig)
    return out_path


def resource_per_layer(nodes, out_path, title="Resource utilisation per layer"):
    """Figs. 8 / 11 equivalent."""
    plt = _plt()
    if plt is None:
        print("matplotlib absent; skipped {}".format(out_path))
        return None
    names = [n["name"] for n in nodes]
    fig, axes = plt.subplots(3, 1, figsize=(max(6, 0.32 * len(names)), 7),
                             sharex=True)
    for ax, key, lab in zip(axes, ("lut", "dsp", "bram18k"),
                            ("LUT (provisional)", "DSP (exact)",
                             "BRAM18K (provisional)")):
        ax.bar(range(len(names)), [n.get(key, 0) for n in nodes])
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    axes[-1].set_xticks(range(len(names)))
    axes[-1].set_xticklabels(names, rotation=90, fontsize=6)
    axes[0].set_title(title)
    _save(fig, out_path)
    plt.close(fig)
    return out_path


def from_folding_json(path, out_dir="results/figs"):
    """Render Figs. 10 and 11 equivalents straight from configs/folding.json."""
    with open(path) as fh:
        doc = json.load(fh)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools import cycle_model as CM
    res = CM.evaluate(doc["folding"], doc.get("styles", {}))
    floor = doc.get("balance_floor", {}).get("cycles")
    return [
        cycles_per_layer(res["nodes"], os.path.join(out_dir, "fig10_cycles.png"),
                         floor=floor),
        resource_per_layer(res["nodes"], os.path.join(out_dir, "fig11_resources.png")),
    ]
