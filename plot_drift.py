"""
plot_drift.py - the project's key figure: how far each baseline has moved from its calibrated state.

    python plot_drift.py gradual_poison --seed 1 --days 7
    python plot_drift.py pir_stuck_night --seed 0 --days 5 --out figs/quarantine.png

Runs a simulator scenario through the full pipeline with an always-learning "naive" twin beside it, and plots both
baselines' drift over time. The attack window is shaded and sensor quarantines are marked. Needs matplotlib.
The simulator's labels are never shown to the system; they are used here only to shade the attack on the chart.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from iotguard.pipeline import Phase, Pipeline  # noqa: E402
from iotguard.presets import room_setup  # noqa: E402
from iotguard.scenarios import make_source  # noqa: E402
from iotguard.security import Ledger  # noqa: E402
from iotguard.sources import strip_labels  # noqa: E402

DAY = 86400.0

# Light-mode colours (validated for colour-blind separation): blue = protected baseline, orange = naive twin.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
GATED, NAIVE, ATTACK_BAND = "#2a78d6", "#eb6834", "#e34948"


def collect(name: str, seed: int, days: float | None):
    """Run the scenario; return (manifest, t, gated_drift, naive_drift, quarantine events, alert events)."""
    src = make_source(name, seed, days=days)
    manifest = src.manifest()
    cfg, cons = room_setup()
    pipe = Pipeline(cfg, consensus=cons, ledger=Ledger(), naive_twin=True)
    pipe.start_calibration()
    t, g, n, quarantines, alerts = [], [], [], [], []

    def on_outcome(o) -> None:
        for e in o.events:
            if e.kind == "sensor_quarantined":
                quarantines.append((e.t, e.sensor))
            elif e.kind == "alert":
                alerts.append(e.t)
        if o.phase is Phase.OPERATING and o.decision is not None and not math.isnan(o.drift) and not math.isnan(o.naive_drift):
            t.append(o.t_start); g.append(o.drift); n.append(o.naive_drift)

    pipe.run(strip_labels(src), on_outcome=on_outcome)
    return manifest, t, g, n, quarantines, alerts


def draw(manifest, t, g, n, quarantines, alerts, out: Path) -> None:
    if not t:
        raise SystemExit("no operating windows were produced; run longer (the first 2 simulated days are calibration)")
    step = max(1, len(t) // 4000)                                  # thin the line, not the story
    x = [v / DAY for v in t[::step]]
    gy, ny = g[::step], n[::step]
    end_x = t[-1] / DAY

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    # attack windows (shaded) and legitimate changes (marked)
    spans = sorted((a["start_s"] / DAY, min((a["end_s"] / DAY) if a["end_s"] is not None else end_x, end_x))
                   for a in manifest["attacks"])
    merged: list[list[float]] = []                                 # overlapping attacks become ONE band, not stacked shading
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    for s, e in merged:
        ax.axvspan(s, e, color=ATTACK_BAND, alpha=0.08, lw=0)
    for c in manifest["legit_changes"]:
        ax.axvline(c["start_s"] / DAY, color=INK2, lw=1, ls=":")

    ax.plot(x, ny, color=NAIVE, lw=2, label="Always-learning baseline (no defence)")
    ax.plot(x, gy, color=GATED, lw=2, label="Consensus-gated baseline (yours)")

    ymax = max(max(gy), max(ny)) * 1.3 or 1.0                      # headroom for the legend
    ax.set_ylim(0, ymax)
    right_side = (x[0] + (x[-1] - x[0]) * 0.5)                     # put the note on the emptier side of the line
    marks = [(qt, f"{sensor.upper()} quarantined") for qt, sensor in quarantines]
    if alerts and not quarantines:
        marks = [(alerts[0], "first alert")]
    for mt, text in marks:
        ax.axvline(mt / DAY, color=INK, lw=1.2, ls="--")
        left_of_line = mt / DAY > right_side
        ax.text(mt / DAY + (-0.05 if left_of_line else 0.05), ymax * 0.62, text + (" " if left_of_line else ""),
                va="center", ha="right" if left_of_line else "left", fontsize=9, color=INK)

    # direct labels at the right end, nudged apart if the lines finish close together
    ly = [gy[-1], ny[-1]]
    gap = ymax * 0.07
    if abs(ly[0] - ly[1]) < gap:
        mid = (ly[0] + ly[1]) / 2
        ly = [mid - gap / 2, mid + gap / 2] if ly[0] <= ly[1] else [mid + gap / 2, mid - gap / 2]
    ax.text(x[-1], ly[0], f"  gated {gy[-1]:.1f}", color=INK, va="center", fontsize=9)
    ax.text(x[-1], ly[1], f"  naive {ny[-1]:.1f}", color=INK, va="center", fontsize=9)
    ax.set_xlim(x[0], x[-1] + (x[-1] - x[0]) * 0.10)

    ax.set_xlabel("Days since the start of the simulation (calibration took the first 2 days)")
    ax.set_ylabel("Baseline drift from the calibrated baseline")
    atk = ", ".join(sorted({a["channel"].upper() for a in manifest["attacks"]}))
    key = ([f"red band = attack on {atk}"] if atk else ["no attack in this scenario"]) \
        + (["dotted line = legitimate change"] if manifest["legit_changes"] else []) \
        + (["dashed line = sensor quarantined"] if quarantines else [])
    ax.set_title(f"{manifest['title']}  (seed {manifest['seed']})", loc="left", fontsize=13, color=INK, pad=22)
    key_text = "; ".join(key)
    ax.text(0, 1.02, key_text[0].upper() + key_text[1:] + ".",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    leg = ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=SURFACE, framealpha=1.0, fontsize=9, labelcolor=INK)
    leg.set_zorder(10)                                             # opaque, so a marker line never cuts through the text
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenario")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--out", default=None, help="PNG path (default figs/drift_<scenario>_seed<N>.png)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path("figs") / f"drift_{a.scenario}_seed{a.seed}.png"
    manifest, t, g, n, q, al = collect(a.scenario, a.seed, a.days)
    draw(manifest, t, g, n, q, al, out)
    print(f"saved {out}   final drift: gated {g[-1]:.2f}, always-learning {n[-1]:.2f}"
          + "".join(f"   [{s} quarantined at day {tq / DAY:.2f}]" for tq, s in q))
    return 0


if __name__ == "__main__":
    sys.exit(main())
