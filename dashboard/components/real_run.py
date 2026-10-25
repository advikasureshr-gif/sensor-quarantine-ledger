"""Loader and figures for a run exported by `demo_pipeline.py` (exports/<stamp>/). Reads files only; never runs the engine."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import AMBER, BLUE, BORDER, CYAN, GREEN, MUTED, PANEL, RED, TEXT

ROOT = Path(__file__).resolve().parents[1]                       # the dashboard folder
REQUIRED = ("meta.json", "windows.csv", "events.json", "ledger.jsonl", "chain.json", "recording_summary.json", "recording_series.csv")
MODE_COLOR = {"full": GREEN, "slow": AMBER, "freeze": RED}
LABEL_COLOR = {"TRUSTED": GREEN, "UNCERTAIN": AMBER, "SUSPICIOUS": RED}
EVENT_COLOR = {"alert": AMBER, "sensor_quarantined": RED, "admin_release": CYAN, "sensor_recovered": GREEN, "sensor_probation": CYAN}


def search_dirs() -> list[Path]:
    dirs = [ROOT / "outputs" / "real_runs", ROOT.parent / "exports"]
    extra = os.environ.get("IOTGUARD_EXPORTS")
    if extra:
        dirs.append(Path(extra))
    return dirs


def is_run(p: Path) -> bool:
    return p.is_dir() and all((p / f).exists() for f in REQUIRED)


def find_runs() -> list[Path]:
    found: list[Path] = []
    for d in search_dirs():
        if d.is_dir():
            found += [p for p in d.iterdir() if is_run(p)]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


@dataclass
class RealRun:
    path: Path
    meta: dict
    windows: pd.DataFrame
    events: list
    marks: list
    ledger: list
    chain: dict
    rec: dict
    series: pd.DataFrame
    notes: list = field(default_factory=list)

    def mark_h(self, kind: str):
        for m in self.marks:
            if m["kind"] == kind:
                return m["t_h"]
        return None

    @property
    def attack_span(self):
        a, b = self.mark_h("attack_start"), self.mark_h("attack_end")
        return (a, b) if a is not None and b is not None else None


def load_run(path) -> RealRun:
    p = Path(path)
    missing = [f for f in REQUIRED if not (p / f).exists()]
    if missing:
        raise FileNotFoundError(f"{p} is not a saved run (missing: {', '.join(missing)})")
    ev = json.loads((p / "events.json").read_text(encoding="utf-8"))
    ledger = [json.loads(line) for line in (p / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    w = pd.read_csv(p / "windows.csv")
    for c in ("muted", "contradictions", "reason", "occupancy"):
        if c in w:
            w[c] = w[c].fillna("")
    return RealRun(p, json.loads((p / "meta.json").read_text(encoding="utf-8")), w, ev.get("events", []), ev.get("marks", []), ledger,
                   json.loads((p / "chain.json").read_text(encoding="utf-8")),
                   json.loads((p / "recording_summary.json").read_text(encoding="utf-8")), pd.read_csv(p / "recording_series.csv"))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL, font=dict(color=TEXT, family="Segoe UI, sans-serif"),
                      margin=dict(l=60, r=20, t=50, b=40), legend=dict(orientation="v", x=1.01, y=1.0, xanchor="left", bgcolor="rgba(0,0,0,0)"),
                      hovermode="x unified")
    for a in fig.layout.annotations:                       # subplot titles: left-aligned so labels at the top never collide with them
        if a.text and a.yref == "paper" and a.xref == "paper":
            a.update(x=0.0, xanchor="left", font=dict(size=13, color=TEXT))
    fig.update_xaxes(gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False)
    return fig


def _bands(fig: go.Figure, run: RealRun, rows: int, label: bool = True) -> None:
    """The software attack window on every subplot; calibration end and milestone events as vertical lines."""
    span = run.attack_span
    for r in range(1, rows + 1):
        if span:
            fig.add_vrect(x0=span[0], x1=span[1], fillcolor=RED, opacity=0.10, line_width=0, row=r, col=1)
        cal = run.meta.get("calibration_end_h")
        if cal is not None:
            fig.add_vline(x=cal, line=dict(color=BLUE, dash="dot", width=1), row=r, col=1)
    if label and span:
        fig.add_annotation(x=span[0], y=1.0, yref="paper", text="attack injected (software only)", showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(color=RED, size=11), bgcolor=PANEL)
    if label and run.meta.get("calibration_end_h") is not None:
        fig.add_annotation(x=run.meta["calibration_end_h"], y=1.0, yref="paper", text="calibration ends", showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(color=BLUE, size=11), bgcolor=PANEL)


def timeline_figure(run: RealRun, show_naive: bool = True, focus=None) -> go.Figure:
    w = run.windows
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.34, 0.16, 0.30, 0.20],
                        subplot_titles=("Anomaly score (Mahalanobis D)", "Learning mode", "Baseline drift since calibration",
                                        "Sensors masked (quarantined or on probation)"))
    fig.add_trace(go.Scattergl(x=w["t_h"], y=w["score"], mode="lines", name="score D", line=dict(color=BLUE, width=1)), row=1, col=1)
    for lab, color in LABEL_COLOR.items():
        m = w[w["label"] == lab]
        if len(m) and lab != "TRUSTED":
            fig.add_trace(go.Scattergl(x=m["t_h"], y=m["score"], mode="markers", name=lab, marker=dict(color=color, size=4)), row=1, col=1)
    if run.meta.get("threshold") is not None:
        fig.add_hline(y=run.meta["threshold"], line=dict(color=AMBER, dash="dash"), row=1, col=1,
                      annotation_text=f"threshold D = {run.meta['threshold']:g}", annotation_font_color=AMBER)
    for mode, color in MODE_COLOR.items():
        m = w[w["mode"] == mode]
        fig.add_trace(go.Scattergl(x=m["t_h"], y=[mode] * len(m), mode="markers", name=f"learning: {mode}",
                                   marker=dict(color=color, size=7, symbol="line-ns-open")), row=2, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=["freeze", "slow", "full"], row=2, col=1)
    fig.add_trace(go.Scattergl(x=w["t_h"], y=w["drift"], mode="lines", name="gated baseline (this system)", line=dict(color=CYAN, width=1.6)), row=3, col=1)
    if show_naive and w["naive_drift"].notna().any():
        fig.add_trace(go.Scattergl(x=w["t_h"], y=w["naive_drift"], mode="lines", name="always-learning twin (no gate)",
                                   line=dict(color=AMBER, width=1.2, dash="dot")), row=3, col=1)
    n_muted = w["muted"].map(lambda s: len([x for x in str(s).split(",") if x]))
    fig.add_trace(go.Scattergl(x=w["t_h"], y=n_muted, mode="lines", name="sensors masked", line=dict(color=RED, width=1.4, shape="hv"),
                               fill="tozeroy", fillcolor="rgba(239,68,68,0.18)", text=w["muted"].replace("", "none"),
                               hovertemplate="%{text}<extra>masked</extra>"), row=4, col=1)
    fig.update_yaxes(title_text="D", row=1, col=1)
    fig.update_yaxes(title_text="drift", row=3, col=1)
    fig.update_yaxes(title_text="count", dtick=1, rangemode="tozero", row=4, col=1)
    fig.update_xaxes(title_text="recording time (hours since the first reading)", row=4, col=1)
    _bands(fig, run, 4)
    ytop = float(w["score"].max()) * 1.12 if len(w) else 6.0
    groups: dict[str, list] = {}
    for e in run.events:
        if e["kind"] in ("alert", "sensor_quarantined", "sensor_recovered") and e.get("t_h") is not None:
            groups.setdefault(e["kind"], []).append(e)
    if run.mark_h("admin_release") is not None:
        groups["admin_release"] = [{"kind": "admin_release", "t_h": run.mark_h("admin_release"), "detail": "wrong secret refused, then the correct secret"}]
    for kind, items in groups.items():
        fig.add_trace(go.Scatter(x=[i["t_h"] for i in items], y=[ytop] * len(items), mode="markers", name=kind.replace("sensor_", "").replace("_", " "),
                                 marker=dict(color=EVENT_COLOR.get(kind, CYAN), size=12, symbol="diamond", line=dict(color=TEXT, width=1)),
                                 text=[i.get("detail", "") for i in items], hovertemplate="%{text}<extra>" + kind.replace("_", " ") + " at %{x:.2f} h</extra>"),
                      row=1, col=1)
    fig.update_yaxes(range=[0, ytop * 1.1], row=1, col=1)
    if focus:
        fig.update_xaxes(range=list(focus))
    return _style(fig, 820)


def mode_table(run: RealRun) -> pd.DataFrame:
    t = run.windows.groupby(["period", "mode"]).size().unstack(fill_value=0)
    for m in ("full", "slow", "freeze"):
        if m not in t:
            t[m] = 0
    t = t[["full", "slow", "freeze"]]
    order = [p for p in ("before the attack", "during the attack", "after the attack") if p in t.index]
    t = t.loc[order]
    t["windows"] = t.sum(axis=1)
    return t


def mode_figure(run: RealRun) -> go.Figure:
    t = mode_table(run)
    fig = go.Figure()
    for m, c in MODE_COLOR.items():
        fig.add_trace(go.Bar(x=t.index, y=100.0 * t[m] / t["windows"], name=m, marker_color=c, customdata=t[m],
                             hovertemplate="%{y:.1f}% (%{customdata} windows)<extra>" + m + "</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="% of windows")
    return _style(fig, 300)


def recording_figure(run: RealRun) -> go.Figure:
    s = run.series
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.38, 0.27, 0.35], specs=[[{}], [{}], [{"secondary_y": True}]],
                        subplot_titles=("PIR (share of each minute high) and door", "Light (LDR: 0 = light, 4095 = dark)", "Temperature and humidity"))
    fig.add_trace(go.Scattergl(x=s["t_h"], y=s["pir"], mode="lines", name="PIR (fraction of the minute)", line=dict(color=CYAN, width=1), fill="tozeroy",
                               fillcolor="rgba(34,211,238,0.15)"), row=1, col=1)
    d = s[s["door"] > 0]
    fig.add_trace(go.Scattergl(x=d["t_h"], y=[1.08] * len(d), mode="markers", name="door open", marker=dict(color=AMBER, size=7, symbol="triangle-down")),
                  row=1, col=1)
    fig.add_trace(go.Scattergl(x=s["t_h"], y=s["ldr"], mode="lines", name="LDR (minute mean)", line=dict(color=BLUE, width=1.2, shape="hv")), row=2, col=1)
    fig.add_trace(go.Scattergl(x=s["t_h"], y=s["temp"], mode="lines", name="temperature (°C)", line=dict(color=RED, width=1.2)), row=3, col=1)
    fig.add_trace(go.Scattergl(x=s["t_h"], y=s["hum"], mode="lines", name="humidity (%)", line=dict(color=GREEN, width=1.2)), row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="PIR", range=[0, 1.2], row=1, col=1)
    fig.update_yaxes(title_text="LDR", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="%RH", row=3, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="recording time (hours)", row=3, col=1)
    span = run.attack_span
    if span:
        for r in (1, 2, 3):
            fig.add_vrect(x0=span[0], x1=span[1], fillcolor=RED, opacity=0.08, line_width=0, row=r, col=1)
        fig.add_annotation(x=span[0], y=1.0, yref="paper", text="software attack window (the raw data below is unmodified)", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(color=RED, size=11))
    return _style(fig, 720)


def recording_notes(rec: dict) -> list[tuple[str, str]]:
    """Plain-language checks on the raw recording, in the spirit of `iotguard.record --check` (tone: ok / warn / info)."""
    out = [("info", f"{rec['readings']:,} readings over {rec['hours']:.1f} h" + (f", {rec['bad_lines']} bad lines skipped" if rec["bad_lines"] else ""))]
    out.append(("ok" if rec["gaps_over_5s"] == 0 else "warn",
                "no gaps in the stream" if rec["gaps_over_5s"] == 0 else f"{rec['gaps_over_5s']} gaps over 5 s (longest {rec['max_gap_s']:.0f} s)"))
    out.append(("ok" if rec["dht_null_pct"] < 1 else "warn", f"DHT11: {rec['dht_null']} failed reads ({rec['dht_null_pct']:.2f}%)"))
    longest = rec["pir_longest_high_s"]
    out.append(("ok" if longest < 300 else "warn",
                f"PIR high {rec['pir_high_pct']:.1f}% of the time, {rec['pir_triggers']} triggers, longest {longest:.0f} s"
                + ("" if longest < 300 else " (300 s or more reads as a stuck PIR)")))
    out.append(("info", f"door changed {rec['door_changes']} times, open {rec['door_open_pct']:.1f}% of the time"))
    if rec["ldr_other_pct"] < 1 and rec["ldr_0_pct"] + rec["ldr_4095_pct"] > 99:
        out.append(("info", f"LDR is digital-only: {rec['ldr_0_pct']:.0f}% light (0), {rec['ldr_4095_pct']:.0f}% dark (4095), so the saturation warning is expected"))
    else:
        out.append(("info", f"LDR: {rec['ldr_0_pct']:.0f}% at 0, {rec['ldr_4095_pct']:.0f}% at 4095, {rec['ldr_other_pct']:.1f}% in between"))
    return out
