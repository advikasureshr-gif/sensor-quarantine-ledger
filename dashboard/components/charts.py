"""Plotly figures in the SOC theme. Every figure is built from recorded pipeline output; nothing is synthesised."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from .theme import AMBER, BLUE, BORDER, CYAN, DECISION_COLORS, GREEN, MUTED, PANEL, RED, TEXT

CHANNELS = {
    "temp_mean": ("Temperature (window mean)", "°C"),
    "hum_mean": ("Humidity (window mean)", "%RH"),
    "pir_count": ("PIR motion events per window", "rising edges"),
    "pir_duty": ("PIR duty (fraction of window high)", "0–1"),
    "door_count": ("Door transitions per window (open+close)", "count"),
    "ldr_mean": ("Light / LDR (window mean)", "ADC 0–4095"),
    "ldr_std": ("Light / LDR (window std-dev)", "ADC"),
}


def _layout(fig: go.Figure, title: str = "", height: int = 360, ytitle: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=TEXT)), height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
                      font=dict(color=TEXT, family="Segoe UI, sans-serif"), margin=dict(l=50, r=20, t=44, b=40),
                      legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    fig.update_xaxes(title="simulated time (hours)", gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(title=ytitle, gridcolor=BORDER, zeroline=False)
    return fig


def with_hours(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hours"] = out["t_start"] / 3600.0
    return out


def _spans(fig: go.Figure, manifest: dict, label: bool = True) -> None:
    """Ground-truth bands from the scenario manifest (simulator only) + the end of calibration."""
    if not manifest:
        return
    cal = manifest.get("calibration_s")
    if cal:
        fig.add_vline(x=cal / 3600.0, line=dict(color=BLUE, dash="dot", width=1), annotation_text="calibration ends" if label else None,
                      annotation_font_color=BLUE, annotation_position="top left")
    total = manifest.get("duration_s") or 0
    for a in manifest.get("attacks", []):
        end = a.get("end_s")
        fig.add_vrect(x0=a["start_s"] / 3600.0, x1=(end if end is not None else total) / 3600.0, fillcolor=RED, opacity=0.10, line_width=0,
                      annotation_text=a["tag"] if label else None, annotation_font_color=RED, annotation_position="top left")
    for c in manifest.get("legit_changes", []):
        end = c.get("label_end_s")
        fig.add_vrect(x0=c["start_s"] / 3600.0, x1=(end if end is not None else total) / 3600.0, fillcolor=BLUE, opacity=0.08, line_width=0,
                      annotation_text=f"legit: {c['kind']}" if label else None, annotation_font_color=BLUE, annotation_position="bottom left")


def channel_figure(df: pd.DataFrame, channel: str, manifest: Optional[dict] = None, markers: bool = True, ground_truth: bool = True) -> go.Figure:
    d = with_hours(df)
    title, unit = CHANNELS[channel]
    fig = go.Figure(go.Scattergl(x=d["hours"], y=d[channel], mode="lines", name=channel, line=dict(color=CYAN, width=1.2)))
    if markers:
        for lab, color, sym in (("UNCERTAIN", AMBER, "circle"), ("SUSPICIOUS", RED, "x")):
            m = d[d["decision"] == lab]
            if len(m):
                fig.add_trace(go.Scattergl(x=m["hours"], y=m[channel], mode="markers", name=f"{lab} window", marker=dict(color=color, size=6, symbol=sym)))
    if ground_truth:
        _spans(fig, manifest or {})
    return _layout(fig, title, 340, unit)


def score_figure(df: pd.DataFrame, threshold: Optional[float], manifest: Optional[dict] = None) -> go.Figure:
    d = with_hours(df[df["score"].notna()])
    fig = go.Figure(go.Scattergl(x=d["hours"], y=d["score"], mode="lines", name="Mahalanobis D", line=dict(color=BLUE, width=1)))
    for lab, color in DECISION_COLORS.items():
        m = d[d["decision"] == lab]
        if len(m):
            fig.add_trace(go.Scattergl(x=m["hours"], y=m["score"], mode="markers", name=lab, marker=dict(color=color, size=4)))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=AMBER, dash="dash"), annotation_text=f"engine threshold D = {threshold:g}", annotation_font_color=AMBER)
    _spans(fig, manifest or {})
    return _layout(fig, "Anomaly score (Mahalanobis distance, muted sensors excluded)", 380, "D")


def decision_strip(df: pd.DataFrame, manifest: Optional[dict] = None) -> go.Figure:
    d = with_hours(df[df["decision"].notna()])
    fig = go.Figure()
    for lab, color in DECISION_COLORS.items():
        m = d[d["decision"] == lab]
        fig.add_trace(go.Scattergl(x=m["hours"], y=[lab] * len(m), mode="markers", name=lab, marker=dict(color=color, size=7, symbol="line-ns-open")))
    fig.update_yaxes(categoryorder="array", categoryarray=["SUSPICIOUS", "UNCERTAIN", "TRUSTED"])
    _spans(fig, manifest or {}, label=False)
    return _layout(fig, "Decision per window", 230)


def drift_figure(df: pd.DataFrame, manifest: Optional[dict] = None) -> go.Figure:
    d = with_hours(df[df["drift"].notna()])
    fig = go.Figure(go.Scattergl(x=d["hours"], y=d["drift"], mode="lines", name="baseline drift", line=dict(color=CYAN)))
    _spans(fig, manifest or {}, label=False)
    return _layout(fig, "Baseline drift since calibration (engine's drift_distance)", 300, "drift")


def daily_decisions(df: pd.DataFrame) -> go.Figure:
    d = df[df["decision"].notna()].copy()
    d["day"] = (d["t_start"] // 86400).astype(int)
    t = d.groupby(["day", "decision"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for lab, color in DECISION_COLORS.items():
        if lab in t:
            fig.add_trace(go.Bar(x=t.index, y=t[lab], name=lab, marker_color=color))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="simulated day (0 = start)")
    return _layout(fig, "Windows per day by decision", 300, "windows").update_xaxes(title="simulated day (0 = start)")


def confusion_figure(tp: int, fp: int, fn: int, tn: int) -> go.Figure:
    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted normal", "Predicted SUSPICIOUS"], y=["Actually normal", "Actually attack window"], colorscale=[[0, PANEL], [1, BLUE]],
                               text=[[str(v) for v in r] for r in z], texttemplate="%{text}", showscale=False, textfont=dict(size=18, color=TEXT)))
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL, font=dict(color=TEXT), margin=dict(l=50, r=20, t=30, b=40))
    return fig


def compare_bar(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    d = df[df[column].notna()]
    fig = go.Figure(go.Bar(x=[f"{r.scenario}\nseed {r.seed}\n{r.run_id[-6:]}" for r in d.itertuples()], y=d[column], marker_color=CYAN))
    fig.update_xaxes(title="", gridcolor=BORDER).update_yaxes(title=column, gridcolor=BORDER)
    fig.update_layout(title=title, height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL, font=dict(color=TEXT), margin=dict(l=50, r=20, t=44, b=60))
    return fig
