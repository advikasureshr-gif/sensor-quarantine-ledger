"""Shared drawing code for a running session (Live COM3, Demo Replay, Security Overview): two figures, status chips, and the
'deal with quarantined sensor' dialog."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from iotguard.security import AuthError

from . import admin, theme
from .runner import EngineRun
from .theme import AMBER, BLUE, BORDER, CYAN, GREEN, MUTED, PANEL, RED, TEXT

LABEL_COLOR = {"TRUSTED": GREEN, "UNCERTAIN": AMBER, "SUSPICIOUS": RED}
STATE_TONE = {"ACTIVE": "green", "QUARANTINED": "red", "PROBATION": "amber"}


@st.cache_resource(show_spinner=False)
def hub() -> dict:
    """The sessions that live as long as the server: at most one live and one replay."""
    return {"live": None, "replay": None}


def current(prefer: str | None = None) -> EngineRun | None:
    h = hub()
    order = [prefer] if prefer else []
    order += ["live", "replay"]
    for k in order:
        r = h.get(k)
        if r is not None and (r.alive or r.rows):
            return r
    return None


def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL, font=dict(color=TEXT, family="Segoe UI, sans-serif"),
                      margin=dict(l=60, r=20, t=40, b=70), legend=dict(orientation="h", x=0, y=-0.22), hovermode="x unified")
    for a in fig.layout.annotations:
        if a.yref == "paper" and a.xref == "paper":
            a.update(x=0.0, xanchor="left", font=dict(size=13, color=TEXT))
    fig.update_xaxes(gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False)
    return fig


def learning_figure(run: EngineRun, window_h: float | None = None, height: int = 520) -> go.Figure:
    """Row 1: anomaly score coloured by label. Row 2: learning rate. Row 3: baseline drift, gated vs always-learning twin."""
    df = run.frame()
    live = run.kind == "live"
    if live and len(df):
        x = (df["t_h"] - run.t0_h) * 60.0
        xl = "minutes since the live stream began"
    else:
        x, xl = (df["t_h"] if len(df) else df.get("t_h")), "device hours"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.42, 0.22, 0.36],
                        subplot_titles=("Is it suspicious? Anomaly score, coloured by the engine's label", "How fast is the baseline learning? (alpha)",
                                        "Learning curve: baseline drift, gated vs always-learning twin"))
    if len(df):
        fig.add_trace(go.Scattergl(x=x, y=df["score"], mode="lines", name="score", line=dict(color=BLUE, width=1), showlegend=False), row=1, col=1)
        for lab, c in LABEL_COLOR.items():
            m = df["label"] == lab
            if m.any():
                fig.add_trace(go.Scattergl(x=x[m], y=df["score"][m], mode="markers", name=lab, marker=dict(color=c, size=5 if lab != "TRUSTED" else 3)), row=1, col=1)
        fig.add_trace(go.Scattergl(x=x, y=df["alpha"], mode="lines", name="alpha", line=dict(color=GREEN, width=1.5, shape="hv"), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scattergl(x=x, y=df["drift"], mode="lines", name="gated baseline (this system)", line=dict(color=CYAN, width=1.8)), row=3, col=1)
        if df["naive_drift"].notna().any():
            fig.add_trace(go.Scattergl(x=x, y=df["naive_drift"], mode="lines", name="always-learning twin (no gate)",
                                       line=dict(color=AMBER, width=1.3, dash="dot")), row=3, col=1)
        if run.threshold is not None:
            fig.add_hline(y=run.threshold, line=dict(color=AMBER, dash="dash"), row=1, col=1)
        scale = (lambda h: (h - run.t0_h) * 60.0) if live else (lambda h: h)
        for m in run.marks:
            if m["kind"] in ("attack_start", "attack_end", "admin_release", "sensor_replaced") and (not live or m["t_h"] >= run.t0_h):
                col = RED if m["kind"].startswith("attack") else CYAN
                for r in (1, 2, 3):
                    fig.add_vline(x=scale(m["t_h"]), line=dict(color=col, dash="dot", width=1), row=r, col=1)
        for e in run.events:
            if e["kind"] == "sensor_quarantined" and (not live or e["t_h"] >= run.t0_h):
                for r in (1, 2, 3):
                    fig.add_vline(x=scale(e["t_h"]), line=dict(color=RED, width=2), row=r, col=1)
        if window_h and len(df):
            xe = float(x.iloc[-1])
            span = window_h * 60.0 if live else window_h
            fig.update_xaxes(range=[max(float(x.iloc[0]), xe - span), xe])
    fig.update_yaxes(title_text="D", row=1, col=1)
    fig.update_yaxes(title_text="alpha", row=2, col=1)
    fig.update_xaxes(title_text=xl, row=3, col=1)
    return _style(fig, height)


def raw_figure(run: EngineRun, height: int = 400) -> go.Figure:
    """What the sensors are saying right now: the last few minutes of raw values."""
    df = run.raw_frame()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.4, 0.3, 0.3],
                        subplot_titles=("Temperature (°C) and humidity (%RH)", "PIR motion and door (1 = open)", "Light (LDR: 0 = light, 4095 = dark)"))
    if len(df):
        s = (df["t_h"] - df["t_h"].iloc[-1]) * 3600.0
        fig.add_trace(go.Scattergl(x=s, y=df["temp"], name="temp °C", line=dict(color=AMBER)), row=1, col=1)
        fig.add_trace(go.Scattergl(x=s, y=df["hum"], name="humidity %", line=dict(color=CYAN)), row=1, col=1)
        fig.add_trace(go.Scattergl(x=s, y=df["pir"], name="PIR (sensor)", line=dict(color=GREEN, shape="hv")), row=2, col=1)
        fig.add_trace(go.Scattergl(x=s, y=df["door"] * 0.9, name="door", line=dict(color=BLUE, shape="hv")), row=2, col=1)
        if "pir_used" in df and (df["pir_used"] != df["pir"]).any():
            fig.add_trace(go.Scattergl(x=s, y=df["pir_used"] + 0.04, name="PIR fed to the engine (attack)", line=dict(color=RED, shape="hv", dash="dot")), row=2, col=1)
        fig.add_trace(go.Scattergl(x=s, y=df["ldr"], name="LDR", line=dict(color=MUTED, shape="hv")), row=3, col=1)
    fig.update_xaxes(title_text="seconds ago", row=3, col=1)
    return _style(fig, height)


def status_chips(run: EngineRun) -> str:
    return " ".join(f"<b style='color:{MUTED}'>{s.upper()}</b> " + theme.pill(v, STATE_TONE.get(v, "cyan")) for s, v in run.sensor_states().items())


def latest_label(run: EngineRun):
    df = run.frame()
    return None if df.empty else df.iloc[-1]


def kpis(run: EngineRun) -> None:
    last = latest_label(run)
    c = st.columns(5)
    with c[0]:
        theme.kpi("Status", run.status.replace("_", " "), run.note[:48], "amber" if run.status == "waiting_admin" else "cyan")
    if last is None:
        lab, tone, mode = "—", "muted", "—"
    else:
        lab = last["label"]
        tone = {"TRUSTED": "green", "UNCERTAIN": "amber", "SUSPICIOUS": "red"}[lab]
        mode = {"full": "full rate", "slow": "slow", "freeze": "FROZEN"}[last["mode"]]
    with c[1]:
        theme.kpi("Latest window", lab, "30 s window", tone)
    with c[2]:
        theme.kpi("Baseline learning", mode, f"alpha {last['alpha']:.4f}" if last is not None else "", tone)
    with c[3]:
        theme.kpi("Score D", f"{last['score']:.2f}" if last is not None else "—", f"threshold {run.threshold:g}" if run.threshold else "", "blue")
    with c[4]:
        q = run.quarantined()
        theme.kpi("Quarantined", ", ".join(s.upper() for s in q) or "none", "needs an admin" if q else "all sensors trusted", "red" if q else "green")
    st.markdown("**Sensors:** " + status_chips(run), unsafe_allow_html=True)


@st.dialog("Deal with quarantined sensor")
def quarantine_dialog(run: EngineRun) -> None:
    q = run.quarantined()
    if not q:
        st.success("No sensor is quarantined any more.")
        return
    if not admin.is_admin():
        st.markdown("**Step 1 of 2 — authenticate.** Only the admin can put a quarantined sensor back to work.")
        admin.login_form("dlg_login", hint=True, in_dialog=True)
        return
    st.markdown("**Step 2 of 2 — decide.**")
    sensor = st.selectbox("Quarantined sensor", [s.upper() for s in q]).lower()
    choice = st.radio("What did you do?", ["Unquarantine: I inspected it and it is fine", "Replace: I swapped in a new sensor"], label_visibility="collapsed")
    st.caption("Either way the sensor does not return to full trust at once: it goes on probation and must show 120 clean windows (about an hour) "
               "before it influences scoring and learning again. The decision and your authentication are written to the ledger. "
               "A replacement keeps the room's learned baseline (it is not relearned from scratch).")
    note = st.text_input("Note for the ledger (optional)", "")
    if st.button("Confirm", type="primary"):
        try:
            msg = run.admin_action(sensor, "replace" if choice.startswith("Replace") else "unquarantine", admin.secret() or "", note)
            st.session_state["_flash"] = f"{sensor.upper()}: {msg}"
            st.rerun()
        except AuthError:
            admin.sign_out()
            st.error("Authentication failed. Sign in again.")
        except Exception as e:
            st.error(str(e))


def quarantine_banner(run: EngineRun | None, key: str) -> None:
    """The red bar + 'Deal with quarantined sensor' button. Draws nothing when no sensor is quarantined."""
    flash = st.session_state.pop("_flash", None)
    if flash:
        st.success(flash)
    if run is None or not run.quarantined():
        return
    names = ", ".join(s.upper() for s in run.quarantined())
    a, b = st.columns([3, 1])
    a.error(f"**{names} is quarantined.** Its readings are ignored by scoring and learning; the other sensors keep working. Only the admin can bring it back.")
    if b.button("Deal with quarantined sensor", key=f"deal_{key}", type="primary", use_container_width=True):
        quarantine_dialog(run)
