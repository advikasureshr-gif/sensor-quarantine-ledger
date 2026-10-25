"""Security Overview - the state of the running session (live board or demo replay): labels, learning, quarantined sensors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from components import admin, session_view as sv, theme

theme.page_header("Security Overview", "IoTGuard — behavioural anomaly detection with consensus-gated learning")
admin.sidebar_status()
h = sv.hub()
cands = {"Live COM3": h["live"], "Demo Replay": h["replay"]}
cands = {k: v for k, v in cands.items() if v is not None and (v.alive or v.rows)}
if not cands:
    theme.empty_state("Nothing is running", "Start the Live page (real sensors) or the Demo Replay (recording with a software attack), then come back here.")
    a, b, c = st.columns(3)
    a.page_link("pages/7_Live_COM3.py", label="Live sensors (COM3) →", icon="📡")
    b.page_link("pages/8_Demo_Replay.py", label="Demo replay →", icon="🎬")
    c.page_link("pages/2_Attack_Simulator.py", label="Attack simulator →", icon="🎯")
    st.stop()
name = st.sidebar.radio("Session", list(cands)) if len(cands) > 1 else next(iter(cands))
run = cands[name]


@st.fragment(run_every=2.0)
def body():
    r = run
    if r.status == "error":
        st.error(r.error)
        return
    if r.status == "calibrating":
        st.info(r.note)
        return
    sv.kpis(r)
    sv.quarantine_banner(r, "overview")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(sv.learning_figure(r, window_h=2.0 if r.kind == "live" else None, height=460), use_container_width=True)
    with right:
        st.markdown("#### Recent security events")
        if r.events:
            ev = pd.DataFrame(r.events)
            ev = ev[ev["kind"] != "baseline_checkpoint"].sort_values("t_h", ascending=False).head(12)
            st.dataframe(ev, hide_index=True, use_container_width=True, height=400)
        else:
            st.caption("No security events so far.")


body()
