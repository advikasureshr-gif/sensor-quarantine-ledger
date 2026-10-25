"""Demo Replay - the whole pipeline on a recording: calibration, a software PIR attack, detection, quarantine, admin recovery."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components import admin, session_view as sv, theme
from components.runner import EngineRun, find_recordings

theme.page_header("Demo Replay", "A recording replayed through the engine. The PIR attack is injected in software; nothing physical is touched.")
admin.sidebar_status()
h = sv.hub()
run: EngineRun | None = h["replay"]
running = run is not None and run.alive

if not running and (run is None or not run.rows):
    recs = find_recordings()
    if not recs:
        theme.empty_state("No recording found", "Put a recording (.jsonl, e.g. room_merged.jsonl or sim6d.jsonl) in the project's logs folder.")
        st.stop()
    rec = st.selectbox("Recording", recs, format_func=lambda p: p.name)
    c = st.columns(4)
    factor = c[0].number_input("Calibration shortening", 1.0, 8.0, 2.0, 0.5, help="2 = 24 h calibration. The recording must be at least calibration + attack + 3.5 h.")
    ahours = c[1].number_input("Attack length (h)", 3.2, 8.0, 4.5, 0.5, help="Quarantine needs about 3 h of sustained suspicion.")
    default_start = 24.3 if factor == 2 else 48.3
    astart = c[2].number_input("Attack starts at (device hour)", 1.0, 400.0, default_start, 0.1,
                               help="Must be after calibration, in a stretch where nobody was in the room. `python -m iotguard.inject --suggest FILE` finds one.")
    speed = c[3].slider("Speed (simulated min per second)", 30, 600, 180, 30)
    st.caption("The first part (calibration) is fast-forwarded in a few seconds; from an hour before the attack the replay slows down so it can be watched.")
    if st.button("▶ Start replay", type="primary"):
        run = EngineRun("replay", f"Replay of {rec.name}", admin.auth_service(), admin.server_secret(), demo=float(factor))
        h["replay"] = run
        run.start_replay(str(rec), float(astart), float(ahours), float(speed))
        st.rerun()
    if run is not None and run.error:
        st.error(run.error)
    st.stop()

c = st.columns([1, 1, 4])
if c[0].button("■ Stop", disabled=not running):
    run.stop()
    st.rerun()
if c[1].button("↺ New replay", disabled=running):
    h["replay"] = None
    st.rerun()


@st.fragment(run_every=1.5)
def body():
    r = run
    if r.status == "error":
        st.error(r.error)
        return
    if r.status == "calibrating":
        st.info("Calibrating: " + r.note)
        return
    sv.kpis(r)
    sv.quarantine_banner(r, "replay")
    st.plotly_chart(sv.learning_figure(r), use_container_width=True)
    st.caption("Red dotted lines: the software attack starts and ends. Solid red line: the PIR is quarantined. Cyan dotted line: the admin dealt with it. "
               "During the attack the label turns SUSPICIOUS and alpha drops to 0, so the gated baseline stays flat while the always-learning twin is dragged away.")
    if r.events:
        ev = pd.DataFrame(r.events)
        ev = ev[~ev["kind"].isin(["baseline_checkpoint"])]
        st.markdown("**Security events**")
        st.dataframe(ev.sort_values("t_h", ascending=False), hide_index=True, use_container_width=True, height=220)
    if r.status == "finished":
        st.success("Replay finished. The ledger and blockchain for this run are on the Admin page.")


body()
