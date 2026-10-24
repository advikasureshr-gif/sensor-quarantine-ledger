"""Page A - Security Overview (rendered through app.py's navigation)."""
import pandas as pd
import streamlit as st

from components import theme
from components.metrics import operating, sensor_health_table
from components.state import get_manager, pick_run

theme.page_header("Security Overview", "IoTGuard — Behavioural Anomaly Detection & Attack Simulation Platform")
theme.sim_badge()
mgr = get_manager()
active = mgr.active
if active is not None:
    st.info(f"Run {active.run_id} is in progress ({active.progress:.0%}). Open the Attack Simulator page to follow it.")
rec = pick_run()
if rec is None:
    theme.empty_state("No simulation has been run yet", "Open the Attack Simulator, choose a scenario and press Start. Every number here comes from a completed run of the real pipeline.")
    st.page_link("pages/2_Attack_Simulator.py", label="Go to the Attack Simulator →", icon="🎯")
    st.stop()

s, df = rec.summary, rec.windows
if rec.status != "completed":
    st.warning(f"This run is **{rec.status}**; figures are partial. {rec.error or ''}")
dec = s.get("decisions", {})
op = operating(df)
c = st.columns(6)
with c[0]: theme.kpi("Windows processed", f"{s.get('windows_processed', 0):,}", f"{s.get('calibration_windows', 0):,} calibration · {s.get('operating_windows', 0):,} operating")
with c[1]: theme.kpi("Trusted", f"{dec.get('TRUSTED', 0):,}", "operating phase", "green")
with c[2]: theme.kpi("Uncertain", f"{dec.get('UNCERTAIN', 0):,}", "operating phase", "amber")
with c[3]: theme.kpi("Suspicious", f"{dec.get('SUSPICIOUS', 0):,}", "flagged, not necessarily attacks", "red")
with c[4]: theme.kpi("Security events", f"{s.get('security_events', 0):,}", ", ".join(f"{k}:{v}" for k, v in s.get("events_by_kind", {}).items()) or "none", "blue")
with c[5]: theme.kpi("Latest score D", theme.fmt(s.get("latest_score")), f"last decision: {s.get('latest_decision') or '—'}", "cyan")

st.markdown("&nbsp;")
a, b = st.columns([1, 1])
with a:
    st.markdown("#### Pipeline & calibration")
    cal = s.get("calibration", {})
    ledger_ok = s.get("ledger_chain_ok")
    st.markdown(f"Phase: {theme.pill(s.get('phase', '—'), 'green' if s.get('phase') == 'OPERATING' else 'amber')} &nbsp; "
                f"Baseline version: **{s.get('baseline_version')}** &nbsp; Ledger hash-chain: "
                f"{theme.pill('verified' if ledger_ok else 'BROKEN' if ledger_ok is False else 'n/a', 'green' if ledger_ok else 'red')}", unsafe_allow_html=True)
    st.caption(cal.get("summary") or "No calibration result recorded.")
    st.caption(f"Run: {rec.label} · input: {s.get('input', '—')} · wall time {s.get('wall_seconds', '—')} s")
with b:
    st.markdown("#### Quarantine status")
    muted = s.get("muted_sensors", [])
    if muted:
        st.error("Sensors currently quarantined or on probation: " + ", ".join(m.upper() for m in muted))
    else:
        st.success("No sensor is quarantined or on probation at the end of this run.")
st.markdown("#### Sensor health (end of run)")
st.dataframe(sensor_health_table(s, df), hide_index=True, use_container_width=True)
st.caption("Latest values are the final window's aggregates of simulated readings. The watchdog cannot tell a dead PIR or door from an empty room (engine documentation).")

st.markdown("#### Recent alerts and events")
ev = rec.events_df
if ev.empty:
    st.caption("The engine raised no security events in this run.")
else:
    st.dataframe(ev.tail(10).iloc[::-1], hide_index=True, use_container_width=True)
