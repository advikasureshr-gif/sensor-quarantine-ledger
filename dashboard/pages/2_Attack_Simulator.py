"""Page B - Attack Simulator: run a real engine scenario through the real pipeline."""
import pandas as pd
import streamlit as st

import dashboard_service as ds
from components import charts, theme
from components.state import get_manager

theme.page_header("Attack Simulator", "Pick a scenario from the engine's own scenario list, run it through the real pipeline, inspect the outcome.")
theme.sim_badge()
mgr = get_manager()
catalog = {c["name"]: c for c in ds.scenario_catalog()}

active = mgr.active
left, right = st.columns([1, 1.3])
with left:
    st.markdown("#### Configuration")
    name = st.selectbox("Scenario", list(catalog), format_func=lambda n: f"{n} — {catalog[n]['title']}", disabled=active is not None)
    cat = catalog[name]
    seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=0, step=1, disabled=active is not None, help="Same scenario + seed + settings regenerates the identical stream.")
    days = st.slider("Duration (simulated days)", min_value=float(cat["min_days"]), max_value=float(ds.MAX_DAYS), value=float(max(cat["min_days"], cat["default_days"])),
                     step=0.5, disabled=active is not None,
                     help=f"Minimum {cat['min_days']:g} d so every scripted attack falls inside the run. The first {cat['calibration_days']:g} d are calibration. Longer runs take longer.")
    strength = st.slider("Strength", 0.1, 3.0, 1.0, 0.1, disabled=active is not None, help="Engine 'strength' parameter: scales attack size / affected-sample fraction and legitimate-change magnitude.")
    profile = st.selectbox("Room / sensor profile", ds.profile_names(), disabled=active is not None)
    st.caption(f"Rough cost: the engine processes about 10 simulated days in a few minutes on a laptop; this run is {days:g} d.")

    b1, b2, b3 = st.columns(3)
    start = b1.button("▶ Start", type="primary", disabled=active is not None, use_container_width=True)
    cancel = b2.button("■ Cancel", disabled=active is None, use_container_width=True)
    new = b3.button("↺ New run", disabled=active is not None, use_container_width=True, help="Clears the on-screen selection; saved runs stay available.")
    if new:
        st.session_state.pop("selected_run", None)
        st.rerun()
    if cancel:
        mgr.cancel_active()
        st.rerun()
    if start:
        try:
            params = ds.validate_params(name, int(seed), float(days), float(strength), profile)
            rec = mgr.start(params)
            st.session_state["selected_run"] = rec.run_id
            st.rerun()
        except (ds.ParamError, ds.RunBusyError) as e:
            st.error(str(e))

with right:
    st.markdown(f"#### {cat['title']}")
    st.markdown(f"**Project experiment:** {cat['experiment']}")
    st.markdown(f"**Author's expectation (a hypothesis, not a result):** {cat['expect']}")
    if cat["contaminates_calibration"]:
        st.warning("This scenario deliberately tampers with the calibration period.")
    try:
        sched = ds.preview_schedule(ds.validate_params(name, int(seed), float(days), float(strength), profile))
        rows = [{"kind": "attack", "tag": a["tag"], "channel": a["channel"], "type": a["kind"], "start (h)": round(a["start_s"] / 3600, 1),
                 "end (h)": None if a["end_s"] is None else round(a["end_s"] / 3600, 1)} for a in sched["attacks"]]
        rows += [{"kind": "legitimate change", "tag": c["kind"], "channel": "—", "type": "—", "start (h)": round(c["start_s"] / 3600, 1),
                  "end (h)": None if c["label_end_s"] is None else round(c["label_end_s"] / 3600, 1)} for c in sched["legit_changes"]]
        st.markdown(f"**Ground-truth schedule** (simulated hours; calibration ends at {sched['calibration_s'] / 3600:g} h; empty end = continues to the end of the run)")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No attacks or legitimate changes are scripted: this is a baseline-behaviour run.")
    except ds.ParamError as e:
        st.error(str(e))

st.divider()


def _progress_body():
    rec = mgr.runs.get(st.session_state.get("selected_run", ""))
    if rec is None:
        theme.empty_state("No run selected", "Configure a scenario and press Start.")
        return
    if rec.status in ("queued", "running"):
        st.progress(rec.progress, text=f"Run {rec.run_id}: {rec.status} — {rec.progress:.0%} (simulated day {len(rec.rows) * 30 / 86400:.1f})")
        last = rec.rows[-1] if rec.rows else None
        if last:
            st.caption(f"Phase {last['phase']} · latest decision {last['decision'] or '(calibrating)'} · events so far {len(rec.events)}")
    else:
        st.session_state["_just_finished"] = rec.run_id
        st.rerun()                                          # leave the auto-refresh fragment and show the results


if active is not None:
    st.fragment(run_every=1.5)(_progress_body)()
else:
    rec = mgr.runs.get(st.session_state.get("selected_run", ""))
    if rec is None:
        theme.empty_state("No run selected", "Configure a scenario and press Start. The result appears here and on every other page.")
    elif rec.status == "failed":
        st.error(f"Run {rec.run_id} failed: {rec.error}")
    else:
        if rec.status == "cancelled":
            st.warning(f"Run {rec.run_id} was cancelled. Results below are partial.")
        s = rec.summary
        st.markdown(f"#### Result — {rec.label}")
        c = st.columns(5)
        with c[0]: theme.kpi("Windows", f"{s.get('windows_processed', 0):,}", "processed")
        with c[1]: theme.kpi("Trusted", s["decisions"]["TRUSTED"], "", "green")
        with c[2]: theme.kpi("Uncertain", s["decisions"]["UNCERTAIN"], "", "amber")
        with c[3]: theme.kpi("Suspicious", s["decisions"]["SUSPICIOUS"], "", "red")
        with c[4]:
            lat = s.get("latency", {})
            val = "n/a" if not lat.get("available") else ("missed" if not lat.get("detected") else f"{lat['latency_s']:.0f} s")
            theme.kpi("Detection latency", val, f"first attack: {lat.get('attack_tag', '—')}", "blue")
        thr = rec.config.get("anomaly", {}).get("mahalanobis_threshold")
        st.plotly_chart(charts.score_figure(rec.windows, thr, rec.manifest), use_container_width=True)
        st.plotly_chart(charts.decision_strip(rec.windows, rec.manifest), use_container_width=True)
        st.caption("Red bands = scripted attack windows from the simulator's ground truth; blue = legitimate changes. SUSPICIOUS means the engine held the window; it is not proof of an attack.")
        if rec.events:
            st.markdown("**Security events**")
            st.dataframe(rec.events_df, hide_index=True, use_container_width=True)
        st.page_link("pages/3_Detection_Analysis.py", label="Open Detection Analysis →", icon="🔬")
        st.page_link("pages/4_Experiment_Results.py", label="Compare & export runs →", icon="📊")
