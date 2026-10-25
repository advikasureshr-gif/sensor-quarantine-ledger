"""Page E - Experiment Results: compare completed runs; metrics only where ground truth supports them."""
import streamlit as st

import dashboard_service as ds
from components import charts, theme
from components.state import get_manager

theme.page_header("Experiment Results", "Compare runs, then export CSV or JSON. Metrics appear only where the simulator supplies ground truth.")
mgr = get_manager()
runs = [r for r in mgr.ordered() if r.status != "queued" and r.status != "running"]
if not runs:
    theme.empty_state("No completed runs", "Run one or more scenarios in the Attack Simulator, then compare them here.")
    st.stop()
table = ds.comparison_table(runs)
st.dataframe(table, hide_index=True, use_container_width=True)
st.caption("Precision / recall / F1 / FPR: 30-second windows in the operating phase; positive = window overlaps a scripted attack; "
           "predicted positive = SUSPICIOUS. Blank = not computable (e.g. no attack windows, or no positive predictions). "
           "Windows after an attack ends count as negatives, so the engine's deliberate hold after an attack lowers precision. "
           "Single seeds are anecdotes, not statistics.")
ids = st.multiselect("Select runs to compare or inspect", [r.run_id for r in runs], default=[runs[0].run_id])
chosen = [r for r in runs if r.run_id in ids]
if chosen:
    sel = ds.comparison_table(chosen)
    c1, c2 = st.columns(2)
    if sel["latency_s"].notna().any():
        c1.plotly_chart(charts.compare_bar(sel, "latency_s", "Detection latency of first attack (s)"), use_container_width=True)
    c2.plotly_chart(charts.compare_bar(sel, "SUSPICIOUS", "SUSPICIOUS windows"), use_container_width=True)
    for r in chosen:
        ev = r.summary.get("evaluation", {})
        with st.expander(f"{r.label} — {r.status}"):
            if r.error:
                st.error(r.error)
            if not ev.get("available"):
                st.info(f"Evaluation not available: {ev.get('reason') or 'no summary'}")
                continue
            m = ev["strict"]
            st.markdown(f"Evaluated {ev['n_eval_windows']:,} windows ({ev['n_attack_windows']:,} attack, {ev['n_legit_change_windows']:,} legitimate-change). "
                        f"Strict (SUSPICIOUS only): precision {theme.fmt(m['precision'], 3)}, recall {theme.fmt(m['recall'], 3)}, F1 {theme.fmt(m['f1'], 3)}, FPR {theme.fmt(m['fpr'], 4)}.")
            lo = ev["loose"]
            st.caption(f"If UNCERTAIN were also treated as flagged: precision {theme.fmt(lo['precision'], 3)}, recall {theme.fmt(lo['recall'], 3)}, FPR {theme.fmt(lo['fpr'], 4)}.")
            st.plotly_chart(charts.confusion_figure(m["tp"], m["fp"], m["fn"], m["tn"]), use_container_width=True)
st.markdown("#### Export")
sel_runs = chosen or runs
e1, e2, e3, e4 = st.columns(4)
e1.download_button("Comparison CSV", ds.comparison_table(sel_runs).to_csv(index=False), "iotguard_comparison.csv", "text/csv", use_container_width=True)
e2.download_button("Summary JSON", ds.export_json(sel_runs), "iotguard_runs.json", "application/json", use_container_width=True)
if len(sel_runs) == 1:
    e3.download_button("Windows CSV (selected run)", sel_runs[0].windows.to_csv(index=False), f"{sel_runs[0].run_id}_windows.csv", "text/csv", use_container_width=True)
    e4.download_button("Full JSON incl. windows", ds.export_json(sel_runs, include_windows=True), f"{sel_runs[0].run_id}_full.json", "application/json", use_container_width=True)
else:
    e3.caption("Select exactly one run for per-window exports.")
st.caption(f"Runs are also saved on disk in {ds.OUTPUT_DIR}.")
