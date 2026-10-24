"""Page D - Detection Analysis: the pipeline's recorded outputs and the evidence behind each decision."""
import pandas as pd
import streamlit as st

from components import charts, theme
from components.metrics import explain_window, operating, parse_evidence
from components.state import pick_run

theme.page_header("Detection Analysis", "What the engine decided, with which score, thresholds and evidence.")
theme.sim_badge()
rec = pick_run()
if rec is None:
    theme.empty_state("No data yet", "Run a scenario in the Attack Simulator first.")
    st.stop()
df, cfg = rec.windows, rec.config
thr = cfg.get("anomaly", {}).get("mahalanobis_threshold")
st.plotly_chart(charts.score_figure(df, thr, rec.manifest), use_container_width=True)
st.plotly_chart(charts.decision_strip(df, rec.manifest), use_container_width=True)
a, b = st.columns(2)
a.plotly_chart(charts.drift_figure(df, rec.manifest), use_container_width=True)
b.plotly_chart(charts.daily_decisions(df), use_container_width=True)
st.caption("Drift is the engine's own measure of how far the learned baseline has moved from its calibration. "
           "A SUSPICIOUS decision means the consensus gate held the window; it does not confirm an attack.")

with st.expander("Thresholds and configuration actually used in this run", expanded=False):
    st.json({k: cfg.get(k) for k in ("window_s", "sample_period_s", "anomaly", "online_stats", "security", "calibration", "preset")})
    st.caption("Consensus (cross-sensor) settings:")
    st.json(cfg.get("consensus", {}))

st.markdown("#### Why did the engine decide that? — inspect a window")
op = operating(df)
mode = st.radio("Show", ["SUSPICIOUS", "UNCERTAIN", "TRUSTED", "any"], horizontal=True)
pool = op if mode == "any" else op[op["decision"] == mode]
if pool.empty:
    st.info(f"No {mode} windows in this run.")
else:
    pick = st.select_slider("Window (by simulated hour)", options=list(pool["index"]), format_func=lambda i: f"#{i} · {pool[pool['index'] == i]['t_start'].iloc[0] / 3600:.2f} h")
    row = pool[pool["index"] == pick].iloc[0]
    for line in explain_window(row, thr):
        st.markdown("- " + line)
    ev = parse_evidence(row["evidence"])
    if ev:
        st.dataframe(pd.DataFrame(ev, columns=["evidence", "signed value", "detail"]), hide_index=True, use_container_width=True)

st.markdown("#### Reasons the engine reported (operating phase)")
rs = op["reason"].fillna("").value_counts().rename_axis("engine reason").reset_index(name="windows")
st.dataframe(rs, hide_index=True, use_container_width=True)
