"""Page C - Sensor Monitor: per-window aggregates of the SIMULATED sensor channels."""
import streamlit as st

from components import charts, theme
from components.state import pick_run

theme.page_header("Sensor Monitor", "Window-level aggregates of the simulated DHT, PIR, door and LDR channels, with the engine's flags overlaid.")
theme.sim_badge()
rec = pick_run()
if rec is None:
    theme.empty_state("No data yet", "Run a scenario in the Attack Simulator first.")
    st.stop()
df = rec.windows
chosen = st.multiselect("Channels", list(charts.CHANNELS), default=["temp_mean", "hum_mean", "pir_duty", "door_count", "ldr_mean"],
                        format_func=lambda c: charts.CHANNELS[c][0])
c1, c2 = st.columns(2)
markers = c1.toggle("Mark UNCERTAIN / SUSPICIOUS windows", value=True)
truth = c2.toggle("Show simulator ground-truth bands", value=True)
hmax = float(df["t_start"].max() / 3600.0)
lo, hi = st.slider("Time range (simulated hours)", 0.0, max(hmax, 1.0), (0.0, max(hmax, 1.0)))
view = df[(df["t_start"] >= lo * 3600) & (df["t_start"] <= hi * 3600)]
st.caption("Each point is one 30-second window (mean/count over its readings after the engine's preprocessing), not a raw sample. "
           "The door channel is the number of confirmed open/close transitions in the window, not the door's state. Markers are drawn only in the operating phase.")
for ch in chosen:
    st.plotly_chart(charts.channel_figure(view, ch, rec.manifest, markers, truth), use_container_width=True)
st.markdown("#### Recent windows")
cols = ["index", "t_start", "phase", "decision", "score", *charts.CHANNELS]
st.dataframe(df[cols].tail(50).iloc[::-1], hide_index=True, use_container_width=True)
