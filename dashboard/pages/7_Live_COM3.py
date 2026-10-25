"""Live page - the real ESP32 on a serial port: raw values, the engine's label for each 30 s window, and the learning rate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from components import admin, session_view as sv, theme
from components.runner import EngineRun, find_recordings, list_serial_ports

theme.page_header("Live sensors (COM3)", "Real readings from the ESP32, scored by the real pipeline as they arrive.")
admin.sidebar_status()
h = sv.hub()
run: EngineRun | None = h["live"]
running = run is not None and run.alive

if not running:
    recs = find_recordings()
    if run is not None and run.error:
        st.error(f"The last live session ended with an error: {run.error}")
    if not recs:
        theme.empty_state("No recording found", "The baseline is learned from a recording first. Put the merged recording (room_merged.jsonl) in the project's logs folder.")
        st.stop()
    c1, c2 = st.columns(2)
    sim = c2.checkbox("No board connected: pretend the board is sending (replays the recording as a live stream)", value=False)
    ports = list_serial_ports()
    port = c1.text_input("Serial port", ports[0] if ports else "COM3", help="Close the Arduino Serial Monitor and the recorder first: only one program can hold the port.")
    if ports:
        c1.caption("Ports found: " + ", ".join(ports))
    warm = st.selectbox("Learn the baseline from this recording", recs, format_func=lambda p: p.name)
    factor = st.number_input("Calibration shortening factor", 1.0, 8.0, 2.0, 0.5,
                             help="2 = a 24 h calibration (needs a recording of at least 24 h). 1 = the full 48 h.")
    st.caption("Starting takes about 10 seconds: the engine learns the room from the recording, then the live readings follow. "
               "No attack is injected unless you switch the software attack on below the charts.")
    if st.button("▶ Start live", type="primary"):
        def factory():
            if sim:
                from iotguard.sources import ReplaySource
                return ReplaySource(str(warm), realtime=30.0)
            from iotguard.sources import SerialSource
            return SerialSource(port)
        run = EngineRun("live", f"Live on {port}", admin.auth_service(), admin.server_secret(), demo=float(factor))
        h["live"] = run
        run.start_live(str(warm), factory)
        st.rerun()
    st.stop()

top = st.columns([1, 1, 3])
if top[0].button("■ Stop"):
    run.stop()
    st.rerun()
atk = top[1].toggle("Software PIR attack", value=run.attack_on,
                    help="Forces the PIR value fed to the engine to 1 while the real sensor is untouched. The raw chart shows both.")
run.attack_on = atk


@st.fragment(run_every=1.5)
def body():
    r = run
    if r.status == "error":
        st.error(r.error)
        return
    if r.status == "calibrating":
        st.info(r.note)
        return
    st.caption(f"Source: {r.stats_text() or '—'}  ·  saving the real readings to {r.log_path or '—'} ({r.log_lines:,} lines; never touches your logs folder)")
    if atk:
        st.markdown('<div class="ig-sim" style="border-color:#ef4444;color:#ef4444">SOFTWARE ATTACK ON: the engine is being fed PIR = 1; the real sensor is not touched.</div>', unsafe_allow_html=True)
    sv.kpis(r)
    sv.quarantine_banner(r, "live")
    last = r.latest
    if last:
        c = st.columns(5)
        vals = [("Temperature", "—" if last["temp"] is None else f"{last['temp']:.1f} °C"), ("Humidity", "—" if last["hum"] is None else f"{last['hum']:.0f} %"),
                ("PIR", "MOTION" if last["pir"] else "none"), ("Door", "OPEN" if last["door"] else "closed"),
                ("Light", "dark" if last["ldr"] > 2000 else "light")]
        for col, (k, v) in zip(c, vals):
            with col:
                theme.kpi(k, v, "", "cyan")
    st.plotly_chart(sv.raw_figure(r), use_container_width=True)
    st.plotly_chart(sv.learning_figure(r, window_h=2.0), use_container_width=True)
    st.caption("Every point is one 30-second window. SUSPICIOUS windows freeze learning (alpha = 0). A real quarantine needs about 3 hours of "
               "sustained suspicion, so it is shown on the Demo Replay page instead.")


body()
