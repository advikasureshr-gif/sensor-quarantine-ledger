"""IoTGuard dashboard entry point. Run with:  python -m streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from components import theme

st.set_page_config(page_title="IoTGuard — Behavioural Anomaly Detection", page_icon="🛡️", layout="wide")
theme.inject_css()
theme.sidebar_brand()

nav = st.navigation([
    st.Page("overview.py", title="Security Overview", icon="🛡️", default=True),
    st.Page("pages/2_Attack_Simulator.py", title="Attack Simulator", icon="🎯"),
    st.Page("pages/1_Sensor_Monitor.py", title="Sensor Monitor", icon="📡"),
    st.Page("pages/3_Detection_Analysis.py", title="Detection Analysis", icon="🔬"),
    st.Page("pages/4_Experiment_Results.py", title="Experiment Results", icon="📊"),
    st.Page("pages/5_Security_Events.py", title="Security Events", icon="🚨"),
])
st.sidebar.caption("Simulator-driven. No physical hardware is connected.")
nav.run()
