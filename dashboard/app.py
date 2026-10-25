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
    st.Page("pages/7_Live_COM3.py", title="Live sensors (COM3)", icon="📡"),
    st.Page("pages/8_Demo_Replay.py", title="Demo Replay", icon="🎬"),
    st.Page("pages/2_Attack_Simulator.py", title="Attack Simulator", icon="🎯"),
    st.Page("pages/6_Real_Recording_Run.py", title="Saved Run", icon="📼"),
    st.Page("pages/9_Admin.py", title="Admin", icon="🔒"),
])
st.sidebar.caption("Live values, replay and the simulator are open. Ledger, blocks and downloads need the admin.")
nav.run()
