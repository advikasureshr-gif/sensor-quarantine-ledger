"""Shared state: one RunManager per server process, and the 'selected run' shared by every page."""
from __future__ import annotations

from typing import Optional

import streamlit as st

from dashboard_service import RunManager, RunRecord


@st.cache_resource(show_spinner=False)
def get_manager() -> RunManager:
    return RunManager()


def results_runs() -> list[RunRecord]:
    """Runs that have windows to look at (finished, or cancelled/failed with partial data)."""
    return [r for r in get_manager().ordered() if r.status in ("completed", "cancelled", "failed") and len(r.windows) > 0]


def pick_run(key: str = "selected_run") -> Optional[RunRecord]:
    runs = results_runs()
    if not runs:
        return None
    ids = [r.run_id for r in runs]
    if st.session_state.get(key) not in ids:
        st.session_state[key] = ids[0]
    by_id = {r.run_id: r for r in runs}
    chosen = st.sidebar.selectbox("Run", ids, key=key,
                                  format_func=lambda i: by_id[i].label + ("  (partial)" if by_id[i].status != "completed" else ""))
    return by_id[chosen]
