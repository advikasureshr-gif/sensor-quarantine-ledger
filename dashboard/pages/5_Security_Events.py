"""Page F - Security Events and the engine's hash-chained ledger."""
import json

import pandas as pd
import streamlit as st

import dashboard_service as ds
from components import theme
from components.state import pick_run

theme.page_header("Security Events", "Events and ledger records produced by the engine in the selected run.")
theme.sim_badge()
rec = pick_run()
if rec is None:
    theme.empty_state("No data yet", "Run a scenario in the Attack Simulator first.")
    st.stop()
ev = rec.events_df
st.markdown("#### Security events")
if ev.empty:
    st.info("The engine raised no security events (alerts, quarantines, calibration failures or hardware-watchdog reports) in this run.")
else:
    kinds = st.multiselect("Event kind", sorted(ev["kind"].unique()), default=sorted(ev["kind"].unique()))
    st.dataframe(ev[ev["kind"].isin(kinds)], hide_index=True, use_container_width=True)
    st.download_button("Events CSV", ev.to_csv(index=False), f"{rec.run_id}_events.csv", "text/csv")
st.markdown("#### Ledger")
if not rec.ledger:
    st.info("No ledger records are stored for this run.")
else:
    ok, bad = ds.verify_records(rec.ledger)
    st.markdown(f"Hash-chain check (engine's `Ledger.verify`, re-run now on the stored records): "
                f"{theme.pill('chain intact' if ok else f'BROKEN at record {bad}', 'green' if ok else 'red')} · {len(rec.ledger)} records", unsafe_allow_html=True)
    st.caption("Each record's SHA-256 covers the previous record's hash, so edits to stored records are detectable. This is tamper-EVIDENCE only: "
               "the chain has no secret key or external anchor, so someone who can rewrite the whole file could recompute it, and it is not called immutable here. "
               "In this dashboard the ledger lives in memory/outputs of a simulation, not in a protected store.")
    led = pd.DataFrame([{"seq": r["seq"], "event": r["event"], "t": r["t"], "wall": r["wall"], "data": str(r["data"]), "hash": r["hash"][:16] + "…"} for r in rec.ledger])
    st.dataframe(led, hide_index=True, use_container_width=True)
    st.download_button("Ledger JSON (full hashes)", json.dumps(rec.ledger), f"{rec.run_id}_ledger.json", "application/json")
