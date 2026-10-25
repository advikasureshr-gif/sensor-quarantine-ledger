"""Admin - sign in, then see and download the ledger, blocks and calibration records of the last live or replay session."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components import admin, session_view as sv, theme
from iotguard.chain import Blockchain, Network
from iotguard.security import Ledger

theme.page_header("Admin", "Ledger, blockchain and calibration records. Locked until the admin signs in.")
admin.sidebar_status()
if not admin.is_admin():
    st.markdown("#### 🔒 Sign in")
    admin.login_form("admin_page_login")
    st.info("Everything else (live sensors, replay, simulator) is open. Ledger, blocks and downloads are admin-only.")
    st.stop()

h = sv.hub()
opts = {k: v for k, v in (("Demo Replay", h["replay"]), ("Live COM3", h["live"])) if v is not None and v.ledger_records()}
if not opts:
    theme.empty_state("No session yet", "Start the Demo Replay or the Live page; its ledger appears here.")
    st.stop()
pick = st.sidebar.radio("Session", list(opts)) if len(opts) > 1 else next(iter(opts))
run = opts[pick]
records = run.ledger_records()
led = Ledger()
led.records = [dict(r) for r in records]
ok, bad = led.verify()
kinds = {"calibration_started", "calibration_finished", "recalibration_started", "baseline_checkpoint", "state_saved", "state_loaded", "baseline_registered"}

t_led, t_cal, t_chain = st.tabs(["Ledger", "Calibration records", "Blockchain"])
with t_led:
    st.markdown("**Hash-chain check, re-run now:** " + (theme.pill("INTACT", "green") if ok else theme.pill(f"BROKEN at record {bad}", "red")), unsafe_allow_html=True)
    df = pd.DataFrame([{"seq": r["seq"], "event": r["event"], "t_h": None if r.get("t") is None else round(r["t"] / 3600.0, 2),
                        "sensor": r["data"].get("sensor") or "", "detail": str(r["data"].get("detail", ""))[:80], "hash": r["hash"][:16], "prev": r["prev"][:16]}
                       for r in records])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("Download ledger (JSONL)", "\n".join(json.dumps(r, sort_keys=True) for r in records), f"{run.ledger_path.name}", "application/x-ndjson")
with t_cal:
    cal = [r for r in records if r["event"] in kinds or "calibration" in r["event"]]
    st.caption("Calibration start and finish, the baseline versions and checkpoints. Wrong admin attempts also show up here when they happen.")
    st.dataframe(pd.DataFrame([{"seq": r["seq"], "event": r["event"], "t_h": None if r.get("t") is None else round(r["t"] / 3600.0, 2),
                                "data": json.dumps(r["data"], sort_keys=True)[:140]} for r in cal]), hide_index=True, use_container_width=True)
    auth = [r for r in records if r["event"].startswith("auth_")]
    if auth:
        st.markdown("**Failed or locked admin attempts**")
        st.dataframe(pd.DataFrame([{"seq": r["seq"], "event": r["event"], "data": json.dumps(r["data"])} for r in auth]), hide_index=True, use_container_width=True)
with t_chain:
    if len(records) < 2:
        st.info("Not enough records yet.")
    else:
        chain = Blockchain.build([dict(r) for r in records])
        cok, cblk, _ = chain.verify()
        st.markdown(f"{len(records)} records in {len(chain.blocks)} blocks. Chain check: " + (theme.pill("INTACT", "green") if cok else theme.pill(f"BROKEN at block {cblk}", "red")), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"block": b.index, "records": f"#{b.records[0]['seq']}–#{b.records[-1]['seq']}", "merkle root": b.merkle[:14], "nonce": b.nonce,
                                    "hash": b.hash[:16], "previous block": b.prev_hash[:16]} for b in chain.blocks]), hide_index=True, use_container_width=True)
        st.download_button("Download blockchain (JSON)", json.dumps([{"index": b.index, "records": b.records, "merkle": b.merkle, "nonce": b.nonce, "difficulty": b.difficulty, "hash": b.hash, "prev_hash": b.prev_hash} for b in chain.blocks]), "blockchain.json", "application/json")
        st.markdown("#### Three replicas, one vote")
        key = f"net_{id(run)}_{len(chain.blocks)//1000}"
        victim = next((r for r in records if r["event"] == "sensor_quarantined"), records[len(records) // 2])
        if key not in st.session_state:
            st.session_state[key] = {"net": Network(chain, 3), "log": []}
        s = st.session_state[key]

        def blame(r):
            r["data"]["sensor"] = "ldr"
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Attack 1: edit one record on node-B"):
            s["net"].edit_record("node-B", victim["seq"], blame)
            s["log"].append(f"node-B: record #{victim['seq']} edited; nothing else changed.")
        if b2.button("Attack 2: skilled rewrite on node-B"):
            s["net"].rewrite_history("node-B", victim["seq"], blame)
            s["log"].append(f"node-B: record #{victim['seq']} rewritten and every later hash recomputed.")
        if b3.button("Repair from the majority"):
            s["log"].append(f"repaired: {', '.join(s['net'].heal()) or 'nothing to repair'}.")
        if b4.button("Reset"):
            st.session_state[key] = {"net": Network(chain, 3), "log": []}
            st.rerun()
        net = s["net"]
        vote = net.vote()
        rows = []
        for nd in net.nodes:
            o_ok, o_blk, o_why = nd.chain.verify()
            rows.append({"replica": nd.name, "own check": "passes" if o_ok else f"FAILS at block {o_blk}: {o_why}", "chain head": nd.chain.head[:16],
                         "majority vote": f"DISAGREES from block {vote.dissenters[nd.name]}" if nd.name in vote.dissenters else "agrees"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for x in s["log"]:
            st.markdown(f"- {x}")
        st.markdown(theme.pill("VOTE: " + ", ".join(vote.dissenters) + " REJECTED", "red") if vote.dissenters else theme.pill("VOTE: all replicas agree", "green"), unsafe_allow_html=True)
