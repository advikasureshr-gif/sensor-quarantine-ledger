"""Page G - Real Recording Run: a saved run of demo_pipeline.py on a recording, with the attack injected in software."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components import real_run as rr
from components import admin, theme
from iotguard.chain import Blockchain, Network
from iotguard.security import Ledger

admin.sidebar_status()
theme.page_header("Real Recording Run", "A recording replayed through the engine: calibration, a software-injected attack, quarantine, recovery, and the ledger.")


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float):
    return rr.load_run(path)


runs = rr.find_runs()
custom = st.sidebar.text_input("…or a run folder path", "", help="A folder written by demo_pipeline.py (it contains meta.json and windows.csv).")
path = None
if custom.strip():
    if rr.is_run(Path(custom.strip())):
        path = Path(custom.strip())
    else:
        st.sidebar.error("That folder is not a saved run.")
elif runs:
    path = st.sidebar.selectbox("Saved run", runs, format_func=lambda p: f"{p.name}  ({p.parent.name})")
if path is None:
    theme.empty_state("No saved run yet", "Run:  python demo_pipeline.py --replay logs/<recording>.jsonl --attack-start-h <H>   "
                      "then reload this page. Runs are saved in the project's exports/ folder.")
    st.stop()

run = _load(str(path), path.stat().st_mtime)
meta, w = run.meta, run.windows
rec_name = Path(meta.get("recording", "")).name
simulated = "sim" in rec_name.lower()
st.markdown(
    f'<div class="ig-sim" style="border-color:{theme.CYAN};color:{theme.CYAN};background:rgba(34,211,238,.06)">'
    f'Replay of <b>{rec_name}</b>' + (" (a simulated stand-in recording)" if simulated else " (readings recorded from the ESP32)") +
    ". The attack was injected <b>in software</b> into the data stream: no sensor was touched. Everything below is the engine's real output for this run.</div>",
    unsafe_allow_html=True)
st.write("")

alerts = [e for e in run.events if e["kind"] == "alert"]
atk = meta.get("attack", {})
q_h = meta.get("quarantined_at_h")
c = st.columns(6)
with c[0]:
    theme.kpi("Recording", f"{run.rec['hours']:.1f} h", f"{run.rec['readings']:,} readings", "cyan")
with c[1]:
    theme.kpi("Calibration ends", f"{meta.get('calibration_end_h', 0):.1f} h", f"preset: {meta.get('preset')}", "blue")
with c[2]:
    theme.kpi("Attack (software)", f"{atk.get('hours', 0):g} h", f"from {atk.get('start_h', 0):g} h, {atk.get('sensor', 'pir').upper()} forced to {atk.get('forced_to', 1)}", "red")
with c[3]:
    theme.kpi("First alert", f"{alerts[0]['t_h']:.1f} h" if alerts else "none", f"{len(alerts)} alert(s) in total", "amber")
with c[4]:
    theme.kpi("Quarantine", f"{q_h:.1f} h" if q_h is not None else "none",
              f"{q_h - atk.get('start_h', 0):.1f} h after the attack began" if q_h is not None else "no sensor isolated", "red" if q_h is not None else "muted")
with c[5]:
    theme.kpi("Recovered", "yes" if meta.get("recovered") else "no", "ledger intact" if meta.get("ledger_intact") else "LEDGER BROKEN",
              "green" if meta.get("recovered") else "muted")
st.write("")

t_time, t_learn, t_events, t_chain, t_rec = st.tabs(["Timeline", "Learning rate", "Events", "Ledger & blockchain", "Recording logs"])

with t_time:
    o1, o2 = st.columns([1, 1])
    view = o1.radio("View", ["Whole run", "Around the attack"], horizontal=True)
    naive = o2.toggle("Show the always-learning twin (a baseline with no gate)", value=True)
    focus = None
    if view == "Around the attack" and run.attack_span:
        last = max([e["t_h"] for e in run.events if e.get("t_h") is not None] + [run.mark_h("admin_release") or 0.0, run.attack_span[1]])
        focus = (run.attack_span[0] - 4.0, last + 2.5)
    st.plotly_chart(rr.timeline_figure(run, naive, focus), use_container_width=True)
    st.caption("During the attack the gate labels windows SUSPICIOUS and freezes learning. After the sensor is quarantined it is masked out, "
               "so the remaining sensors can be trusted and learning resumes without it. The twin has no gate and learns whatever it sees.")

with t_learn:
    t = rr.mode_table(run)
    a, b = st.columns([1, 1])
    a.plotly_chart(rr.mode_figure(run), use_container_width=True)
    b.dataframe(t, use_container_width=True)
    st.markdown(
        "- **full** (TRUSTED): the sensors corroborate each other, so the baseline learns at full rate.\n"
        "- **slow** (UNCERTAIN): weak or mixed evidence, so it learns slowly.\n"
        "- **freeze** (SUSPICIOUS): a strong contradiction, so learning stops and an attacker cannot teach the baseline a false normal.")
    st.caption("Learning rate (alpha) per window is in windows.csv. Windows during the attack that are `full` come after the PIR was quarantined and masked.")

with t_events:
    ev = pd.DataFrame(run.events).sort_values("t_h") if run.events else pd.DataFrame(columns=["kind", "t_h", "sensor", "detail"])
    st.markdown("#### Security events")
    st.dataframe(ev, hide_index=True, use_container_width=True)
    st.markdown("#### Run milestones")
    st.dataframe(pd.DataFrame(run.marks), hide_index=True, use_container_width=True)

with t_chain:
    if not admin.is_admin():
        st.markdown("#### 🔒 Ledger and blockchain are admin-only")
        admin.login_form("run_chain_login")
    else:
        led = Ledger()
        led.records = [dict(r) for r in run.ledger]
        ok, bad = led.verify()
        st.markdown(f"**Hash-chain check, re-run now on the saved records:** " + (theme.pill("INTACT", "green") if ok else theme.pill(f"BROKEN at record {bad}", "red")),
                    unsafe_allow_html=True)
        st.caption("Each record stores the hash of the one before it, like a git commit naming its parent. Edit an old record and every hash after it stops matching.")
        ldf = pd.DataFrame([{"seq": r["seq"], "event": r["event"], "t_h": None if r.get("t") is None else round(r["t"] / 3600.0, 2),
                             "sensor": r["data"].get("sensor") or "", "hash": r["hash"][:16], "prev": r["prev"][:16]} for r in run.ledger])
        st.dataframe(ldf, hide_index=True, use_container_width=True)

        st.markdown("#### As a blockchain")
        chain = Blockchain.build([dict(r) for r in run.ledger])
        cok, cblk, cwhy = chain.verify()
        st.markdown(f"Records are batched into blocks of {len(chain.blocks[0].records)}. Each block has a Merkle root of its records, a proof-of-work nonce "
                    f"(difficulty {chain.blocks[0].difficulty} hex zeros) and the hash of the previous block. Chain check: "
                    + (theme.pill("INTACT", "green") if cok else theme.pill(f"BROKEN at block {cblk}", "red")), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"block": b.index, "records": f"#{b.records[0]['seq']}–#{b.records[-1]['seq']}", "merkle root": b.merkle[:14], "nonce": b.nonce,
                                    "hash": b.hash[:16], "previous block": b.prev_hash[:16]} for b in chain.blocks]), hide_index=True, use_container_width=True)

        st.markdown("#### Three replicas, one vote")
        st.caption("Three copies of the chain are kept. A block is accepted only if most copies hold the same hash. "
                   "The replicas run in one process here (a simulation, not a network).")
        key = f"net_{path}"
        victim = next((r for r in run.ledger if r["event"] == "sensor_quarantined"), run.ledger[len(run.ledger) // 2])
        if key not in st.session_state:
            st.session_state[key] = {"net": Network(chain, 3), "log": []}
        s = st.session_state[key]

        def blame_ldr(r):
            r["data"]["sensor"] = "ldr"

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Attack 1: edit one record on node-B", key=f"a1_{path}"):
            s["net"].edit_record("node-B", victim["seq"], blame_ldr)
            s["log"].append(f"node-B: record #{victim['seq']} ({victim['event']}) edited to blame the LDR; nothing else changed.")
        if b2.button("Attack 2: skilled rewrite on node-B", key=f"a2_{path}"):
            s["net"].rewrite_history("node-B", victim["seq"], blame_ldr)
            s["log"].append(f"node-B: record #{victim['seq']} rewritten and every later hash, Merkle root and block recomputed.")
        if b3.button("Repair from the majority", key=f"a3_{path}"):
            fixed = s["net"].heal()
            s["log"].append(f"repaired from the majority: {', '.join(fixed) or 'nothing to repair'}.")
        if b4.button("Reset", key=f"a4_{path}"):
            st.session_state[key] = {"net": Network(chain, 3), "log": []}
            st.rerun()
        net = s["net"]
        vote = net.vote()
        rows = []
        for nd in net.nodes:
            own_ok, own_blk, own_why = nd.chain.verify()
            rows.append({"replica": nd.name, "own check": "passes" if own_ok else f"FAILS at block {own_blk}: {own_why}", "chain head": nd.chain.head[:16],
                         "majority vote": f"DISAGREES from block {vote.dissenters[nd.name]}" if nd.name in vote.dissenters else "agrees"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if s["log"]:
            st.markdown("\n".join(f"- {x}" for x in s["log"]))
        if vote.dissenters:
            st.markdown(theme.pill("VOTE: " + ", ".join(vote.dissenters) + " REJECTED", "red"), unsafe_allow_html=True)
            st.caption("A skilled attacker can make one replica pass its own check, but cannot make it match the other two.")
        else:
            st.markdown(theme.pill("VOTE: all replicas agree", "green"), unsafe_allow_html=True)

with t_rec:
    st.markdown("#### What the raw recording looked like")
    cols = st.columns(4)
    r = run.rec
    with cols[0]:
        theme.kpi("PIR", f"{r['pir_high_pct']:.1f}% high", f"{r['pir_triggers']} triggers, longest {r['pir_longest_high_s']:.0f} s", "cyan")
    with cols[1]:
        theme.kpi("Door", f"{r['door_changes']} changes", f"open {r['door_open_pct']:.1f}% of the time", "amber")
    with cols[2]:
        theme.kpi("DHT11", f"{r['dht_null_pct']:.1f}% failed", f"{r['dht_null']} failed reads; {r['temp_min']}–{r['temp_max']} °C, {r['hum_min']}–{r['hum_max']} %RH" if r["temp_min"] is not None else f"{r['dht_null']} failed reads", "green")
    with cols[3]:
        theme.kpi("Stream", f"{r['gaps_over_5s']} gaps", f"longest gap {r['max_gap_s']:.0f} s", "green" if r["gaps_over_5s"] == 0 else "red")
    st.write("")
    tone = {"ok": "green", "warn": "amber", "info": "blue"}
    for kind, text in rr.recording_notes(r):
        st.markdown(theme.pill({"ok": "OK", "warn": "CHECK", "info": "INFO"}[kind], tone[kind]) + "&nbsp; " + text, unsafe_allow_html=True)
    if r.get("quietest_stretches"):
        q = r["quietest_stretches"][0]
        st.caption(f"Longest quiet stretch (no door change, no PIR high longer than 6 s): {q['start_h']:.1f}–{q['end_h']:.1f} h ({q['hours']:.1f} h). "
                   "That is where an attack is best injected.")
    st.plotly_chart(rr.recording_figure(run), use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("Recording summary (JSON)", json.dumps(r, indent=1), f"{rec_name}_summary.json", "application/json")
    d2.download_button("Per-window results (CSV)", w.to_csv(index=False), "windows.csv", "text/csv")
