# IoTGuard — Behavioural Anomaly Detection & Attack Simulation Platform (dashboard)

A Streamlit dashboard that runs your teammate's **unmodified** `iotguard` engine on its built-in simulator scenarios and shows the
engine's real decisions, scores, security events and ledger. This package is self-contained: the needed engine files are copied
into `iotguard/` (see `iotguard/NOTICE.md`). It does not touch your original project folder.

**The simulator pages are simulated.** No ESP32 or physical sensor is connected and nothing live is read. The extra page **Real Recording Run** only *loads files* saved by the project's `demo_pipeline.py` (`../exports/<run>/` or `outputs/real_runs/<run>/`); it never talks to a sensor and never runs the engine.

## Prerequisites
* Windows 10/11, **Python 3.10 or newer** (python.org installer, tick "Add python.exe to PATH")
* Internet access for the first run only (to install Streamlit, Plotly, Pandas, NumPy)

## Launch (VS Code / PowerShell)
1. Extract the ZIP into a new folder and open that folder in VS Code (`File > Open Folder`).
2. Open a terminal (`Ctrl+``) and run:
```
.\run_dashboard.bat
```
   (In PowerShell use `.\run_dashboard.bat`; in cmd just `run_dashboard.bat`.) The first run creates `.venv` and installs dependencies; later runs start immediately. The browser opens at **http://localhost:8501**; if not, open that address yourself.

Manual alternative:
```
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Use it
1. Open **Attack Simulator**, choose a scenario (only scenarios defined in `iotguard/scenarios.py` are offered), set seed / duration / strength / profile, press **Start**.
2. Progress updates live. A multi-day run processes about 20,000 windows per 7 simulated days; expect roughly 1–2 minutes for a 7-day run (about 68 s measured in the author's sandbox; your PC will differ). Only one run can execute at a time; **Cancel** stops a run and keeps its partial results.
3. Review the result on the same page, then **Detection Analysis**, **Sensor Monitor**, **Security Events**.
4. In **Experiment Results**, compare runs and export CSV/JSON. Runs are saved under `outputs/runs/<run_id>/` (JSON + CSV, no pickle) and reloaded next time.

## What the numbers mean
* One row = one 30 s window, as produced by the engine's pipeline. Calibration (first 2 simulated days) produces no decisions.
* SUSPICIOUS = the engine's consensus gate held the window. It is **not** proof of an attack. Ground truth (red bands) comes only from the simulator's own scenario manifest.
* Precision/recall/F1/FPR use windows of the operating phase; positive = overlaps a scripted attack; predicted positive = SUSPICIOUS. They are blank when undefined. Detection latency measures only the first scripted attack in the operating phase. One seed is one sample, not a statistic.
* The ledger is a SHA-256 hash chain: it makes edits **detectable** (tamper-evident). It has no secret key or external anchor, so it is not described as immutable.
* Sensor charts show per-window aggregates; the "door" chart is the count of open/close transitions, not the door's state.
* The engine's authors list known limits (e.g. light-only attacks and sudden self-consistent poisoning are not caught; `gradual_poison` seed 0 is detected late). See `iotguard/ORIGINAL_README.md`. The dashboard does not hide or tune these.

## Tests
```
.\.venv\Scripts\python.exe -m pytest tests -q
```
`tests/test_dashboard_service.py` covers validation, ground-truth labelling, metric maths, ledger tamper detection, the one-run-at-a-time lock, and one real 2.5-day engine run (marked `slow`, about 1 minute). Skip it with `-m "not slow"`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| "Python was not found" | Install Python 3.10+ and tick "Add to PATH", reopen the terminal. |
| `running scripts is disabled` in PowerShell | Run `cmd /c run_dashboard.bat` or `Set-ExecutionPolicy -Scope Process Bypass`. |
| pip install fails | Check internet/proxy; delete `.venv` and rerun. To reinstall after editing requirements, delete `.venv\.deps_ok`. |
| Port 8501 busy | `.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8600` |
| "A run is already in progress" | Wait, or press Cancel. Restarting the server also clears it. |
| `ModuleNotFoundError: iotguard` | Run from the package folder; `iotguard/` must sit beside `app.py`. |
| Blank charts | No run selected yet; start a run in the Attack Simulator. |

## Layout
`app.py` (navigation) · `overview.py` (Security Overview) · `dashboard_service.py` (the only engine integration) · `pages/` · `components/` · `iotguard/` (vendored engine, unmodified) · `tests/` · `integration/README.md` · `outputs/` (generated runs)


## Real Recording Run page (added)
Shows a saved `demo_pipeline.py` run: timeline (score, learning mode, drift, masked sensors) with the attack window and events, learning-rate table, events, the ledger and its blocks with a live three-replica vote, and a summary of the raw recording. `outputs/real_runs/sample_sim_run/` is an example. Tests: `tests/test_real_run_page.py`.

## Pages (current)
- **Security Overview** - the running session (live board or demo replay): label, learning rate, sensor states, events. A quarantined sensor shows a
  **Deal with quarantined sensor** button: admin sign-in, then *Unquarantine* or *Replace with a new sensor*; both put the sensor on probation.
- **Live sensors (COM3)** - real readings from the ESP32 (needs `pyserial`), the engine's label for each 30 s window, the learning rate and the
  learning curve. The baseline is first learned from a recording in `..\logs` (about 10 s). A *Software PIR attack* switch feeds the engine PIR = 1
  without touching the sensor. The real readings are saved to `dashboard\outputs\live_logs\` (separate from your logs). Close the Arduino Serial Monitor and the recorder first: only one program can hold the port.
- **Demo Replay** - a recording through the whole pipeline with a software PIR attack; pauses when the PIR is quarantined and waits for the admin.
- **Attack Simulator** - the friend's simulator (open to everyone). **Saved Run** - a run saved by `demo_pipeline.py`.
- **Admin** - ledger, calibration records, blockchain with the three-replica vote, and their downloads.
Admin sign-in: secret from the `IOTGUARD_ADMIN_SECRET` environment variable, otherwise the demo secret `demo-admin-secret`; 3 wrong attempts lock
sign-in for 2 minutes, and failed attempts are written to the session ledger. The older analysis pages are kept in `archive/` (not in the menu).
