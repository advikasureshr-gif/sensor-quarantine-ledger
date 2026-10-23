# online-behavioural-anomaly

ESP32 + DHT11 + HC-SR501 PIR + magnetic reed (door) + LDR: consensus-gated online behavioural
anomaly detection. A Mahalanobis baseline only learns from windows that other sensors corroborate
(TRUSTED -> full update, UNCERTAIN -> slow update, SUSPICIOUS -> freeze), so slow attacks cannot
poison it, while genuine changes in the room are still learned.

All attack results are on the built-in simulator. The ESP32 streams real readings into the same input path,
but the detector has not been calibrated or validated on a real room (a real calibration needs about 2 days of data).

## Setup
    python -m venv .venv && source .venv/bin/activate      (Windows: .venv\Scripts\activate)
    pip install -r requirements.txt

## Quick checks (run from this folder)
    python -m pytest iotguard -q          (30 tests)
    python -m iotguard.scenarios list

## Demo
    python demo.py all                    legit change, slow poisoning, PIR quarantine (about 4 minutes)
    python demo.py quarantine --plot      one act, plus the drift figure in figs/
    python demo.py hardware --port COM3   live ESP32 stream (Ctrl+C to stop)
    python plot_drift.py gradual_poison --seed 1 --days 7
`demo.py` prints only events and daily reports; the full per-window output goes to `logs/`.
`plot_drift.py` draws the gated baseline against an always-learning twin, with the attack window and quarantines marked.

## Full pipeline on one recording (`demo_pipeline.py`)
Replays a recording as a live stream, calibrates (admin-authenticated), runs normally, then forces the PIR high IN SOFTWARE
(no sensor is touched) while the room is empty, and shows: alert, quarantine of the PIR alone, a refused wrong admin secret,
authenticated release, probation and recovery, then the hash-chained ledger, a tamper test, and the blockchain layer (below).

    python -m iotguard.record --port COM3 --out logs/room.jsonl            (leave running; laptop awake and plugged in)
    python -m iotguard.inject --suggest logs/room.jsonl --min-start-h 50      (finds a quiet stretch to attack)
    python demo_pipeline.py --replay logs/room.jsonl --attack-start-h 56
    python demo_pipeline.py --make-sim logs/sim6d.jsonl                       (a simulated stand-in recording)
    python demo_pipeline.py --replay logs/sim6d.jsonl --attack-start-h 86

How much recording you need (device hours; the attack must start after calibration and sit in a stretch with nobody in the room):
- Full calibration: 48 h + a 4.5 h attack + about 3.5 h afterwards for release and recovery, so at least about 56 h.
- `--demo 2` (24 h calibration): at least about 33 h. On the simulator it gave 2 extra false alerts but no false quarantine.
- `--demo 4` (12 h calibration): not recommended. On the simulator it raised many false alerts and wrongly quarantined the LDR,
  because 12 h never shows the baseline a night or a weekend.
Quarantine needs 3 h of continuous suspicion, so the attack must last 3.2 h or more. The admin release waits 1.5 h after the
attack ends (`--release-after-h`) because the unexplained-motion counter needs about that long to drain; releasing sooner failed
probation in testing. The events are also in the ledger (`logs/pipeline_*_ledger.jsonl`, one JSON object per line), which is a
ready-made feed for a UI. `docs/sample_pipeline_output.txt` and `docs/sample_ledger.jsonl` show a complete run.

## Blockchain layer (`iotguard/chain.py`)
The event ledger is a SHA-256 hash chain. `chain.py` batches its records into blocks (Merkle root per block, small proof of work,
each block linking to the previous one) and keeps three replica copies that vote block by block (simple majority).
A lone node catches a naive edit; only the vote catches an attacker who rewrites a record and recomputes every hash.

    python -m iotguard.chain logs/pipeline_xxx_ledger.jsonl       build and verify the blocks
Stage 8b of `demo_pipeline.py` shows both attacks and the repair. Scope: a permissioned chain with a fixed set of 3 replicas
simulated in ONE process (no network, no signatures); the proof of work is token-sized. It reads the ledger and never affects detection.

## Simulation
    python -m iotguard.run --simulate normal --days 4 --quiet
    python -m iotguard.run --simulate pir_stuck_night --days 5
    python -m iotguard.run --simulate gradual_poison --days 7 --naive
The first 2 simulated days are calibration, so `--days` must be larger than 2.

## Recording / replaying real data
    python -m iotguard.record --list
    python -m iotguard.record --port <PORT> --out logs/room.jsonl      (never overwrites: use a new name each time)
    python -m iotguard.record --check logs/room.jsonl
    python -m iotguard.run --replay logs/room.jsonl
Opening the serial port resets the ESP32, so a few "bad lines" of boot text at the start are normal.

## Live hardware
Flash `firmware/esp32_sensor_hub/esp32_sensor_hub.ino` (Arduino IDE, "ESP32 Dev Module", 115200 baud,
Adafruit DHT library), then:
    python -m iotguard.run --port <PORT> --ledger logs/ledger.jsonl --state logs/state.npz
Add `--log logs/run.log` to keep a time-stamped text log. Ctrl+C or a kill signal stops cleanly: the state is saved and the ledger verified.
Stdin commands while running: `release <sensor>`, `recalibrate`. Close the Arduino Serial Monitor first: only one program can hold the port.

Wiring used (change the pins at the top of the sketch if yours differ): DHT11 data GPIO 27, PIR out GPIO 13,
reed switch GPIO 14 to GND (internal pull-up; door reads open when the circuit is open), LDR module GPIO 33.
- The LDR module in use has only a digital output (DO), so `LDR_DIGITAL = true` and the value is just 0 or 4095. With an analog
  output (AO) or a bare LDR plus divider, set it to false for real brightness values.
- HC-SR501: turn the time-delay pot to its minimum and leave sensitivity mid-way, otherwise the output stays high for minutes,
  which looks like a stuck PIR (limit: 300 s continuously high). Allow about 60 s after power-up before judging it.
- This DHT11 reports tenths of a degree, so the recorder's "not whole numbers" warning is a false alarm. The recorder's
  "LDR saturated" warning is expected with a digital-only LDR.

## Hardware watchdog
`iotguard/watchdog.py` reports (it never changes a detection decision): a stream gap, a DHT with no valid reading for 5 min,
an LDR value frozen at exactly one number for 10 min, or an LDR pinned at an end of the scale for 12 h. Reports appear in the
console, the log and the ledger (`sensor_fault`, `sensor_fault_cleared`, `stream_gap`). It cannot tell a dead PIR or door from an
empty room or a closed door, so it does not try. With a digital-only LDR the 12 h "pinned" report can fire on a long dark night.

## Experiments
    python -m iotguard.experiments compare

## Layout
- `iotguard/` pipeline, models, consensus, security, simulator, experiments, and the tests (`test_*.py`, `conftest.py`)
- `firmware/` ESP32 sketch
- `demo.py`, `plot_drift.py` demo runner and the drift figure
- `demo_pipeline.py`, `iotguard/inject.py` full-pipeline demo and the software attack injector
- `iotguard/chain.py` blocks, Merkle roots, replica voting
- `docs/` a sample console output and ledger from a complete run
- `logs/`, `figs/` created on first use

## Known limits
- The blockchain is a small permissioned simulation (replicas in one process, no signatures), not a public or distributed chain.
- Sudden, self-consistent poisoning and DHT-only drift are not caught.
- Light-only attacks are undetected as an attack: light evidence is deliberately weak, so an LDR-only attack slows learning
  but a sensor is never quarantined for it (only the PIR, door and temperature/humidity rules can quarantine).
- gradual_poison on seed 0 is detected late (cause not yet diagnosed); seeds 1 and 2 are detected cleanly.
- The watchdog thresholds are untested on real hardware.
- The demo preset (`presets.demo_setup`) is unvalidated.
- Quarantine recovery (`release <sensor>`) is typed while a run is live; `demo_pipeline.py` performs it in code at a fixed delay instead.
- A quarantined sensor is released into probation and recovers; it is not reset and relearned as a new sensor.
- Shortened calibration (`--demo`) is unvalidated: 12 h gave false alerts and a false LDR quarantine on the simulator.
- Real-room thresholds have not been tuned: everything was tuned on the simulator.
