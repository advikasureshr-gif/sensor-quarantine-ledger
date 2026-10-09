# online-behavioural-anomaly

ESP32 + DHT11 + HC-SR501 PIR + magnetic reed (door) + LDR: consensus-gated online behavioural
anomaly detection. A Mahalanobis baseline only learns from windows that other sensors corroborate
(TRUSTED -> full update, UNCERTAIN -> slow update, SUSPICIOUS -> freeze), so slow attacks cannot
poison it, while genuine changes in the room are still learned.

All validation so far is on the built-in simulator. Nothing has been validated on real hardware yet.

## Setup
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Quick checks (run from this folder)
    python3 -m pytest tests -q
    python3 -m iotguard.scenarios list

## Simulation
    python3 -m iotguard.run --simulate normal --days 4 --quiet
    python3 -m iotguard.run --simulate pir_stuck_night --days 5
    python3 -m iotguard.run --simulate gradual_poison --days 7 --naive

## Recording / replaying real data
    python3 -m iotguard.record --list
    python3 -m iotguard.record --port <PORT> --out logs/room.jsonl
    python3 -m iotguard.record --check logs/room.jsonl
    python3 -m iotguard.run --replay logs/room.jsonl

## Live hardware
Flash `firmware/esp32_sensor_hub/esp32_sensor_hub.ino` (Arduino IDE, "ESP32 Dev Module", 115200 baud,
Adafruit DHT library), then:
    python3 -m iotguard.run --port <PORT> --ledger logs/ledger.jsonl --state logs/state.npz
Add `--log logs/run.log` to keep a time-stamped text log. Ctrl+C or a kill signal stops cleanly: the state is saved and the ledger verified.
Stdin commands while running: `release <sensor>`, `recalibrate`.

## Hardware watchdog
`iotguard/watchdog.py` reports (it never changes a detection decision): a stream gap, a DHT with no valid reading for 5 min,
an LDR value frozen at exactly one number for 10 min, or an LDR pinned at an end of the scale for 12 h. Reports appear in the
console, the log and the ledger (`sensor_fault`, `sensor_fault_cleared`, `stream_gap`). It cannot tell a dead PIR or door from an
empty room or a closed door, so it does not try.

## Experiments
    python3 -m iotguard.experiments compare

## Layout
- `iotguard/` pipeline, models, consensus, security, simulator, experiments
- `firmware/` ESP32 sketch (not yet compiled or flashed)
- `tests/` pytest suite

## Known limits
- Sudden, self-consistent poisoning and DHT-only drift are not caught.
- Light-only attacks are undetected.
- gradual_poison on seed 0 is detected late (cause not yet diagnosed).
- The watchdog thresholds are untested on real hardware.
- The demo preset (`presets.demo_setup`) is unvalidated.