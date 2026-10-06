"""
record.py - record the live ESP32 stream to a log file, and check that the hardware is behaving.

    python -m iotguard.record --list                            show serial ports
    python -m iotguard.record --port COM3 --out logs/s1.jsonl   record (Ctrl+C to stop)
    python -m iotguard.record --check logs/s1.jsonl             sanity report on a log

While recording, type a note and press Enter to drop a marker into the log ("door opened", "lamp on",
"left room"...). Markers are '#' comment lines, so ReplaySource ignores them, but they say what was happening
when, which is exactly what you need later to tune thresholds and fit the simulator. Each marker records the
device clock, so it lines up with the readings.

The log holds one JSON line per reading with the DEVICE's own clock (not this computer's), exactly the
format ReplaySource reads. A status line shows the live sensor values every couple of seconds, so you can
see immediately whether each sensor responds. At the end, a sanity report points at likely wiring or
configuration problems.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, Optional, Sequence

from .contract import ADC_MAX, Reading, format_line
from .sources import ReplaySource, SerialSource, list_serial_ports


# ---------------------------------------------------------------------------
# sanity report
# ---------------------------------------------------------------------------
def sanity_report(readings: Sequence[Reading], dht: str = "DHT11", expected_period_s: float = 0.5) -> list[tuple[str, str]]:
    """
    (level, message) findings about a recording; level is 'ok', 'info', 'warn' or 'error'.
    These are hints for a person standing next to the hardware, not proof of a fault: a short test
    that never opens the door will naturally "fail" the door check.
    """
    out: list[tuple[str, str]] = []
    n = len(readings)
    if n < 20:
        return [("error", f"only {n} readings: is the board connected and printing JSON lines?")]
    t = [r.t for r in readings]
    dts = [b - a for a, b in zip(t, t[1:])]
    dur = t[-1] - t[0]
    med = median(dts)
    out.append(("info", f"{n} readings over {dur / 60:.1f} min of device time (median spacing {med:.2f} s)"))
    if abs(med - expected_period_s) > 0.1:
        out.append(("warn", f"readings arrive every {med:.2f} s, not {expected_period_s:g} s: check SAMPLE_PERIOD_MS in the firmware "
                            f"(the pipeline assumes {expected_period_s:g} s)"))
    resets = sum(1 for d in dts if d < -5.0)
    if resets:
        out.append(("warn", f"the device clock went backwards {resets} time(s): the board restarted (power glitch, brown-out or watchdog)"))
    gaps = [d for d in dts if d > 5.0]
    if gaps:
        out.append(("warn", f"{len(gaps)} gap(s) longer than 5 s in the data (longest {max(gaps):.0f} s): dropped serial data or a stalled loop"))

    # --- DHT
    temps = [r.temp for r in readings if r.temp is not None]
    nulls = 1.0 - len(temps) / n
    if not temps:
        out.append(("error", "the DHT never produced a reading (temp/hum always null): check the wiring on GPIO 4, 3.3 V and ground, "
                             "the 10k pull-up on a bare 4-pin sensor, and that DHT_TYPE matches the sensor"))
    else:
        if nulls > 0.2:
            out.append(("warn", f"{nulls:.0%} of readings have no DHT value: a loose connection, a missing pull-up or reading too fast"))
        else:
            out.append(("ok", f"DHT works ({nulls:.1%} null)"))
        whole = sum(1 for v in temps if abs(v - round(v)) < 1e-9) / len(temps)
        if dht == "DHT11" and whole < 0.99:
            out.append(("warn", "temperatures are not whole numbers, but a DHT11 only reports whole degrees: is DHT_TYPE set to DHT22?"))
        if dht == "DHT22" and whole > 0.99 and len(temps) > 200:
            out.append(("warn", "temperatures are all whole numbers, which is typical of a DHT11: is DHT_TYPE set to DHT22 by mistake?"))
        lo, hi = min(temps), max(temps)
        hums = [r.hum for r in readings if r.hum is not None]
        if lo < -5 or hi > 60 or (hums and (min(hums) < 0 or max(hums) > 100)):
            out.append(("warn", f"implausible DHT values (temp {lo:.0f}..{hi:.0f} C): check the sensor type and wiring"))

    # --- PIR
    duty = sum(r.pir for r in readings) / n
    edges = sum(1 for a, b in zip(readings, readings[1:]) if a.pir == 0 and b.pir == 1)
    if edges == 0 and duty == 0 and dur > 120:
        out.append(("warn", "the PIR never triggered: wave a hand in front of it AFTER its 60 s warm-up; check OUT on GPIO 27, VCC on VIN (5 V)"))
    elif duty > 0.9:
        out.append(("warn", f"the PIR is high {duty:.0%} of the time: still warming up, sensitivity turned too high, or the pin is floating"))
    else:
        out.append(("ok" if edges else "info", f"PIR: {edges} trigger(s), high {duty:.1%} of the time"))

    # --- door
    changes = sum(1 for a, b in zip(readings, readings[1:]) if a.door != b.door)
    open_frac = sum(r.door for r in readings) / n
    if changes == 0:
        if open_frac > 0.95:
            out.append(("warn", "the door reads OPEN the whole time: if the door is closed, the polarity is probably reversed "
                                "(change DOOR_OPEN_LEVEL to LOW in the firmware)"))
        elif dur > 120:
            out.append(("info", "the door state never changed: open and close it (move the magnet) to test the reed switch"))
    else:
        out.append(("ok", f"door: {changes} change(s), open {open_frac:.1%} of the time"
                          + (" -- check polarity: it is open most of the time" if open_frac > 0.5 else "")))

    # --- LDR
    ldr = [r.ldr for r in readings]
    lo, hi = min(ldr), max(ldr)
    sat = sum(1 for v in ldr if v <= 5 or v >= ADC_MAX - 5) / n
    if hi - lo < 30 and dur > 60:
        out.append(("warn", f"the LDR barely changes ({lo}..{hi}): cover it with your hand or shine a light at it; check the wiring on GPIO 34"))
    elif sat > 0.5:
        out.append(("warn", f"the LDR is saturated (at 0 or {ADC_MAX}) {sat:.0%} of the time: wrong divider resistor or a floating pin"))
    else:
        out.append(("ok", f"LDR range {lo}..{hi}" + (" (also check that covering it lowers/raises the value as you expect)" if hi - lo > 30 else "")))
    return out


def format_report(findings: Iterable[tuple[str, str]]) -> str:
    tag = {"ok": "[ ok ]", "info": "[info]", "warn": "[WARN]", "error": "[FAIL]"}
    return "\n".join(f"{tag[lvl]} {msg}" for lvl, msg in findings)


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------
class Recorder:
    """Writes readings (device clock) and markers to a log file. Use `handle` as SerialSource's on_line."""

    def __init__(self, path, *, port: str = "", baud: Optional[int] = None, note: str = "",
                 status_every_s: float = 2.0, echo: Callable[[str], None] = print, clock=time.monotonic) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            raise FileExistsError(f"{self.path} already exists; choose another name (recordings are never overwritten)")
        self._f = self.path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self.echo, self._clock = echo, clock
        self.status_every_s, self._last_status = status_every_s, clock()
        self.readings: list[Reading] = []
        self.bad_lines = 0
        self.markers = 0
        self._last_t: Optional[float] = None
        self._last_flush = clock()
        self._write(f"# iotguard recording, started {datetime.now().isoformat(timespec='seconds')}")
        if port:
            self._write(f"# port={port} baud={baud}")
        if note:
            self._write(f"# NOTE {note}")

    def _write(self, line: str) -> None:
        with self._lock:
            self._f.write(line + "\n")
            now = self._clock()
            if now - self._last_flush >= 1.0:
                self._f.flush()
                self._last_flush = now

    def handle(self, text: str, reading: Optional[Reading]) -> None:
        """Called for every complete line from the device."""
        if reading is None:
            if text and not text.startswith("#"):
                self.bad_lines += 1
            return
        self._write(format_line(reading))
        self.readings.append(reading)
        self._last_t = reading.t
        now = self._clock()
        if now - self._last_status >= self.status_every_s:
            self._last_status = now
            r = reading
            temp = "--" if r.temp is None else f"{r.temp:g}"
            hum = "--" if r.hum is None else f"{r.hum:g}"
            self.echo(f"t={r.t:8.1f}s  temp {temp:>4}  hum {hum:>4}  PIR {r.pir}  door {'OPEN' if r.door else 'closed':<6}  LDR {r.ldr:>4}"
                      f"   ({len(self.readings)} readings, {self.bad_lines} bad lines, {self.markers} markers)")

    def mark(self, text: str) -> None:
        """Insert a marker (a '#' comment) stamped with the device clock and this computer's time."""
        text = " ".join(text.split()) or "mark"
        t = "?" if self._last_t is None else f"{self._last_t:.1f}"
        self._write(f"# MARK t={t} wall={datetime.now().isoformat(timespec='seconds')} | {text}")
        self.markers += 1
        self.echo(f"  marker saved at device t={t}s: {text}")

    def close(self) -> None:
        with self._lock:
            self._f.flush()
            self._f.close()

    def summary(self, dht: str = "DHT11") -> str:
        lines = [f"saved {len(self.readings)} readings and {self.markers} marker(s) to {self.path}"]
        lines.append(format_report(sanity_report(self.readings, dht)))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _main(argv: Optional[list[str]] = None, *, opener=None, input_lines: Optional[Iterable[str]] = None,
          echo: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(prog="python -m iotguard.record", description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--check", metavar="LOG", help="sanity report on an existing log and exit")
    ap.add_argument("--port", help="serial port, e.g. COM3 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", help="log file to create")
    ap.add_argument("--note", default="", help="free text stored at the top of the log")
    ap.add_argument("--minutes", type=float, default=None, help="stop automatically after this long")
    ap.add_argument("--dht", default="DHT11", choices=["DHT11", "DHT22"], help="sensor type, for the sanity report")
    a = ap.parse_args(argv)

    if a.list:
        ports = list_serial_ports()
        echo("\n".join(f"{dev}  {desc}" for dev, desc in ports) if ports else "no serial ports found")
        return 0
    if a.check:
        src = ReplaySource(a.check)
        readings = list(src)
        echo(f"{a.check}: {src.stats.summary()}")
        echo(format_report(sanity_report(readings, a.dht)))
        return 0
    if not a.port or not a.out:
        ap.error("recording needs --port and --out (or use --list / --check)")

    rec = Recorder(a.out, port=a.port, baud=a.baud, note=a.note, echo=echo)
    src = SerialSource(a.port, a.baud, opener=opener, on_line=rec.handle, on_event=lambda m: echo(f"[serial] {m}"))

    def read_notes() -> None:
        try:
            for line in (input_lines if input_lines is not None else sys.stdin):
                rec.mark(line.strip())
        except Exception:
            pass
    threading.Thread(target=read_notes, daemon=True).start()
    if a.minutes:
        threading.Timer(a.minutes * 60.0, src.stop).start()
    echo(f"recording to {a.out} (Ctrl+C to stop). Type a note + Enter to add a marker, e.g. 'door opened'.")
    try:
        for _ in src:
            pass
    except KeyboardInterrupt:
        pass
    except (OSError, ImportError) as e:
        echo(f"stopped: {e}")
    finally:
        src.close()
        rec.close()
    echo(rec.summary(a.dht))
    return 0


if __name__ == "__main__":
    sys.exit(_main())