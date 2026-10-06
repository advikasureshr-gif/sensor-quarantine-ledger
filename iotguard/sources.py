"""
sources.py - where Readings come from.

The pipeline only ever sees an iterator of `Reading` objects. This module
provides the concrete producers of that iterator:

    IterSource    wraps any iterable of Readings (the simulator, tests)
    ReplaySource  reads a recorded .jsonl / serial-monitor log from disk

(A SerialSource for the live ESP32 will be added once the hardware is confirmed.
It will subclass Source the same way, so nothing downstream changes.)

What the base class guarantees to the pipeline
----------------------------------------------
1. Every item is a valid `Reading` (bad lines are counted, never yielded).
2. `t` never goes backwards. Real logs break this in two ways, handled here:
     * ESP32 reboot  -> t restarts near 0. Detected (a regression bigger than
       `reset_gap_s`) and the clock is continued from where it left off, so
       windowing sees one continuous timeline.
     * duplicate / jittered line -> a small regression. The line is dropped.
3. Counts of everything that was skipped are in `source.stats`, so a silent
   data-quality problem shows up as a number instead of mysteriously
   bad results.

Accepted line shapes (so you can paste straight from a serial monitor):
    {"t":1.0,"temp":27.4,"hum":61.0,"pir":0,"door":0,"ldr":1873}
    12:34:56.789 -> {"t":1.0,...}        (Arduino IDE 2 timestamp prefix)
    # esp32_sensor_hub ready             (comment, skipped silently)
    <blank line>                         (skipped silently)
Anything else is counted as a bad line (boot chatter, truncated lines, ...).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from .contract import ContractError, Reading, format_line, parse_line

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
@dataclass
class SourceStats:
    lines: int = 0            # raw lines examined (text-based sources only)
    readings: int = 0         # Readings yielded to the pipeline
    comments: int = 0         # '#' lines and blanks, skipped on purpose
    bad_lines: int = 0        # unparseable / contract-violating lines
    out_of_order: int = 0     # dropped: t went (slightly) backwards
    time_resets: int = 0      # clock restarts (e.g. ESP32 reboot) stitched over
    reconnects: int = 0       # live sources: times the connection was re-established
    first_errors: list[str] = field(default_factory=list)  # first few messages, for debugging

    MAX_ERRORS_KEPT = 5

    def note_bad(self, lineno: int, msg: str) -> None:
        self.bad_lines += 1
        if len(self.first_errors) < self.MAX_ERRORS_KEPT:
            self.first_errors.append(f"line {lineno}: {msg}")

    def summary(self) -> str:
        parts = [f"{self.readings} readings"]
        if self.lines:
            parts.append(f"{self.lines} lines")
        if self.bad_lines:
            parts.append(f"{self.bad_lines} bad")
        if self.out_of_order:
            parts.append(f"{self.out_of_order} out-of-order dropped")
        if self.time_resets:
            parts.append(f"{self.time_resets} clock reset(s) stitched")
        if self.reconnects:
            parts.append(f"{self.reconnects} reconnect(s)")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------
class Source(ABC):
    """
    Subclasses implement `_raw()` and yield Readings as they find them.
    Callers iterate the Source itself; the base class applies the time
    guarantees on top.
    """

    def __init__(self, reset_gap_s: float = 5.0, sample_period_s: float = 0.5) -> None:
        # A backwards jump of more than `reset_gap_s` is a reboot/clock reset;
        # a smaller one is jitter or a duplicate. 5 s = ten missed samples at 2 Hz,
        # comfortably larger than any realistic reordering.
        self.reset_gap_s = reset_gap_s
        self.sample_period_s = sample_period_s   # spacing used when stitching a reset
        self.stats = SourceStats()

    @abstractmethod
    def _raw(self) -> Iterator[Reading]:
        """Yield readings in arrival order. Do not enforce time rules here."""

    def __iter__(self) -> Iterator[Reading]:
        self.stats = SourceStats()  # fresh counts per pass
        last_t: Optional[float] = None   # last t in the OUTPUT timeline
        offset = 0.0                      # added to raw t after reset stitching
        last_raw_t: Optional[float] = None

        for r in self._raw():
            raw_t = r.t
            if last_raw_t is not None and raw_t < last_raw_t:
                if last_raw_t - raw_t > self.reset_gap_s:
                    # Clock restarted: continue the timeline one sample after the last output time.
                    offset = (last_t if last_t is not None else 0.0) + self.sample_period_s - raw_t
                    self.stats.time_resets += 1
                else:
                    self.stats.out_of_order += 1
                    continue
            last_raw_t = raw_t

            out = r if offset == 0.0 else replace(r, t=raw_t + offset)
            last_t = out.t
            self.stats.readings += 1
            yield out

    def _parse_text(self, text: str, lineno: int) -> Optional[Reading]:
        """
        One line of device text -> Reading, or None. Blank lines and '#' comments are skipped silently;
        anything unparseable is counted (and its message kept for debugging). Used by text-based sources.
        """
        s = text.strip()
        if not s or s.startswith("#"):
            self.stats.comments += 1
            return None
        try:
            return parse_line(_extract_json(s))
        except ContractError as e:
            self.stats.note_bad(lineno, str(e))
            return None

    # -- lifecycle (no-ops for sources holding no resources) -----------------
    def close(self) -> None:
        pass

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# IterSource: any iterable of Readings
# ---------------------------------------------------------------------------
class IterSource(Source):
    """Wraps a list/generator of Readings. Used by the simulator and by tests."""

    def __init__(self, readings: Iterable[Reading], **kw) -> None:
        super().__init__(**kw)
        self._readings = readings

    def _raw(self) -> Iterator[Reading]:
        yield from self._readings


# ---------------------------------------------------------------------------
# ReplaySource: recorded log on disk
# ---------------------------------------------------------------------------
def _extract_json(line: str) -> str:
    """Drop a serial-monitor timestamp prefix ("12:34:56.789 -> {...}") if present."""
    s = line.strip()
    if s and not s.startswith("{"):
        i = s.find("{")
        if i > 0 and s.endswith("}"):
            return s[i:]
    return s


class ReplaySource(Source):
    """
    Replays a recorded log through the pipeline.

    realtime=None   as fast as the pipeline can consume (default; experiments)
    realtime=1.0    paced like the original capture (live-feeling demo)
    realtime=10.0   ten times faster than the original capture

    Pacing uses the readings' own `t`, so gaps in the recording are preserved.
    """

    def __init__(self, path: PathLike, realtime: Optional[float] = None, **kw) -> None:
        super().__init__(**kw)
        if realtime is not None and realtime <= 0:
            raise ValueError("realtime must be > 0 or None")
        self.path = Path(path)
        self.realtime = realtime
        if not self.path.is_file():
            raise FileNotFoundError(f"log file not found: {self.path}")

    def _raw(self) -> Iterator[Reading]:
        t0_wall = time.monotonic()
        t0_log: Optional[float] = None
        # errors="replace": serial captures sometimes contain stray bytes at boot.
        with self.path.open("r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                self.stats.lines += 1
                s = line.strip()
                if not s or s.startswith("#"):
                    self.stats.comments += 1
                    continue
                try:
                    r = parse_line(_extract_json(s))
                except ContractError as e:
                    self.stats.note_bad(lineno, str(e))
                    continue

                if self.realtime is not None:
                    if t0_log is None:
                        t0_log = r.t
                    due = t0_wall + (r.t - t0_log) / self.realtime
                    delay = due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                yield r


# ---------------------------------------------------------------------------
# SerialSource: the live ESP32 over USB
# ---------------------------------------------------------------------------
def list_serial_ports() -> list[tuple[str, str]]:
    """(device, description) for every serial port the computer can see. Needs pyserial."""
    try:
        from serial.tools import list_ports
    except ImportError:
        raise ImportError("pyserial is required for live input: pip install pyserial") from None
    return [(p.device, p.description) for p in list_ports.comports()]


def _open_pyserial(port: str, baud: int, timeout: float):
    try:
        import serial
    except ImportError:
        raise ImportError("pyserial is required for live input: pip install pyserial") from None
    # serial_for_url accepts plain names ("COM3", "/dev/ttyUSB0") and test URLs such as "loop://"
    ser = serial.serial_for_url(port, baudrate=baud, timeout=timeout)
    try:
        ser.reset_input_buffer()          # drop anything buffered before we opened the port
    except Exception:
        pass
    return ser


class SerialSource(Source):
    """
    Live readings from the ESP32's USB serial port (one JSON line per reading, see contract.py).

    * Reassembles lines that arrive in pieces; ignores boot messages and garbage (counted as bad lines);
      accepts '#' comments, CRLF endings and serial-monitor timestamp prefixes.
    * Survives the board resetting (its clock restarts; the base class stitches the timeline) and a
      cable being unplugged: it reconnects every `reconnect_wait_s` (forever by default; set
      `max_reconnects` to give up). No data for `max_idle_s` also counts as a lost connection.
    * `on_line(text, reading_or_None)` is called for every complete line, which is how the log recorder
      sees exactly what the device sent. `on_event(message)` reports connect/disconnect.
    * `stop()` ends iteration cleanly (within about `read_timeout_s`).
    Time comes from the device (`t` = millis()/1000), never from this computer's clock.
    """

    def __init__(self, port: str, baud: int = 115200, *, read_timeout_s: float = 1.0, max_idle_s: float = 20.0,
                 reconnect: bool = True, reconnect_wait_s: float = 2.0, max_reconnects: Optional[int] = None,
                 opener=None, on_line=None, on_event=None, **kw) -> None:
        super().__init__(**kw)
        self.port, self.baud = port, baud
        self.read_timeout_s, self.max_idle_s = read_timeout_s, max_idle_s
        self.reconnect, self.reconnect_wait_s, self.max_reconnects = reconnect, reconnect_wait_s, max_reconnects
        self._opener = opener or (lambda: _open_pyserial(self.port, self.baud, self.read_timeout_s))
        self.on_line, self.on_event = on_line, on_event
        self._stop = False
        self._ser = None

    def stop(self) -> None:
        self._stop = True

    def _event(self, msg: str) -> None:
        if self.on_event:
            self.on_event(msg)

    def close(self) -> None:
        self._stop = True
        ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _raw(self) -> Iterator[Reading]:
        self._stop = False
        attempts = 0                              # consecutive failed (re)connections
        lineno = 0
        while not self._stop:
            try:
                self._ser = ser = self._opener()
            except ImportError:
                raise                             # missing driver: retrying cannot help
            except Exception as e:
                attempts += 1
                self._event(f"cannot open {self.port}: {e}")
                if not self.reconnect or (self.max_reconnects is not None and attempts > self.max_reconnects):
                    raise
                self._sleep(self.reconnect_wait_s)
                continue
            self._event(f"connected to {self.port}")
            if attempts or self.stats.lines:
                self.stats.reconnects += 1
            buf = b""
            last_data = time.monotonic()
            healthy = False                       # has this connection delivered anything yet?
            try:
                while not self._stop:
                    chunk = ser.readline()
                    now = time.monotonic()
                    if not chunk:                                    # read timed out
                        if now - last_data > self.max_idle_s:
                            raise TimeoutError(f"no data for {self.max_idle_s:g} s")
                        continue
                    last_data = now
                    if not healthy:                                  # data arrived: the connection works, so forget past failures
                        healthy, attempts = True, 0
                    if not chunk.endswith(b"\n"):                    # a piece of a line: keep it
                        buf += chunk
                        continue
                    data, buf = buf + chunk, b""
                    lineno += 1
                    self.stats.lines += 1
                    text = data.decode("utf-8", errors="replace").strip()
                    r = self._parse_text(text, lineno)
                    if self.on_line:
                        self.on_line(text, r)
                    if r is not None:
                        yield r
            except OSError as e:                                     # unplugged, reset, idle timeout
                self._event(f"lost connection: {e}")
                if not self.reconnect:
                    raise
            finally:
                try:
                    ser.close()
                except Exception:
                    pass
                self._ser = None
            if not self._stop:
                attempts += 1
                if self.max_reconnects is not None and attempts > self.max_reconnects:
                    raise ConnectionError(f"gave up on {self.port} after {self.max_reconnects} reconnect attempts")
                self._sleep(self.reconnect_wait_s)

    def _sleep(self, seconds: float) -> None:
        """Sleep in small steps so stop() is honoured promptly."""
        end = time.monotonic() + seconds
        while not self._stop and time.monotonic() < end:
            time.sleep(min(0.05, max(0.0, end - time.monotonic())))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def write_jsonl(readings: Iterable[Reading], path: PathLike, include_labels: bool = True) -> int:
    """
    Save readings as a .jsonl log that ReplaySource can read back. Returns the count.
    Use this to record simulator runs (reproducible experiments) or live sessions.
    """
    n = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in readings:
            if not include_labels and (r.label is not None or r.attack_kind is not None):
                r = replace(r, label=None, attack_kind=None)
            f.write(format_line(r) + "\n")
            n += 1
    return n


def strip_labels(readings: Iterable[Reading]) -> Iterator[Reading]:
    """
    Yield readings with ground truth removed. The pipeline is fed THROUGH this,
    so it is structurally unable to peek at labels; the evaluator keeps its own
    labelled copy of the stream for scoring.
    """
    for r in readings:
        if r.label is None and r.attack_kind is None:
            yield r
        else:
            yield replace(r, label=None, attack_kind=None)