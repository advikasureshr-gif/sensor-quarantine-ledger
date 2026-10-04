"""
contract.py - the ONE data contract between every input source and the pipeline.

The simulator, the ESP32 (over serial) and recorded sessions (replay) all
produce `Reading` objects. Nothing downstream may depend on where a reading
came from.

Wire format: one JSON object per line (JSON Lines), compact, e.g.

    {"t":12.5,"temp":27.4,"hum":61.0,"pir":1,"door":0,"ldr":1873}

Fields
------
t      float   Seconds on the SOURCE's own monotonic clock (ESP32: millis()/1000.0).
               Never wall-clock time. The pipeline closes windows by comparing
               reading timestamps, so live, fast-forward and replay all behave alike.
temp   float|null  Degrees C. null when the DHT read failed.
hum    float|null  Percent RH. null when the DHT read failed.
pir    0|1     1 = motion currently detected.
door   0|1     1 = door OPEN (magnet apart from reed switch), 0 = closed.
ldr    int     Raw 12-bit ADC value, 0..4095. Higher = brighter or darker depending
               on wiring; the pipeline only cares about relative change.

Optional ground-truth fields (simulator / manually marked demo runs ONLY;
real hardware never sends them, and the pipeline must never read them):

label        "normal" | "legit_change" | "attack"
attack_kind  free-text tag such as "gradual_poison"; only allowed when label == "attack"

Design notes
------------
* This module checks STRUCTURE and TYPES, not physical plausibility. A DHT that
  reports 85 C is a valid line; deciding it is garbage is preprocessing's job.
* Unknown extra keys are ignored, so firmware can add fields without breaking us.
* Malformed lines raise ContractError. Sources catch it, count the bad line, and
  carry on (serial noise such as boot messages is normal).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional

ADC_MAX = 4095  # ESP32 12-bit ADC
LABELS = ("normal", "legit_change", "attack")

_REQUIRED_KEYS = ("t", "temp", "hum", "pir", "door", "ldr")


class ContractError(ValueError):
    """A reading or line violates the data contract."""


# --------------------------------------------------------------------------
# field validators
# --------------------------------------------------------------------------
def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite_float(v: Any, name: str) -> float:
    if not _is_number(v):
        raise ContractError(f"{name} must be a number, got {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise ContractError(f"{name} must be finite, got {v!r}")
    return f


def _optional_float(v: Any, name: str) -> Optional[float]:
    """Number or None. NaN counts as 'no reading' and becomes None."""
    if v is None:
        return None
    if _is_number(v) and math.isnan(v):
        return None
    return _finite_float(v, name)


def _binary(v: Any, name: str) -> int:
    # Accept 0/1 and False/True (bool is an int subclass), nothing else.
    if isinstance(v, (bool, int)) and v in (0, 1):
        return int(v)
    raise ContractError(f"{name} must be 0 or 1, got {v!r}")


def _adc(v: Any, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ContractError(f"{name} must be an integer, got {v!r}")
    if not 0 <= v <= ADC_MAX:
        raise ContractError(f"{name} must be in 0..{ADC_MAX}, got {v!r}")
    return v


# --------------------------------------------------------------------------
# the reading
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Reading:
    t: float
    temp: Optional[float]
    hum: Optional[float]
    pir: int
    door: int
    ldr: int
    label: Optional[str] = None
    attack_kind: Optional[str] = None

    def __post_init__(self) -> None:
        # Frozen dataclass: normalise via object.__setattr__.
        t = _finite_float(self.t, "t")
        if t < 0:
            raise ContractError(f"t must be >= 0, got {t!r}")
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "temp", _optional_float(self.temp, "temp"))
        object.__setattr__(self, "hum", _optional_float(self.hum, "hum"))
        object.__setattr__(self, "pir", _binary(self.pir, "pir"))
        object.__setattr__(self, "door", _binary(self.door, "door"))
        object.__setattr__(self, "ldr", _adc(self.ldr, "ldr"))

        if self.label is not None and self.label not in LABELS:
            raise ContractError(f"label must be one of {LABELS} or null, got {self.label!r}")
        if self.attack_kind is not None:
            if self.label != "attack":
                raise ContractError("attack_kind is only allowed when label == 'attack'")
            if not isinstance(self.attack_kind, str) or not self.attack_kind:
                raise ContractError("attack_kind must be a non-empty string")

    # ---- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "t": self.t,
            "temp": self.temp,
            "hum": self.hum,
            "pir": self.pir,
            "door": self.door,
            "ldr": self.ldr,
        }
        if self.label is not None:
            d["label"] = self.label
        if self.attack_kind is not None:
            d["attack_kind"] = self.attack_kind
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "Reading":
        if not isinstance(d, dict):
            raise ContractError(f"expected a JSON object, got {type(d).__name__}")
        missing = [k for k in _REQUIRED_KEYS if k not in d]
        if missing:
            raise ContractError(f"missing keys: {', '.join(missing)}")
        return cls(
            t=d["t"],
            temp=d["temp"],
            hum=d["hum"],
            pir=d["pir"],
            door=d["door"],
            ldr=d["ldr"],
            label=d.get("label"),
            attack_kind=d.get("attack_kind"),
        )

    @property
    def dht_ok(self) -> bool:
        """True when the DHT produced both values for this reading."""
        return self.temp is not None and self.hum is not None


# --------------------------------------------------------------------------
# line <-> reading
# --------------------------------------------------------------------------
def format_line(reading: Reading) -> str:
    """Reading -> one compact JSON line (no trailing newline)."""
    return json.dumps(reading.to_dict(), separators=(",", ":"), allow_nan=False)


def parse_line(line: str) -> Reading:
    """One line of text -> Reading. Raises ContractError on anything malformed."""
    s = line.strip()
    if not s:
        raise ContractError("empty line")
    if not s.startswith("{"):
        # ESP32 boot chatter ("ets Jun  8 2016 ...", "rst:0x1 ...") lands here.
        raise ContractError(f"not a JSON object line: {s[:40]!r}")
    try:
        # parse_constant: the tokens NaN / Infinity / -Infinity become None
        # ("no reading") instead of sneaking in as floats.
        obj = json.loads(s, parse_constant=lambda _tok: None)
    except json.JSONDecodeError as e:
        raise ContractError(f"invalid JSON ({e.msg})") from e
    return Reading.from_dict(obj)