"""Small helpers that turn a RunRecord into display data. No detection logic lives here."""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from .theme import fmt

SENSOR_FEATURES = {"dht": ("temp_mean", "hum_mean"), "pir": ("pir_count", "pir_duty"), "door": ("door_count",), "ldr": ("ldr_mean", "ldr_std")}
FEATURE_LABEL = {"temp_mean": "temperature °C", "hum_mean": "humidity %RH", "pir_count": "PIR events/window", "pir_duty": "PIR duty",
                 "door_count": "door transitions/window", "ldr_mean": "light ADC mean", "ldr_std": "light ADC std"}


def operating(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["phase"] == "OPERATING") & df["decision"].notna()]


def latest_values(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    last = df.iloc[-1]
    return {k: last[k] for k in FEATURE_LABEL}


def sensor_health_table(summary: dict, df: pd.DataFrame) -> pd.DataFrame:
    states, faults, last = summary.get("sensor_states", {}), summary.get("hardware_faults_active", {}), latest_values(df)
    rows = []
    for s, feats in SENSOR_FEATURES.items():
        fault = [f for f in faults if s in f.lower()]
        rows.append({"sensor": s.upper(), "quarantine state": states.get(s, "unknown"),
                     "watchdog fault": ", ".join(fault) if fault else "none reported",
                     "latest window values": "; ".join(f"{FEATURE_LABEL[f]}={fmt(last.get(f))}" for f in feats) if last else "—"})
    return pd.DataFrame(rows)


def parse_evidence(raw) -> list[tuple[str, Optional[float], str]]:
    try:
        return [(n, v, d) for n, v, d in json.loads(raw or "[]")]
    except (ValueError, TypeError):
        return []


def explain_window(row: pd.Series, threshold: Optional[float]) -> list[str]:
    """Plain-language facts about one window, all taken from recorded engine output. Not a verdict that an attack happened."""
    out = []
    if row.get("decision") is None or pd.isna(row.get("decision")):
        return [f"Phase {row.get('phase')}: the engine makes no TRUSTED/UNCERTAIN/SUSPICIOUS decision during calibration."]
    out.append(f"Engine decision: **{row['decision']}** — reason reported by the engine: _{row.get('reason')}_.")
    if pd.notna(row.get("score")) and threshold is not None:
        rel = "above" if row["score"] > threshold else "at or below"
        out.append(f"Mahalanobis distance D = {fmt(row['score'])}, which is {rel} the configured threshold {threshold:g}.")
    ev = parse_evidence(row.get("evidence"))
    neg = [e for e in ev if e[1] is not None and e[1] < 0]
    pos = [e for e in ev if e[1] is not None and e[1] > 0]
    if neg:
        out.append("Contradicting evidence: " + "; ".join(f"{n} ({d})" for n, _, d in neg))
    if pos:
        out.append("Corroborating evidence: " + "; ".join(f"{n} ({d})" for n, _, d in pos))
    if not ev:
        out.append("No cross-sensor evidence was recorded for this window.")
    if row.get("muted"):
        out.append(f"Sensors excluded from scoring (quarantined or on probation): {row['muted']}.")
    out.append(f"Baseline learning rate applied to this window: {fmt(row.get('alpha'), 4)} (0 = baseline frozen).")
    if row.get("truth") in ("attack", "legit_change", "normal"):
        out.append(f"Simulator ground truth for this window: **{row['truth']}**" + (f" ({row['attack_tags']})" if row.get("attack_tags") else "") + ".")
    return out
