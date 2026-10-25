"""
merge_recordings.py - join several recordings into one continuous timeline.

    python merge_recordings.py logs\\room_oct24.jsonl logs\\room_oct24b.jsonl --out logs\\room_merged.jsonl

Each recording restarts its clock at about 0 (the ESP32 resets when the port opens). This shifts every later file so it
continues 0.5 s after the previous one ended, i.e. the real-world pause between recordings is REMOVED. That is fine for
a hardware check or a replay demo, but it is not one unbroken real recording: say so if you present it.
Never overwrites an existing output file.
"""
import argparse
import json
import sys
from pathlib import Path

STEP = 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f"{out} already exists; pick a new name")
        return 2
    offset, total, notes = 0.0, 0, []
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# merged from {len(a.files)} recording(s): {', '.join(Path(p).name for p in a.files)}\n")
        for p in a.files:
            first = last = None
            for line in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    r = json.loads(line)
                    t = float(r["t"])
                except (ValueError, KeyError, TypeError):
                    continue
                first = t if first is None else first
                last = t
                r["t"] = round(offset + (t - first), 3)
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
                total += 1
            if first is None:
                print(f"no readings in {p}")
                return 1
            offset += (last - first) + STEP
            notes.append(f"{Path(p).name}: {(last - first) / 3600:.2f} h")
    print(f"wrote {total} readings ({offset / 3600:.2f} h) to {out}\n  " + "\n  ".join(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())