#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    result = 0
    while True:
        if pos >= len(buf):
            raise EOFError("varint past end")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 64:
            raise ValueError("varint too long")


def _skip_field(buf: bytes, pos: int, wire: int) -> int:
    if wire == 0:
        _, pos = _read_varint(buf, pos)
        return pos
    if wire == 1:
        return pos + 8
    if wire == 2:
        l, pos = _read_varint(buf, pos)
        return pos + l
    if wire == 5:
        return pos + 4
    raise ValueError(f"unsupported wire type: {wire}")


def _parse_summary_value(msg: bytes) -> tuple[str | None, float | None]:
    """Parse a Summary.Value proto (only tag + simple_value)."""
    pos = 0
    tag: str | None = None
    simple_value: float | None = None
    while pos < len(msg):
        key, pos = _read_varint(msg, pos)
        field = key >> 3
        wire = key & 7
        if field == 1 and wire == 2:
            l, pos = _read_varint(msg, pos)
            tag = msg[pos : pos + l].decode("utf-8", "replace")
            pos += l
        elif field == 2 and wire == 5:
            simple_value = struct.unpack("<f", msg[pos : pos + 4])[0]
            pos += 4
        else:
            pos = _skip_field(msg, pos, wire)
    return tag, simple_value


def _parse_summary(msg: bytes) -> list[tuple[str, float]]:
    """Parse a Summary proto (only scalar values)."""
    pos = 0
    out: list[tuple[str, float]] = []
    while pos < len(msg):
        key, pos = _read_varint(msg, pos)
        field = key >> 3
        wire = key & 7
        if field == 1 and wire == 2:
            l, pos = _read_varint(msg, pos)
            tag, value = _parse_summary_value(msg[pos : pos + l])
            pos += l
            if tag is not None and value is not None:
                out.append((tag, value))
        else:
            pos = _skip_field(msg, pos, wire)
    return out


@dataclass(frozen=True)
class Event:
    step: int | None
    wall_time: float | None
    file_version: str | None
    scalars: list[tuple[str, float]]


def _parse_event(msg: bytes) -> Event:
    """Parse a (brain.)Event proto.

    Compatible with the IsaacLab/skrl events we observed where:
      - wall_time is field 1 (fixed64 / double)
      - step is field 2 (varint)
      - file_version can appear in field 3 (length-delimited string)
      - summary can appear in field 3 or field 5 (length-delimited)
    We accept both summary field numbers to be robust across proto variants.
    """
    pos = 0
    step: int | None = None
    wall_time: float | None = None
    file_version: str | None = None
    scalars: list[tuple[str, float]] = []

    while pos < len(msg):
        key, pos = _read_varint(msg, pos)
        field = key >> 3
        wire = key & 7

        if field == 1 and wire == 1:
            wall_time = struct.unpack("<d", msg[pos : pos + 8])[0]
            pos += 8
        elif field == 2 and wire == 0:
            step, pos = _read_varint(msg, pos)
        elif field in (3, 5) and wire == 2:
            l, pos = _read_varint(msg, pos)
            payload = msg[pos : pos + l]
            pos += l
            # Heuristic: if it decodes cleanly and contains "Event", treat as file version.
            # Otherwise, treat as Summary message.
            if field == 3:
                try:
                    decoded = payload.decode("utf-8")
                except Exception:
                    decoded = None
                if decoded and "Event" in decoded and len(decoded) < 64:
                    file_version = decoded
                    continue
            scalars.extend(_parse_summary(payload))
        else:
            pos = _skip_field(msg, pos, wire)

    return Event(step=step, wall_time=wall_time, file_version=file_version, scalars=scalars)


def _iter_tfrecord_events(path: Path) -> list[Event]:
    raw = path.read_bytes()
    pos = 0
    events: list[Event] = []

    while pos + 12 <= len(raw):
        length = struct.unpack("<Q", raw[pos : pos + 8])[0]
        pos += 8
        pos += 4  # length crc
        data = raw[pos : pos + length]
        pos += length
        pos += 4  # data crc
        events.append(_parse_event(data))

    return events


def _print_key_metrics(series: dict[str, list[tuple[int, float]]]) -> None:
    keys = [
        "Episode / Total timesteps (mean)",
        "Episode / Total timesteps (min)",
        "Episode / Total timesteps (max)",
        "Info / Episode_Reward/termination_penalty",
        "Reward / Total reward (mean)",
        "Loss / Value loss",
        "Loss / Policy loss",
        "Loss / Entropy loss",
        "Policy / Standard deviation",
    ]
    print("=== 关键指标（first -> last）===")
    for k in keys:
        s = series.get(k)
        if not s:
            continue
        first = s[0][1]
        last = s[-1][1]
        print(f"- {k}: {first:.6g} -> {last:.6g} (delta {last - first:+.6g})")


def _print_reward_breakdown_at_step(
    series_by_step: dict[int, dict[str, float]], *, step: int, top_n: int = 12
) -> None:
    d = series_by_step.get(step)
    if not d:
        print(f"[WARN] step={step} not found in event file")
        return

    terms = {k: v for k, v in d.items() if k.startswith("Info / Episode_Reward/")}
    total = sum(terms.values())
    print(f"=== Reward 分解（step={step}）===")
    print(f"- term 数量: {len(terms)}")
    print(f"- 求和: {total:.6g}")
    for k, v in sorted(terms.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]:
        print(f"  {k.split('/')[-1]:32s} {v: .6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TensorBoard tfevents (no TensorFlow dependency).")
    parser.add_argument("path", type=Path, help="Path to events.out.tfevents.*")
    parser.add_argument("--list-tags", action="store_true", help="List all scalar tags")
    parser.add_argument("--key-metrics", action="store_true", help="Print key metrics (first -> last)")
    parser.add_argument("--top-delta", type=int, default=0, help="Print N tags with largest abs(first-last)")
    parser.add_argument(
        "--breakdown",
        choices=["first", "last"],
        default=None,
        help="Print Info / Episode_Reward breakdown at first/last logged step",
    )
    args = parser.parse_args()

    events = _iter_tfrecord_events(args.path)
    if not events:
        raise SystemExit("[ERROR] No events found")

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    series_by_step: dict[int, dict[str, float]] = defaultdict(dict)
    versions = set()

    for e in events:
        if e.file_version:
            versions.add(e.file_version)
        if e.step is None:
            continue
        for tag, value in e.scalars:
            series[tag].append((e.step, value))
            series_by_step[e.step][tag] = value

    print(f"[INFO] file: {args.path}")
    if versions:
        print(f"[INFO] file_version: {sorted(versions)}")
    print(f"[INFO] records: {len(events)}")
    print(f"[INFO] scalar_tags: {len(series)}")

    if args.list_tags:
        print("=== Tags ===")
        for tag in sorted(series):
            print(f"- {tag}")

    if args.key_metrics or (not args.list_tags and not args.top_delta and not args.breakdown):
        _print_key_metrics(series)

    if args.top_delta:
        rows = []
        for tag, s in series.items():
            if len(s) < 2:
                continue
            rows.append((abs(s[-1][1] - s[0][1]), tag, s[0][1], s[-1][1]))
        print(f"=== abs(first-last) Top {args.top_delta} ===")
        for _, tag, first, last in sorted(rows, reverse=True)[: args.top_delta]:
            print(f"- {tag}: {first:.6g} -> {last:.6g} (delta {last - first:+.6g})")

    if args.breakdown:
        steps = sorted(series_by_step)
        if not steps:
            print("[WARN] No scalar steps found for breakdown")
        else:
            step = steps[0] if args.breakdown == "first" else steps[-1]
            _print_reward_breakdown_at_step(series_by_step, step=step)


if __name__ == "__main__":
    main()

