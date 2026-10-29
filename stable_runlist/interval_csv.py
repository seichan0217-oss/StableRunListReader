"""The interval CSV exchanged by the four independent workflow scripts.

One row describes one half-open UTC interval of a run. There is deliberately
no stable/good switch: the rows in the input file define the selection.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

from .time_intervals import format_time, interval_entry, parse_time


FIELDS = (
    "run_number", "configuration", "run_start_time", "run_stop_time",
    "lhc_fill", "start_time", "stop_time", "start_utime", "stop_utime", "reason",
)


def write_table(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_intervals(path, runs, key):
    rows = []
    for run in runs:
        for interval in run.get(key, []):
            rows.append(dict(
                interval, run_number=run["run_number"],
                configuration=run.get("configuration", ""),
                run_start_time=run["start_time"], run_stop_time=run["stop_time"],
            ))
    write_table(path, rows, FIELDS)


def _time(row, prefix):
    numeric = row.get(prefix + "_utime", "")
    text = row.get(prefix + "_time", "")
    value = float(numeric) if numeric else parse_time(text)
    if not math.isfinite(value):
        raise ValueError(f"{prefix} time must be finite")
    if text and abs(parse_time(text) - value) > 1e-5:
        raise ValueError(f"{prefix}_time and {prefix}_utime disagree")
    return value


def read_intervals(path, run_numbers=None):
    """Return run records, using stable_list only as the existing core's adapter."""
    path = Path(path)
    by_run = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if "run_number" not in fields or any(
            not ({prefix + "_utime", prefix + "_time"} & fields)
            for prefix in ("start", "stop")
        ):
            raise ValueError(f"{path}: expected run_number and start/stop time columns")
        for line, row in enumerate(reader, 2):
            try:
                number = int(row["run_number"])
                start, stop = _time(row, "start"), _time(row, "stop")
                if number <= 0 or start >= stop:
                    raise ValueError("require a positive run number and start < stop")
                interval = interval_entry(start, stop)
                if row.get("lhc_fill"):
                    interval["lhc_fill"] = int(row["lhc_fill"])
                if row.get("reason"):
                    interval["reason"] = row["reason"]
                info = by_run.setdefault(number, {
                    "run_number": number, "configuration": row.get("configuration", ""),
                    "start_time": row.get("run_start_time", ""),
                    "stop_time": row.get("run_stop_time", ""),
                    "run_times_from_csv": bool(row.get("run_start_time") and row.get("run_stop_time")),
                    "stable_list": [],
                })
                for key, column in (("configuration", "configuration"),
                                    ("start_time", "run_start_time"),
                                    ("stop_time", "run_stop_time")):
                    if info[key] != row.get(column, ""):
                        raise ValueError(f"inconsistent {column} for run {number}")
                info["stable_list"].append(interval)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error

    for number, run in by_run.items():
        intervals = sorted(run["stable_list"], key=lambda x: x["start_utime"])
        for previous, current in zip(intervals, intervals[1:]):
            if current["start_utime"] < previous["stop_utime"]:
                raise ValueError(f"{path}: overlapping intervals for run {number}")
        run["stable_list"] = intervals
        start = parse_time(run["start_time"]) if run["start_time"] else intervals[0]["start_utime"]
        stop = parse_time(run["stop_time"]) if run["stop_time"] else intervals[-1]["stop_utime"]
        if start > intervals[0]["start_utime"] or stop < intervals[-1]["stop_utime"]:
            raise ValueError(f"{path}: interval outside run {number} start/stop")
        run.update(start_time=format_time(start), stop_time=format_time(stop),
                   start_utime=start, stop_utime=stop,
                   stable_time_sec=sum(x["stop_utime"] - x["start_utime"] for x in intervals))

    requested = list(dict.fromkeys(run_numbers or []))
    if requested:
        missing = [n for n in requested if n not in by_run]
        if missing:
            raise ValueError(f"{path}: requested run(s) have no intervals: {missing}")
        return [by_run[n] for n in requested]
    return list(by_run.values())


def selection_id(runs):
    """Keep products and caches for different CSV selections separate."""
    content = [{k: run[k] for k in (
        "run_number", "configuration", "start_time", "stop_time", "stable_list",
    )} for run in sorted(runs, key=lambda x: x["run_number"])]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]


def selection_directory(base, runs):
    return Path(base) / selection_id(runs)


def prepare_output(directory, runs, input_path):
    """An explicit output directory must not silently mix two interval selections."""
    from .add_stable_luminosity import write_json

    directory = Path(directory)
    manifest = directory / "selection.json"
    identity = selection_id(runs)
    if manifest.exists() and json.loads(manifest.read_text())["selection_id"] != identity:
        raise ValueError(f"{directory} belongs to a different CSV selection; use another --output-dir")
    if directory.exists() and not manifest.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} is nonempty without a selection manifest; use another --output-dir")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(manifest, {"selection_id": identity, "input_csv": str(input_path.resolve()), "runs": runs})
    write_intervals(directory / "selected_intervals.csv", runs, "stable_list")
