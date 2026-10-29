#!/usr/bin/env python3
"""Extract AHCAL runs and stable intervals from RunInfo and LHC BeamData."""

import argparse
import json
from urllib.request import urlopen

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_runlist import config
from stable_runlist.get_ahcal_run_times import DEFAULT_URL
from stable_runlist.interval_csv import write_intervals
from stable_runlist.json_io import write_runs
from stable_runlist.lhc_stable import LHCStableReader
from stable_runlist.time_intervals import format_time, interval_entry, parse_time

def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract AHCAL runs and STABLE BEAMS intervals to CSV.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--input-json", type=Path, help="existing run metadata instead of downloading RunInfo")
    parser.add_argument("--db", default=config.DEFAULT_DB, help="LHC BeamData SQLite database")
    parser.add_argument("--run", type=int, action="append", help="only this run; repeatable")
    parser.add_argument("--output-dir", type=Path, default=config.RUNLIST_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.input_json:
        payload = json.loads(args.input_json.read_text())
    else:
        with urlopen(args.url, timeout=60) as response:
            payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("expected a RunInfo JSON list or an existing AHCAL run-metadata JSON list")
    runs = []
    for row in payload:
        # Existing metadata is already selected as AHCAL. Remote rows carry type.
        if "type" in row and str(row["type"]).strip() != "AHCAL":
            continue
        number = int(row.get("run_number", row.get("runnumber", 0)))
        if args.run and number not in args.run:
            continue
        start = row.get("start_time", row.get("starttime"))
        stop = row.get("stop_time", row.get("stoptime"))
        if not start or not stop:
            continue
        since, until = parse_time(start), parse_time(stop)
        if since >= until:
            continue
        runs.append({"run_number": number,
                     "configuration": row.get("configuration", row.get("configName", "")),
                     "start_time": format_time(since), "stop_time": format_time(until),
                     "start_utime": since, "stop_utime": until})
    runs.sort(key=lambda x: x["run_number"])
    missing = set(args.run or []) - {r["run_number"] for r in runs}
    if missing:
        raise ValueError(f"requested AHCAL runs missing or incomplete: {sorted(missing)}")
    if not runs:
        raise ValueError("no completed AHCAL runs selected")
    reader = LHCStableReader(args.db, verbose=args.verbose)
    try:
        for run in runs:
            run["stable_list"] = reader.stable_intervals(run["start_utime"], run["stop_utime"])
    finally:
        reader.close()
    write_intervals(args.output_dir / "ahcal_runs.csv", [dict(
        run, run_intervals=[interval_entry(run["start_utime"], run["stop_utime"])],
    ) for run in runs], "run_intervals")
    write_intervals(args.output_dir / "stable_intervals.csv", runs, "stable_list")
    write_runs(args.output_dir / "ahcal_stable_intervals.json", runs)
    print(f"Wrote {len(runs)} AHCAL runs and {sum(len(r['stable_list']) for r in runs)} stable intervals to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
