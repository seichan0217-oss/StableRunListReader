#!/usr/bin/env python3
"""Apply automatic exclusions to an interval CSV and write the good intervals."""

import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_runlist import config
from stable_runlist.daq_conditions import DAQConditionReader
from stable_runlist.interval_csv import read_intervals, write_intervals
from stable_runlist.json_io import write_runs
from stable_runlist.luminosity_gaps import LuminosityGapReader
from stable_runlist.raw_daq_conditions import RawDAQConditionReader
from stable_runlist.time_intervals import interval_entry, merge_intervals, subtract_intervals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="stage 1 interval CSV")
    parser.add_argument("--run", type=int, action="append", help="only this run in the CSV; repeatable")
    parser.add_argument("--output-dir", type=Path, default=config.RUNLIST_DIR)
    parser.add_argument("--no-lumi", action="store_true")
    parser.add_argument("--no-daq", action="store_true")
    parser.add_argument("--daq-source", choices=("influx", "raw"), default="influx")
    parser.add_argument("--raw-pattern", action="append", default=[])
    parser.add_argument("--daq-bin-seconds", type=float, default=config.DEFAULT_DAQ_BIN_SECONDS)
    parser.add_argument("--daq-max-gap-seconds", type=float, default=config.DEFAULT_DAQ_MAX_GAP_SECONDS)
    parser.add_argument("--allowed-config-regex", default="AHCALPhysics")
    parser.add_argument("--lumi-tag", default=config.DEFAULT_LUMI_TAG)
    parser.add_argument("--lumi-acct-tag", default=config.DEFAULT_LUMI_ACCT_TAG)
    parser.add_argument("--secret-file", default=config.DEFAULT_SECRET_FILE)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.no_daq and args.daq_source == "raw" and not args.raw_pattern:
        parser.error("--daq-source raw requires --raw-pattern")
    if any(not math.isfinite(n) or n <= 0 for n in (args.daq_bin_seconds, args.daq_max_gap_seconds)):
        parser.error("DAQ bin and gap durations must be positive and finite")

    # Keep the intervals supplied in the CSV. Do not query stable beams again.
    runs = read_intervals(args.input, args.run)
    if not runs:
        raise ValueError("input CSV contains no intervals")
    allowed_config = re.compile(args.allowed_config_regex) if args.allowed_config_regex else None
    enriched = []

    with ExitStack() as resources:
        daq_reader = None
        if not args.no_daq:
            if args.daq_source == "raw":
                daq_reader = RawDAQConditionReader(
                    raw_patterns=args.raw_pattern,
                    bin_seconds=args.daq_bin_seconds,
                    max_gap_seconds=args.daq_max_gap_seconds,
                    verbose=args.verbose,
                )
            else:
                daq_reader = DAQConditionReader(
                    host=config.DEFAULT_INFLUX_HOST, port=config.DEFAULT_INFLUX_PORT,
                    username=config.DEFAULT_INFLUX_USER, password=config.DEFAULT_INFLUX_PASSWORD,
                    database=config.DEFAULT_INFLUX_DATABASE, secret_file=args.secret_file,
                    bin_seconds=args.daq_bin_seconds, max_gap_seconds=args.daq_max_gap_seconds,
                    required_measurements=list(config.DEFAULT_REQUIRED_MEASUREMENTS),
                    verify_ssl=config.DEFAULT_INFLUX_VERIFY_SSL, verbose=args.verbose,
                )
        lumi_reader = None
        if not args.no_lumi:
            lumi_reader = LuminosityGapReader(args.lumi_tag, args.lumi_acct_tag, verbose=args.verbose)
            resources.callback(lumi_reader.close)

        for run in runs:
            stable = run["stable_list"]
            excluded = []
            if allowed_config and not allowed_config.search(run["configuration"]):
                excluded.extend(interval_entry(
                    item["start_utime"], item["stop_utime"], f"Run config {run['configuration']}",
                ) for item in stable)
            if lumi_reader is not None:
                excluded.extend(lumi_reader.excluded_intervals(stable))
            if daq_reader is not None:
                if isinstance(daq_reader, RawDAQConditionReader):
                    daq_reader.prepare_run(run["run_number"], run["start_utime"], run["stop_utime"])
                excluded.extend(daq_reader.excluded_intervals(stable))

            excluded = merge_intervals(excluded)
            good = subtract_intervals(stable, excluded)
            result = dict(
                run, excluded_list=excluded, good_list=good,
                stable_time_sec=sum(i["stop_utime"] - i["start_utime"] for i in stable),
                excluded_time_sec=sum(i["stop_utime"] - i["start_utime"] for i in excluded),
                good_time_sec=sum(i["stop_utime"] - i["start_utime"] for i in good),
            )
            if isinstance(daq_reader, RawDAQConditionReader):
                result["raw_trigger_summary"] = daq_reader.run_summary()
            enriched.append(result)
            if args.verbose:
                print(f"run {run['run_number']}: stable {result['stable_time_sec']:.1f} s, good {result['good_time_sec']:.1f} s")

    # Keep the existing check for a failed luminosity connection.
    if not args.no_lumi and sum(r["good_time_sec"] for r in enriched) == 0 and all(
        r["excluded_list"] and all(
            "Missing ATLAS luminosity" in item.get("reason", "") or
            "Missing ATLAS LBTIME" in item.get("reason", "")
            for item in r["excluded_list"]
        ) for r in enriched
    ):
        raise RuntimeError("ATLAS luminosity lookup excluded every interval; check COOL access or use --no-lumi")

    write_runs(args.output_dir / "ahcal_good_intervals.json", enriched)
    write_intervals(args.output_dir / "good_intervals.csv", enriched, "good_list")
    write_intervals(args.output_dir / "excluded_intervals.csv", enriched, "excluded_list")
    print(f"Wrote good_intervals.csv and excluded_intervals.csv to {args.output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
