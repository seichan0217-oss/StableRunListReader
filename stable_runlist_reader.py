#!/usr/bin/env python3
"""CLI for building AHCAL stable/good intervals."""

import argparse
import sys

from config import (
    DEFAULT_DB,
    DEFAULT_INFLUX_DATABASE,
    DEFAULT_INFLUX_HOST,
    DEFAULT_INFLUX_PASSWORD,
    DEFAULT_INFLUX_PORT,
    DEFAULT_INFLUX_USER,
    DEFAULT_INPUT,
    DEFAULT_LUMI_ACCT_TAG,
    DEFAULT_LUMI_TAG,
    DEFAULT_SECRET_FILE,
)
from reader import StableRunListReader


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read ahcal_run_times.json, attach LHC STABLE BEAMS intervals, "
            "DAQ/luminosity/config excluded intervals, and AHCAL good intervals."
        )
    )
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="input JSON path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output JSON path; defaults to overwriting the input",
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="LHC BeamData COOL sqlite DB")
    parser.add_argument(
        "--run-config-filter-regex",
        default="",
        help="only process runs whose configuration matches this regex; empty disables",
    )
    parser.add_argument(
        "--allowed-config-regex",
        "--physics-config-regex",
        default="AHCALPhysics",
        dest="allowed_config_regex",
        help="configs not matching this regex are excluded over the stable-beam interval",
    )
    parser.add_argument("--no-lumi", action="store_true", help="do not query ATLAS luminosity DB")
    parser.add_argument(
        "--lumi-tag",
        default=DEFAULT_LUMI_TAG,
        help="ATLAS offline luminosity tag used to check missing luminosity",
    )
    parser.add_argument(
        "--lumi-acct-tag",
        default=DEFAULT_LUMI_ACCT_TAG,
        help=(
            "use /TRIGGER/OFLLUMI/LumiAccounting with this tag instead of "
            "OflPrefLumi/LBTIME, matching faser_luminosity.py --acct"
        ),
    )
    parser.add_argument("--no-daq", action="store_true", help="do not query InfluxDB")
    parser.add_argument("--influx-host", default=DEFAULT_INFLUX_HOST)
    parser.add_argument("--influx-port", default=DEFAULT_INFLUX_PORT, type=int)
    parser.add_argument("--influx-user", default=DEFAULT_INFLUX_USER)
    parser.add_argument("--influx-password", default=DEFAULT_INFLUX_PASSWORD)
    parser.add_argument("--influx-database", default=DEFAULT_INFLUX_DATABASE)
    parser.add_argument(
        "--influx-no-verify",
        action="store_false",
        dest="influx_verify_ssl",
        default=True,
        help="disable HTTPS certificate verification for InfluxDB",
    )
    parser.add_argument("--secret-file", default=DEFAULT_SECRET_FILE)
    parser.add_argument("--daq-bin-seconds", default=1.0, type=float)
    parser.add_argument(
        "--daq-max-gap-seconds",
        default=30.0,
        type=float,
        help="DAQ counter gaps larger than this are excluded",
    )
    parser.add_argument(
        "--required-measurement",
        action="append",
        default=None,
        help="InfluxDB counter measurement required during stable beams; repeatable",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        output = StableRunListReader(args).run()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    total_good = sum(run.get("good_time_sec", 0.0) for run in output)
    print(
        f"wrote {len(output)} runs to {args.output or args.input} "
        f"(total AHCAL good time: {total_good:.1f} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
