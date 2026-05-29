#!/usr/bin/env python3
"""Download AHCAL run metadata from FASER runinfo."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_URL = "https://faser-runinfo.app.cern.ch/cgibin/getRunList.py"


def fetch_run_list(url: str) -> pd.DataFrame:
    import pandas as pd

    runs = pd.read_json(url, convert_dates=False)
    required_columns = {"runnumber", "type", "configName", "starttime", "stoptime"}
    missing_columns = required_columns - set(runs.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing expected column(s) from runinfo: {missing}")
    return runs


def extract_runs(
    runs: pd.DataFrame,
    run_type: str,
    include_ongoing: bool,
) -> list[dict[str, object]]:
    selected = runs[runs["type"].astype(str).str.strip() == run_type].copy()
    selected = selected.dropna(subset=["runnumber", "configName", "starttime", "stoptime"])

    if not include_ongoing:
        selected = selected[selected["starttime"] != selected["stoptime"]]

    selected = selected.rename(
        columns={
            "runnumber": "run_number",
            "configName": "configuration",
            "starttime": "start_time",
            "stoptime": "stop_time",
        }
    )
    selected = selected[["run_number", "configuration", "start_time", "stop_time"]]
    selected["run_number"] = selected["run_number"].astype(int)
    selected = selected.sort_values("run_number")

    return selected.to_dict(orient="records")


def write_json(path: str, runs: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as output:
        json.dump(runs, output, indent=2, sort_keys=True)
        output.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch FASER runinfo and write AHCAL run number/configuration/"
            "start/stop times to a JSON file."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ahcal_run_times.json",
        help="output JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="runinfo JSON endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--type",
        default="AHCAL",
        dest="run_type",
        help="exact run type to select (default: %(default)s)",
    )
    parser.add_argument(
        "--include-ongoing",
        action="store_true",
        help="include runs whose start time equals stop time",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        run_list = fetch_run_list(args.url)
        ahcal_runs = extract_runs(run_list, args.run_type, args.include_ongoing)
        write_json(args.output, ahcal_runs)
    except (HTTPError, URLError, TimeoutError, ImportError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"wrote {len(ahcal_runs)} {args.run_type} runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())