"""TLU counter ratios: single-run inspection, batch collection, and CSV input."""

import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import sys
import urllib.request

from .config import (
    ANALYSIS_DIR, DATA_DIR, RUNLIST_JSON, PREPOST_DIR,
    DEFAULT_INFLUX_DATABASE, DEFAULT_INFLUX_HOST, DEFAULT_INFLUX_PASSWORD,
    DEFAULT_INFLUX_PORT, DEFAULT_INFLUX_USER, DEFAULT_INFLUX_VERIFY_SSL,
    DEFAULT_POST_VETO_MEASUREMENT, DEFAULT_PRE_VETO_MEASUREMENT,
    DEFAULT_SECRET_FILE, DEFAULT_VETO_COUNTER_POST_RUN_SECONDS,
)
from .daq_conditions import PrePostVetoRatioReader
from .time_intervals import format_time


DEFAULT_RUN_CSV = DATA_DIR / "runs.csv"


DEFAULT_RUN_JSON = RUNLIST_JSON


DEFAULT_OUTPUT = PREPOST_DIR / "prepost_veto_ratios.csv"


FIELDNAMES = (
    "run_number",
    "configuration",
    "start_time",
    "stop_time",
    "pre_veto_triggers",
    "post_veto_triggers",
    "pre_minus_post_triggers",
    "post_over_pre_ratio",
    "pre_over_post_ratio",
    "veto_fraction",
    "pre_veto_sample_time",
    "post_veto_sample_time",
    "pre_veto_sample_source",
    "post_veto_sample_source",
    "status",
    "error",
)


DEFAULT_PREPOST_CSV = (
    ANALYSIS_DIR / "summary_runs_csv_by_config/prepost_veto_ratios.csv"
)


def _add_connection_arguments(parser):
    """Keep hardware-counter connection options identical for both commands."""
    parser.add_argument("--influx-host", default=DEFAULT_INFLUX_HOST)
    parser.add_argument("--influx-port", default=DEFAULT_INFLUX_PORT, type=int)
    parser.add_argument("--influx-user", default=DEFAULT_INFLUX_USER)
    parser.add_argument("--influx-password", default=DEFAULT_INFLUX_PASSWORD)
    parser.add_argument("--influx-database", default=DEFAULT_INFLUX_DATABASE)
    parser.add_argument("--secret-file", default=DEFAULT_SECRET_FILE)
    parser.add_argument("--pre-veto-measurement", default=DEFAULT_PRE_VETO_MEASUREMENT)
    parser.add_argument("--post-veto-measurement", default=DEFAULT_POST_VETO_MEASUREMENT)
    parser.add_argument("--post-run-seconds", type=float,
                        default=DEFAULT_VETO_COUNTER_POST_RUN_SECONDS)
    parser.add_argument("--influx-no-verify", action="store_false",
                        dest="influx_verify_ssl", default=DEFAULT_INFLUX_VERIFY_SSL)
    parser.add_argument("--influx-verify", action="store_true", dest="influx_verify_ssl")
    parser.add_argument("-v", "--verbose", action="store_true")


def parse_args(batch=False):
    parser = argparse.ArgumentParser(description=(
        "Collect run-final AHCAL PreVeto/PostVeto ratios for selected runs."
        if batch else "Inspect run-final AHCAL PreVeto/PostVeto counters and their ratios."
    ))
    if batch:
        parser.add_argument("--run-csv", type=Path, default=DEFAULT_RUN_CSV)
        parser.add_argument("--run-json", type=Path, default=DEFAULT_RUN_JSON)
        parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    else:
        parser.add_argument("--run", required=True, type=int, help="AHCAL run number")
        parser.add_argument("-o", "--output", help="optional output JSON path")
        parser.add_argument("--bin-seconds", type=float, default=10.0,
                            help="time-bin width for --csv/--plot (default: 10)")
        parser.add_argument("--csv", help="optional binned instantaneous-ratio CSV path")
        parser.add_argument("--plot", help="optional instantaneous-ratio SVG path")
    _add_connection_arguments(parser)
    args = parser.parse_args()
    if args.post_run_seconds < 0:
        parser.error("--post-run-seconds must be >= 0")
    if not batch and args.bin_seconds <= 0:
        parser.error("--bin-seconds must be > 0")
    return args


def _make_reader(args):
    return PrePostVetoRatioReader(
        host=args.influx_host, port=args.influx_port,
        username=args.influx_user, password=args.influx_password,
        database=args.influx_database, secret_file=args.secret_file,
        pre_veto_measurement=args.pre_veto_measurement,
        post_veto_measurement=args.post_veto_measurement,
        post_run_seconds=args.post_run_seconds,
        verify_ssl=args.influx_verify_ssl, verbose=args.verbose,
    )


RUNINFO_URL = "https://faser-runinfo.app.cern.ch/cgibin/getRunInfo.py"


def find_run_time(run_number):
    url = f"{RUNINFO_URL}?runno={int(run_number)}"
    with urllib.request.urlopen(url, timeout=20) as response:
        run_info = json.loads(response.read().decode("utf-8"))
    if not run_info:
        raise RuntimeError(f"run {run_number} was not found in RunInfo")
    if run_info.get("type") != "AHCAL":
        raise RuntimeError(
            f"run {run_number} is type {run_info.get('type', 'unknown')}, not AHCAL"
        )

    start_text = run_info.get("starttime")
    stop_text = run_info.get("stoptime")
    if not start_text or not stop_text:
        raise RuntimeError(f"run {run_number} has no complete start/stop time")

    def parse_utc(value):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()

    return parse_utc(start_text), parse_utc(stop_text)


def write_ratio_csv(path, rows):
    fieldnames = (
        "start_utime",
        "stop_utime",
        "pre_veto_delta",
        "post_veto_delta",
        "pre_over_post_ratio",
        "post_over_pre_ratio",
        "veto_fraction",
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ratio_svg(path, rows, run_number):
    width, height = 1000, 560
    left, right, top, bottom = 85, 30, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    finite_rows = [
        row for row in rows
        if isinstance(row["pre_over_post_ratio"], (int, float))
        and math.isfinite(row["pre_over_post_ratio"])
    ]
    origin = rows[0]["start_utime"] if rows else 0.0
    end = rows[-1]["stop_utime"] if rows else origin + 1.0
    duration = max(end - origin, 1.0)
    y_min = 1.0
    y_max = 1.02

    def x(timestamp):
        return left + (timestamp - origin) / duration * plot_width

    def y(value):
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    title = html.escape(f"Run {run_number} instantaneous PreVeto / PostVeto")
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<defs><clipPath id="plot-area"><rect x="{left}" y="{top}" '
        f'width="{plot_width}" height="{plot_height}"/></clipPath></defs>',
        f'<text x="{width/2}" y="30" text-anchor="middle" '
        f'font-family="sans-serif" font-size="20">{title}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        yy = top + plot_height * (1 - fraction)
        value = y_min + (y_max - y_min) * fraction
        elements.extend([
            f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" '
            f'y2="{yy:.2f}" stroke="#dddddd"/>',
            f'<text x="{left-10}" y="{yy+5:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="13">{value:.3f}</text>',
        ])
    for tick in range(6):
        fraction = tick / 5
        xx = left + plot_width * fraction
        elapsed = duration * fraction
        elements.extend([
            f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" '
            f'y2="{top+plot_height}" stroke="#eeeeee"/>',
            f'<text x="{xx:.2f}" y="{top+plot_height+24}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13">{elapsed:.3g}</text>',
        ])
    points = " ".join(
        f'{x((row["start_utime"]+row["stop_utime"])/2):.3f},'
        f'{y(row["pre_over_post_ratio"]):.3f}'
        for row in finite_rows
    )
    if points:
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="#2878b5" '
            'stroke-width="1.5" stroke-linejoin="round" '
            'clip-path="url(#plot-area)"/>'
        )
    elements.extend([
        f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" '
        f'y2="{top+plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top+plot_height}" stroke="black"/>',
        f'<text x="{left+plot_width/2}" y="{height-18}" text-anchor="middle" '
        'font-family="sans-serif" font-size="15">Time from first bin (s)</text>',
        f'<text x="22" y="{top+plot_height/2}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="15" '
        f'transform="rotate(-90 22 {top+plot_height/2})">'
        'PreVeto / PostVeto</text>',
        "</svg>",
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(elements) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    try:
        since, until = find_run_time(args.run)
        reader = _make_reader(args)
        summary = reader.run_summary(since, until)
        ratio_rows = (
            reader.ratio_series(since, until, args.bin_seconds)
            if args.csv or args.plot
            else None
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    result = {
        "run_number": args.run,
        "start_time": format_time(since),
        "stop_time": format_time(until),
        "prepost_veto_summary": summary,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)
            output_file.write("\n")
    if args.csv:
        write_ratio_csv(args.csv, ratio_rows)
    if args.plot:
        write_ratio_svg(args.plot, ratio_rows, args.run)

    post_over_pre = summary["post_over_pre_ratio"]
    pre_over_post = summary["pre_over_post_ratio"]
    veto_fraction = summary["veto_fraction"]
    print(f"run: {args.run}")
    print(f"PreVeto triggers: {summary['pre_veto_triggers']:,}")
    print(f"PostVeto triggers: {summary['post_veto_triggers']:,}")
    print(f"PreVeto - PostVeto: {summary['pre_minus_post_triggers']:,}")
    print(
        "PostVeto / PreVeto: "
        + (f"{post_over_pre:.10f}" if post_over_pre is not None else "undefined")
    )
    print(
        "PreVeto / PostVeto: "
        + (f"{pre_over_post:.10f}" if pre_over_post is not None else "undefined")
    )
    print(
        "veto fraction: "
        + (f"{veto_fraction:.10f}" if veto_fraction is not None else "undefined")
    )
    if args.output:
        print(f"wrote: {args.output}")
    if args.csv:
        print(f"wrote: {args.csv}")
    if args.plot:
        print(f"wrote: {args.plot}")
    return 0


def run_numbers_from_csv(path):
    """Read positive run numbers from the first CSV column."""
    runs = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or not row[0].strip():
                continue
            value = row[0].strip()
            if line_number == 1 and value.lower() in {"run", "run_number"}:
                continue
            try:
                run = int(value)
            except ValueError as error:
                raise ValueError(
                    f"invalid run number {value!r} in {path}:{line_number}"
                ) from error
            if run <= 0:
                raise ValueError(
                    f"run number must be positive in {path}:{line_number}"
                )
            runs.append(run)
    if not runs:
        raise ValueError(f"no run numbers found in {path}")
    # A run may intentionally appear more than once with different elapsed
    # windows in a true-rate selection CSV.  Its run-final counter ratio is
    # common to every window, so query it only once while preserving order.
    return list(dict.fromkeys(runs))


def selected_run_info(run_csv, run_json):
    """Return JSON run records in the order given by the CSV."""
    requested = run_numbers_from_csv(run_csv)
    with Path(run_json).open(encoding="utf-8") as stream:
        all_runs = json.load(stream)
    by_run = {int(row["run_number"]): row for row in all_runs}
    missing = [run for run in requested if run not in by_run]
    if missing:
        raise ValueError(
            "run(s) missing from run JSON: " + ", ".join(map(str, missing))
        )
    return [by_run[run] for run in requested]


def collect_rows(run_info, reader, progress=None):
    """Read final counters for each run, retaining failures as CSV rows."""
    rows = []
    total = len(run_info)
    for index, info in enumerate(run_info, 1):
        run = int(info["run_number"])
        row = {field: "" for field in FIELDNAMES}
        row.update(
            run_number=run,
            configuration=info.get("configuration", ""),
            start_time=info.get("start_time", ""),
            stop_time=info.get("stop_time", ""),
        )
        try:
            summary = reader.run_summary(
                float(info["start_utime"]), float(info["stop_utime"])
            )
            for field in FIELDNAMES:
                if field in summary:
                    row[field] = summary[field]
            ratio = summary.get("pre_over_post_ratio")
            if ratio is None or not math.isfinite(ratio) or ratio < 1:
                raise ValueError("PreVeto/PostVeto ratio is undefined or invalid")
            row["status"] = "ok"
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            row["status"] = "error"
            row["error"] = str(error)
        rows.append(row)
        if progress is not None:
            progress(index, total, row)
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def collect_main():
    args = parse_args(batch=True)
    run_info = selected_run_info(args.run_csv, args.run_json)
    reader = _make_reader(args)

    def report(index, total, row):
        detail = ""
        if row["status"] == "ok":
            detail = f" Post/Pre={row['post_over_pre_ratio']:.8f}"
        else:
            detail = f" {row['error']}"
        print(
            f"[{index:02d}/{total:02d}] run {row['run_number']}: "
            f"{row['status']}{detail}",
            flush=True,
        )

    rows = collect_rows(run_info, reader, progress=report)
    write_csv(args.output, rows)
    successes = sum(row["status"] == "ok" for row in rows)
    print(f"Wrote {len(rows)} runs ({successes} ok) to {args.output}")
    return 0 if successes == len(rows) else 1


def read_pre_over_post_ratio(run_number, path=DEFAULT_PREPOST_CSV):
    """Return PreVeto/PostVeto for one run, or raise if unavailable."""
    matches = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["run_number"]) != int(run_number):
                continue
            if row.get("status") not in (None, "", "ok"):
                raise ValueError(f"run {run_number} has no valid PreVeto/PostVeto counters")
            ratio = float(row["pre_over_post_ratio"])
            if not math.isfinite(ratio) or ratio < 1.0:
                raise ValueError(f"invalid PreVeto/PostVeto ratio for run {run_number}")
            matches.append(ratio)
    if len(matches) != 1:
        raise ValueError(f"expected one PreVeto/PostVeto ratio for run {run_number}; found {len(matches)}")
    return matches[0]


def selected_ratios(runs, explicit_paths, output, secret_file, allow_missing=False):
    """Use measured ratios; fetch only missing runs using their full run times."""
    from .add_stable_luminosity import (
        DEFAULT_RATIOS, ADDITIONAL_RATIOS, EXTRA_RATIOS, REMAINING_RATIOS, read_ratios,
    )

    paths = explicit_paths if explicit_paths is not None else [p for p in (
        output, DEFAULT_RATIOS, ADDITIONAL_RATIOS, EXTRA_RATIOS, REMAINING_RATIOS,
    ) if p.exists()]
    ratios = read_ratios(paths)
    missing = [run for run in runs if run["run_number"] not in ratios]
    fetchable = [run for run in missing if run.get("run_times_from_csv", False)]
    if fetchable and explicit_paths is None:
        reader = PrePostVetoRatioReader(
            host=DEFAULT_INFLUX_HOST, port=DEFAULT_INFLUX_PORT,
            username=DEFAULT_INFLUX_USER, password=DEFAULT_INFLUX_PASSWORD,
            database=DEFAULT_INFLUX_DATABASE, secret_file=secret_file,
            pre_veto_measurement=DEFAULT_PRE_VETO_MEASUREMENT,
            post_veto_measurement=DEFAULT_POST_VETO_MEASUREMENT,
            post_run_seconds=DEFAULT_VETO_COUNTER_POST_RUN_SECONDS,
            verify_ssl=DEFAULT_INFLUX_VERIFY_SSL,
        )
        # Final counters use the full run boundaries retained in the CSV.
        rows = collect_rows(fetchable, reader)
        fetched = output.with_name("prepost_veto_fetch.csv")
        write_csv(fetched, rows)
        ratios.update(read_ratios([fetched]))
    unresolved = [run["run_number"] for run in runs if run["run_number"] not in ratios]
    if unresolved and not allow_missing:
        raise ValueError(f"missing valid PreVeto/PostVeto ratios for runs {unresolved}; supply --prepost-ratio-csv")
    from .interval_csv import write_table
    write_table(output, [{"run_number": run["run_number"],
                          "pre_over_post_ratio": ratios.get(run["run_number"], ""),
                          "status": "ok" if run["run_number"] in ratios else "error"}
                         for run in runs], ("run_number", "pre_over_post_ratio", "status"))
    if unresolved:
        print(f"PreVeto/PostVeto is unavailable for {unresolved}; effective luminosity/true rates will be incomplete")
    return ratios
