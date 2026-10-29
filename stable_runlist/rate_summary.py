#!/usr/bin/env python3
"""Collect or refresh stable-run rate summaries and their shared deadtime model."""

from .config import OUTPUT_DIR, RUNLIST_JSON, RAW_RATE_DIR, FAIR_SCRIPTS_DIR

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .trigger_rate_correction import correction_scenarios, poisson_rate_summary

HERE = OUTPUT_DIR
DEFAULT_JSON = RUNLIST_JSON
DEFAULT_BATCH = HERE / "deadtime_trigger_rate" / "batch"
DEFAULT_OUT = HERE / "deadtime_trigger_rate" / "summary"
DEFAULT_PREPOST = (
    HERE / "deadtime_trigger_rate" / "summary_runs_csv_by_config"
    / "prepost_veto_ratios.csv"
)
LUMI_DIR = FAIR_SCRIPTS_DIR
DEADTIME_INTERVAL_COUNT = 4
TRUE_RATE_DEADTIME_COUNT = 3
LUMINOSITY_FIELD = "luminosity_smoothed_ub_inv_s"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-json", type=Path, default=DEFAULT_JSON)
    p.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    p.add_argument(
        "--trigger-rate-dir",
        type=Path,
        help=(
            "optional directory containing raw-derived "
            "run_<run>_trigger_rate.csv files"
        ),
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--run", type=int, action="append", default=[], metavar="RUN",
        help=(
            "include only this run; may be repeated (explicit selection "
            "ignores --limit)"
        ),
    )
    p.add_argument(
        "--run-csv", "--runs-csv", dest="run_csv", type=Path,
        help=(
            "read selected run numbers from column 1 of a CSV; column 2 may "
            "contain START:STOP elapsed seconds for the rate/luminosity "
            "means; an optional run_number header is accepted"
        ),
    )
    p.add_argument(
        "--true-rate-run-csv", type=Path,
        help=(
            "add runs from this CSV to the true-trigger-rate reconstruction "
            "without using their dead-time values to build the shared "
            "dead-time model; the optional elapsed-time column has the same "
            "START:STOP format as --run-csv"
        ),
    )
    p.add_argument("--bins", type=int, default=5000)
    p.add_argument(
        "--run-bins", action="append", default=[], metavar="RUN:BINS",
        help="override histogram bins for one run; may be repeated",
    )
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--mintimes", type=int, default=3)
    p.add_argument(
        "--min-events", "--minevents", dest="min_events", type=int, default=0,
        help="minimum number of stable deadtime histogram samples",
    )
    p.add_argument(
        "--run-rate-min-events", type=int, default=0,
        help=(
            "minimum number of stable deadtime histogram samples for the "
            "run-number-vs-mean-trigger-rate plot only (default: 0)"
        ),
    )
    p.add_argument(
        "--assumed-auto-triggered-ms", type=float, default=3.9,
        help="live window L used for the D1-D3 true-rate correction [ms]",
    )
    p.add_argument(
        "--prepost-ratio-csv",
        type=Path,
        default=DEFAULT_PREPOST,
        help=(
            "CSV containing per-run pre_over_post_ratio values; each run's "
            "ratio corrects R_reco before the deadtime inversion"
        ),
    )
    p.add_argument(
        "--additional-prepost-ratio-csv",
        type=Path,
        action="append",
        default=[],
        help=(
            "additional CSV containing per-run PreVeto/PostVeto ratios; "
            "may be repeated"
        ),
    )
    return p.parse_args()


def run_bin_overrides(values):
    result = {}
    for value in values:
        try:
            run_text, bins_text = value.split(":", 1)
            run, bins = int(run_text), int(bins_text)
        except ValueError as error:
            raise ValueError(f"invalid --run-bins {value!r}; expected RUN:BINS") from error
        if run <= 0 or bins <= 0:
            raise ValueError("--run-bins values must be positive")
        result[run] = bins
    return result


def parse_elapsed_range(value, path, line_number):
    """Parse START:STOP elapsed seconds from one run-CSV cell."""
    try:
        start_text, stop_text = value.split(":", 1)
        start, stop = float(start_text), float(stop_text)
    except ValueError as error:
        raise ValueError(
            f"invalid elapsed range {value!r} in {path}:{line_number}; "
            "expected START:STOP in seconds"
        ) from error
    if not all(math.isfinite(item) for item in (start, stop)):
        raise ValueError(
            f"elapsed range must be finite in {path}:{line_number}"
        )
    if start < 0.0 or stop <= start:
        raise ValueError(
            f"elapsed range must satisfy 0 <= START < STOP in "
            f"{path}:{line_number}"
        )
    return start, stop


def run_selection_rows_from_csv(path):
    """Read ordered (run, elapsed range) selections, retaining duplicates."""
    selections = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for line_number, row in enumerate(csv.reader(f), 1):
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
            elapsed_text = row[1].strip() if len(row) > 1 else ""
            elapsed_range = (
                parse_elapsed_range(elapsed_text, path, line_number)
                if elapsed_text else None
            )
            selections.append((run, elapsed_range))
    if not selections:
        raise ValueError(f"no run numbers found in {path}")
    return selections


def run_selections_from_csv(path):
    """Read runs and a legacy run-keyed view of their elapsed ranges."""
    selections = run_selection_rows_from_csv(path)
    runs = [run for run, _elapsed_range in selections]
    elapsed_ranges = {
        run: elapsed_range
        for run, elapsed_range in selections
        if elapsed_range is not None
    }
    return runs, elapsed_ranges


def run_numbers_from_csv(path):
    """Read positive run numbers from the first CSV column."""
    return run_selections_from_csv(path)[0]


def mean_pre_over_post_ratio(path, run_numbers):
    """Return the arithmetic mean PreVeto/PostVeto ratio for selected runs."""
    requested = set(map(int, run_numbers))
    values = []
    found = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                run = int(row["run_number"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid run_number in {path}") from error
            if run not in requested:
                continue
            if row.get("status") not in (None, "", "ok"):
                continue
            try:
                ratio = float(row["pre_over_post_ratio"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid pre_over_post_ratio for run {run} in {path}"
                ) from error
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(
                    f"non-positive or non-finite PreVeto/PostVeto ratio "
                    f"for run {run} in {path}"
                )
            values.append(ratio)
            found.add(run)
    missing = sorted(requested - found)
    if missing:
        raise ValueError(
            "selected run(s) have no valid PreVeto/PostVeto ratio: "
            + ", ".join(map(str, missing))
        )
    if not values:
        raise ValueError(f"no selected PreVeto/PostVeto ratios found in {path}")
    return float(np.mean(values)), len(values)


def pre_over_post_ratios(path, run_numbers):
    """Read one measured PreVeto/PostVeto correction per selected run."""
    requested = set(map(int, run_numbers))
    ratios = {}
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    for ratio_path in paths:
        with Path(ratio_path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                run = int(row["run_number"])
                if (
                    run not in requested
                    or row.get("status") not in (None, "", "ok")
                ):
                    continue
                ratio = float(row["pre_over_post_ratio"])
                if not math.isfinite(ratio) or ratio < 1.0:
                    raise ValueError(
                        f"invalid PreVeto/PostVeto ratio for run {run}"
                    )
                if run in ratios:
                    if not math.isclose(
                        ratios[run], ratio, rel_tol=1e-12, abs_tol=0.0
                    ):
                        raise ValueError(
                            "conflicting duplicate PreVeto/PostVeto ratio "
                            f"for run {run}: {ratios[run]} vs {ratio}"
                        )
                    continue
                ratios[run] = ratio
    missing = requested - ratios.keys()
    if missing:
        raise ValueError(
            "selected run(s) have no valid PreVeto/PostVeto ratio: "
            + ", ".join(map(str, sorted(missing)))
        )
    return ratios


def stable_runs(path, limit, mintimes=3, selected_runs=None):
    with path.open(encoding="utf-8") as f:
        all_rows = json.load(f)

    if selected_runs:
        requested = list(dict.fromkeys(selected_runs))
        if any(run <= 0 for run in requested):
            raise ValueError("--run values must be positive")
        by_run = {int(row["run_number"]): row for row in all_rows}
        missing = [run for run in requested if run not in by_run]
        if missing:
            raise ValueError(
                "requested run(s) not found in run JSON: "
                + ", ".join(map(str, missing))
            )
        rows = [by_run[run] for run in requested]
    else:
        rows = all_rows

    rows = [
        row for row in rows
        if row.get("stable_list")
        and float(row.get("stable_time_sec", 0.0)) >= mintimes
    ]

    if selected_runs:
        eligible = {int(row["run_number"]) for row in rows}
        rejected = [run for run in requested if run not in eligible]
        if rejected:
            raise ValueError(
                "requested run(s) have no qualifying stable interval: "
                + ", ".join(map(str, rejected))
            )
        return rows
    return rows[:limit] if limit else rows


def deadtimes(path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["interval_index"]))
    return rows[:DEADTIME_INTERVAL_COUNT]


def histogram_sample_count(path):
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", newline="") as f:
        return sum(int(r["count"]) for r in csv.DictReader(f))


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def rate_bins(job_dir, run, trigger_rate_dir=None):
    """Read raw-derived total/AnyVetoed/NonVetoed trigger-rate bins."""
    for path in (
        (
            trigger_rate_dir / f"run_{run}_trigger_rate.csv"
            if trigger_rate_dir is not None
            else Path("/__no_trigger_rate_override__")
        ),
        job_dir / f"run_{run}_trigger_rate.csv",
        job_dir.parent / f"run_{run}_bins_5000" / f"run_{run}_trigger_rate.csv",
        RAW_RATE_DIR / f"run{run}_trigger_rate.csv",
    ):
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as f:
                result = []
                for row in csv.DictReader(f):
                    if not row.get("rate_hz"):
                        continue
                    start = float(row["start_utime"])
                    stop = float(row["stop_utime"])

                    def optional(field):
                        value = row.get(field, "")
                        try:
                            return float(value) if value != "" else np.nan
                        except (TypeError, ValueError):
                            return np.nan

                    any_rate = optional("any_vetoed_rate_hz")
                    if not np.isfinite(any_rate):
                        any_rate = optional("smoothed_any_vetoed_rate_hz")
                    non_rate = optional("non_vetoed_rate_hz")
                    if not np.isfinite(non_rate):
                        non_rate = optional("smoothed_non_vetoed_rate_hz")
                    result.append(
                        {
                            "start": start,
                            "stop": stop,
                            "total_rate": float(row["rate_hz"]),
                            "any_vetoed_rate": any_rate,
                            "non_vetoed_rate": non_rate,
                            "prescale_factor": optional(
                                "any_vetoed_prescale_factor"
                            ),
                        }
                    )
            if result:
                return result, path
    # Do not fall back to InfluxDB Recorded_rate_Hz. The AnyVetoed fraction
    # must come from the TLU input bits in the raw files.
    return [], None


def rate_field_bins(rows, field):
    return [
        (row["start"], row["stop"], row[field])
        for row in rows
        if np.isfinite(row[field])
    ]


def luminosity_bins(job_dir, run):
    """Read the smoothed instantaneous luminosity time series for one run."""
    for path in (
        job_dir / f"run_{run}_timeseries.csv",
        job_dir.parent / f"run_{run}_bins_5000" / f"run_{run}_timeseries.csv",
        LUMI_DIR / f"run_{run}_timeseries.csv",
    ):
        if not path.is_file():
            continue
        samples = []
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    samples.append(
                        (utc(row["timestamp_utc"]), float(row[LUMINOSITY_FIELD]))
                    )
                except (KeyError, TypeError, ValueError):
                    pass
        samples = [value for value in samples if np.all(np.isfinite(value))]
        samples.sort()
        if samples:
            delta = np.diff([value[0] for value in samples])
            delta = delta[delta > 0]
            width = float(np.median(delta)) if len(delta) else 0.2
            return [(t, t + width, value) for t, value in samples], path
    return [], None


def stable_rate(bins, intervals):
    rates, weights = [], []
    for start, stop, rate in bins:
        overlap = sum(
            max(0.0, min(stop, b) - max(start, a)) for a, b in intervals
        )
        if overlap > 0 and np.isfinite(rate):
            rates.append(rate)
            weights.append(overlap)
    if not rates:
        return np.nan, np.nan, 0
    rates, weights = np.asarray(rates), np.asarray(weights)
    mean = float(np.average(rates, weights=weights))
    variance = float(np.average((rates - mean) ** 2, weights=weights))
    effective_n = weights.sum() ** 2 / np.square(weights).sum()
    error = np.sqrt(variance / effective_n) if effective_n > 1 else np.nan
    return mean, float(error), len(rates)


def update_raw_rate_statistics(row, bins_data, intervals):
    """Keep time variation separate from count statistics in summary outputs."""
    for field, prefix in (("total_rate", "stable_trigger_rate"),
                          ("any_vetoed_rate", "stable_any_vetoed_rate"),
                          ("non_vetoed_rate", "stable_non_vetoed_rate")):
        mean, stat, exposure = poisson_rate_summary(bins_data, intervals, field)
        _, scatter, _ = stable_rate(rate_field_bins(bins_data, field), intervals)
        row[f"{prefix}_mean_hz"] = float(mean)
        row[f"{prefix}_stat_error_hz"] = float(stat)
        row[f"{prefix}_error_hz"] = float(stat)
        row[f"{prefix}_scatter_error_hz"] = float(scatter)
        if field == "total_rate":
            row["trigger_rate_covered_time_s"] = exposure
    any_mean = row["stable_any_vetoed_rate_mean_hz"]
    non_mean = row["stable_non_vetoed_rate_mean_hz"]
    total = any_mean + non_mean
    row["stable_any_vetoed_fraction"] = any_mean / total if total > 0 else np.nan
    row["stable_any_vetoed_fraction_stat_error"] = (
        math.hypot(non_mean * row["stable_any_vetoed_rate_stat_error_hz"],
                   any_mean * row["stable_non_vetoed_rate_stat_error_hz"]) / total**2
        if total > 0 else np.nan
    )


def restrict_intervals_to_elapsed_range(intervals, elapsed_range, time_origin):
    """Intersect absolute stable intervals with an elapsed-time range."""
    if elapsed_range is None:
        return list(intervals)
    elapsed_start, elapsed_stop = elapsed_range
    selected_start = time_origin + elapsed_start
    selected_stop = time_origin + elapsed_stop
    return [
        (max(start, selected_start), min(stop, selected_stop))
        for start, stop in intervals
        if min(stop, selected_stop) > max(start, selected_start)
    ]


def solve_true_rate_from_reco_deadtimes(r_reco, deadtimes_s, live_window_s):
    """Solve the three-deadtime model, assuming D3 repeats from cycle three."""
    deadtimes_s = tuple(deadtimes_s)
    if len(deadtimes_s) != TRUE_RATE_DEADTIME_COUNT:
        raise ValueError("exactly three dead times are required")
    if any(not math.isfinite(value) or value < 0.0 for value in deadtimes_s):
        raise ValueError("dead times must be finite and non-negative")
    if not math.isfinite(live_window_s) or live_window_s <= 0.0:
        raise ValueError("live window must be finite and positive")
    if not math.isfinite(r_reco) or r_reco <= 0.0:
        return 0.0

    d1_s, d2_s, d3_s = deadtimes_s

    def reco_from_true(r_true):
        q = math.exp(-live_window_s * r_true)
        p = -math.expm1(-live_window_s * r_true)
        if p <= 0.0:
            return 0.0
        expected_deadtime_s = d1_s + q * d2_s + (q**2 / p) * d3_s
        return r_true / (1.0 + r_true * expected_deadtime_s)

    lo = 0.0
    hi = max(1.0, r_reco * 2.0 + 1.0)
    for _ in range(100):
        if reco_from_true(hi) >= r_reco:
            break
        hi *= 2.0
    else:
        return math.nan

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if reco_from_true(mid) < r_reco:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def update_deadtime_mean_fields(row):
    """Recalculate the displayed deadtime summary from D1-D3 only."""
    values = np.asarray(
        [
            row.get(f"deadtime_{i}_ms", np.nan)
            for i in range(1, TRUE_RATE_DEADTIME_COUNT + 1)
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        return row

    row["deadtime_mean_ms"] = float(values.mean())
    row["deadtime_std_ms"] = float(values.std(ddof=1))

    errors = np.asarray(
        [
            row.get(f"deadtime_{i}_error_ms", np.nan)
            for i in range(1, TRUE_RATE_DEADTIME_COUNT + 1)
        ],
        dtype=float,
    )
    if np.all(np.isfinite(errors)):
        row["deadtime_mean_error_ms"] = float(
            np.sqrt(np.square(errors).sum()) / TRUE_RATE_DEADTIME_COUNT
        )

    widths = np.asarray(
        [
            row.get(f"deadtime_{i}_sigma_ms", np.nan)
            for i in range(1, TRUE_RATE_DEADTIME_COUNT + 1)
        ],
        dtype=float,
    )
    if np.all(np.isfinite(widths)):
        row["deadtime_mean_sigma_ms"] = float(
            np.sqrt(np.square(widths).sum()) / TRUE_RATE_DEADTIME_COUNT
        )
        if np.isfinite(float(row.get("deadtime_mean_error_ms", np.nan))):
            row["deadtime_mean_display_error_ms"] = (
                row["deadtime_mean_error_ms"]
                + row["deadtime_mean_sigma_ms"]
            )
    return row


def add_true_rate_corrections(rows, live_window_ms, deadtime_model_rows=None):
    """Build a shared dead-time model, then reconstruct every row's rate."""
    if deadtime_model_rows is None:
        deadtime_model_rows = rows
    vectors = []
    model_run_numbers = []
    seen_model_runs = set()
    for row in deadtime_model_rows:
        run_number = int(row["run_number"])
        if run_number in seen_model_runs:
            continue
        values = np.asarray(
            [
                row[f"deadtime_{i}_ms"]
                for i in range(1, TRUE_RATE_DEADTIME_COUNT + 1)
            ],
            dtype=float,
        )
        if np.all(np.isfinite(values)) and np.all(values >= 0.0):
            vectors.append(values)
            model_run_numbers.append(run_number)
            seen_model_runs.add(run_number)
    if not vectors:
        raise RuntimeError("no runs contain three finite dead-time values")

    matrix = np.asarray(vectors)
    mean_ms = matrix.mean(axis=0)
    std_ms = (
        matrix.std(axis=0, ddof=1)
        if len(matrix) > 1
        else np.zeros(TRUE_RATE_DEADTIME_COUNT, dtype=float)
    )
    low_ms = np.maximum(mean_ms - std_ms, 0.0)
    high_ms = mean_ms + std_ms
    live_window_s = live_window_ms / 1000.0
    scenarios = {
        "stable_trigger_rate_true_hz": mean_ms,
        "stable_trigger_rate_true_deadtime_sys_low_hz": low_ms,
        "stable_trigger_rate_true_deadtime_sys_high_hz": high_ms,
    }

    for row in rows:
        recorded_rate = float(row["stable_trigger_rate_mean_hz"])
        pre_over_post = float(row.get("pre_over_post_ratio", 1.0))
        corrected_reco_rate = recorded_rate * pre_over_post
        row["stable_trigger_rate_pre_veto_reco_hz"] = corrected_reco_rate
        any_vetoed_fraction = float(
            row.get("stable_any_vetoed_fraction", 1.0)
        )
        for field, deadtimes_ms in scenarios.items():
            deadtime_corrected_total = (
                solve_true_rate_from_reco_deadtimes(
                    corrected_reco_rate, deadtimes_ms / 1000.0, live_window_s
                )
                if np.isfinite(corrected_reco_rate) and pre_over_post >= 1.0
                else np.nan
            )
            valid_any_vetoed = (
                np.isfinite(deadtime_corrected_total)
                and np.isfinite(any_vetoed_fraction)
                and 0.0 <= any_vetoed_fraction <= 1.0
            )
            deadtime_corrected_any_vetoed = (
                deadtime_corrected_total * any_vetoed_fraction
                if valid_any_vetoed
                else np.nan
            )
            row[field] = deadtime_corrected_any_vetoed
            if field == "stable_trigger_rate_true_hz":
                row["stable_trigger_rate_deadtime_corrected_total_hz"] = (
                    deadtime_corrected_total
                )
                row["stable_any_vetoed_rate_deadtime_corrected_hz"] = (
                    deadtime_corrected_any_vetoed
                )
        central = row["stable_trigger_rate_true_hz"]
        low = row["stable_trigger_rate_true_deadtime_sys_low_hz"]
        high = row["stable_trigger_rate_true_deadtime_sys_high_hz"]
        row["stable_trigger_rate_true_deadtime_sys_minus_hz"] = central - low
        row["stable_trigger_rate_true_deadtime_sys_plus_hz"] = high - central

        # Statistical variations hold Pre/Post and all deadtime values fixed.
        stat_error = float(row.get("stable_trigger_rate_stat_error_hz", np.nan))
        variations = correction_scenarios(
            recorded_rate, stat_error, pre_over_post,
            {"deadtime_mean_ms": mean_ms, "deadtime_std_ms": std_ms,
             "live_window_ms": live_window_ms},
        )
        # If both trigger classes are available, fluctuate their disjoint
        # counts separately. This also preserves the correlation of the total
        # rate with the AnyVetoed fraction when their proportions vary in time.
        any_rate = float(row.get("stable_any_vetoed_rate_mean_hz", np.nan))
        non_rate = float(row.get("stable_non_vetoed_rate_mean_hz", np.nan))
        any_error = float(row.get("stable_any_vetoed_rate_stat_error_hz", np.nan))
        non_error = float(row.get("stable_non_vetoed_rate_stat_error_hz", np.nan))
        split_errors = {"minus": [], "plus": []}
        if all(np.isfinite(x) and x >= 0 for x in (any_rate, non_rate, any_error, non_error)):
            for index, error in enumerate((any_error, non_error)):
                values = []
                for direction in (-1, 1):
                    components = [any_rate, non_rate]
                    original = components[index]
                    components[index] = max(0.0, original + direction * error)
                    reco = max(0.0, recorded_rate + components[index] - original)
                    fraction = components[0] / sum(components) if sum(components) > 0 else 0.0
                    values.append(solve_true_rate_from_reco_deadtimes(
                        reco * pre_over_post, mean_ms / 1000, live_window_s) * fraction)
                split_errors["minus"].append(max(0, central - min(values)))
                split_errors["plus"].append(max(0, max(values) - central))
        for side, scenario in (("minus", "stat_low"), ("plus", "stat_high")):
            shifted = float(variations[scenario]["true_total"]) * any_vetoed_fraction
            stat = (float(np.linalg.norm(split_errors[side])) if split_errors[side]
                    else abs(shifted - central))
            row[f"stable_trigger_rate_true_stat_{side}_hz"] = stat
            row[f"stable_trigger_rate_true_total_error_{side}_hz"] = math.hypot(
                stat, row[f"stable_trigger_rate_true_deadtime_sys_{side}_hz"]
            )

    reconstructed_run_numbers = list(
        dict.fromkeys(int(row["run_number"]) for row in rows)
    )
    return {
        "model": "three_interval_D3_repeated",
        "deadtime_systematic_definition": (
            "all D1-D3 shifted simultaneously to their per-column "
            "mean +/- sample standard deviation"
        ),
        "deadtime_uncertainty_type": "correlated_one_sigma_envelope",
        "run_count": len(vectors),
        "deadtime_model_run_numbers": model_run_numbers,
        "reconstructed_point_count": len(rows),
        "reconstructed_run_count": len(reconstructed_run_numbers),
        "reconstructed_run_numbers": reconstructed_run_numbers,
        "live_window_ms": live_window_ms,
        "input_rate_definition": (
            "raw recorded total trigger rate * per-run PreVeto/PostVeto"
        ),
        "prepost_source": "per-run TLU counters in prepost_veto_ratios.csv",
        "output_rate_definition": (
            "R_true_total * R_reco_AnyVetoed / "
            "(R_reco_AnyVetoed + R_reco_NonVetoed)"
        ),
        "prescale_source": "not applied",
        "statistical_uncertainty_definition": (
            "independent Poisson raw AnyVetoed/NonVetoed counts; vary each "
            "class rate +/- statistical error through the full correction; "
            "use total reco rate alone if class statistics are unavailable"
        ),
        "prepost_uncertainty": "neglected",
        "trigger_classification_source": "raw TLU input 5/6 bits",
        "deadtime_mean_ms": mean_ms.tolist(),
        "deadtime_std_ms": std_ms.tolist(),
        "deadtime_low_ms": low_ms.tolist(),
        "deadtime_high_ms": high_ms.tolist(),
    }


def write_true_rate_error_from_deadtime_sigma(rows, model, path):
    """Write the per-run true-rate uncertainty induced by deadtime sigma."""
    fieldnames = [
        "run_number",
        "configuration",
        "deadtime_1_mean_ms",
        "deadtime_1_sigma_ms",
        "deadtime_2_mean_ms",
        "deadtime_2_sigma_ms",
        "deadtime_3_mean_ms",
        "deadtime_3_sigma_ms",
        "true_rate_hz",
        "true_rate_at_deadtime_minus_1sigma_hz",
        "true_rate_at_deadtime_plus_1sigma_hz",
        "true_rate_error_minus_hz",
        "true_rate_error_plus_hz",
        "true_rate_error_minus_percent",
        "true_rate_error_plus_percent",
    ]
    means = model["deadtime_mean_ms"]
    sigmas = model["deadtime_std_ms"]
    output_rows = []
    for row in rows:
        central = float(row["stable_trigger_rate_true_hz"])
        low = float(row["stable_trigger_rate_true_deadtime_sys_low_hz"])
        high = float(row["stable_trigger_rate_true_deadtime_sys_high_hz"])
        minus = float(row["stable_trigger_rate_true_deadtime_sys_minus_hz"])
        plus = float(row["stable_trigger_rate_true_deadtime_sys_plus_hz"])
        if not all(np.isfinite(value) for value in (central, low, high, minus, plus)):
            continue
        output_rows.append(
            {
                "run_number": int(row["run_number"]),
                "configuration": row.get("configuration", ""),
                "deadtime_1_mean_ms": means[0],
                "deadtime_1_sigma_ms": sigmas[0],
                "deadtime_2_mean_ms": means[1],
                "deadtime_2_sigma_ms": sigmas[1],
                "deadtime_3_mean_ms": means[2],
                "deadtime_3_sigma_ms": sigmas[2],
                "true_rate_hz": central,
                "true_rate_at_deadtime_minus_1sigma_hz": low,
                "true_rate_at_deadtime_plus_1sigma_hz": high,
                "true_rate_error_minus_hz": minus,
                "true_rate_error_plus_hz": plus,
                "true_rate_error_minus_percent": (
                    100.0 * minus / central if central > 0.0 else np.nan
                ),
                "true_rate_error_plus_percent": (
                    100.0 * plus / central if central > 0.0 else np.nan
                ),
            }
        )
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def plot_true_rate_error_from_deadtime_sigma(
    rows,
    path,
    title="Deadtime sigma contribution to true-rate uncertainty",
    rate_label="True rate [Hz]",
):
    """Plot absolute and relative true-rate errors from the +/-1 sigma cases."""
    selected = []
    for row in rows:
        values = (
            float(row["stable_trigger_rate_true_hz"]),
            float(row["stable_trigger_rate_true_deadtime_sys_minus_hz"]),
            float(row["stable_trigger_rate_true_deadtime_sys_plus_hz"]),
        )
        if all(np.isfinite(value) for value in values) and values[0] > 0.0:
            selected.append(values)
    selected.sort()
    central = np.asarray([value[0] for value in selected], dtype=float)
    minus = np.asarray([value[1] for value in selected], dtype=float)
    plus = np.asarray([value[2] for value in selected], dtype=float)

    fig, (absolute_ax, relative_ax) = plt.subplots(
        2, 1, figsize=(9, 9), sharex=True
    )
    absolute_ax.scatter(
        central, minus, marker="v", s=28, label="lower error: central - (-1 sigma case)"
    )
    absolute_ax.scatter(
        central, plus, marker="^", s=28, label="upper error: (+1 sigma case) - central"
    )
    absolute_ax.set_ylabel("Deadtime-induced true-rate error [Hz]")
    absolute_ax.grid(True, alpha=0.3)
    absolute_ax.legend(fontsize=9)

    relative_ax.scatter(
        central, 100.0 * minus / central, marker="v", s=28, label="lower error"
    )
    relative_ax.scatter(
        central, 100.0 * plus / central, marker="^", s=28, label="upper error"
    )
    relative_ax.set(
        xlabel=rate_label,
        ylabel="Deadtime-induced relative error [%]",
    )
    relative_ax.grid(True, alpha=0.3)
    relative_ax.legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def collect(
    run_info, batch_dir, bins, prescale=None, trigger_rate_dir=None,
    mean_elapsed_range_s=None, deadtime_required=True,
):
    run = int(run_info["run_number"])
    job_dir = batch_dir / f"run_{run}_bins_{bins}"
    dt_path = job_dir / f"run_{run}_deadtime.csv"
    histogram_path = job_dir / f"run_{run}_stable_deadtime_histogram.csv"
    row = {
        "run_number": run,
        "analysis_bins": bins,
        "configuration": run_info.get("configuration", ""),
        "stable_time_s": run_info.get("stable_time_sec", ""),
        "mean_elapsed_start_s": (
            mean_elapsed_range_s[0]
            if mean_elapsed_range_s is not None else np.nan
        ),
        "mean_elapsed_stop_s": (
            mean_elapsed_range_s[1]
            if mean_elapsed_range_s is not None else np.nan
        ),
        "mean_time_origin_utime": np.nan,
        "mean_selected_stable_time_s": np.nan,
        "n_deadtime_samples": histogram_sample_count(histogram_path),
    }
    for i in range(1, DEADTIME_INTERVAL_COUNT + 1):
        row[f"deadtime_{i}_ms"] = np.nan
        row[f"deadtime_{i}_error_ms"] = np.nan
        row[f"deadtime_{i}_sigma_ms"] = np.nan
        row[f"deadtime_{i}_display_error_ms"] = np.nan
    row.update(
        deadtime_mean_ms=np.nan, deadtime_mean_error_ms=np.nan,
        deadtime_mean_sigma_ms=np.nan,
        deadtime_mean_display_error_ms=np.nan,
        deadtime_std_ms=np.nan, stable_trigger_rate_mean_hz=np.nan,
        stable_trigger_rate_error_hz=np.nan, n_deadtime_intervals=0,
        stable_any_vetoed_rate_mean_hz=np.nan,
        stable_any_vetoed_rate_error_hz=np.nan,
        stable_non_vetoed_rate_mean_hz=np.nan,
        stable_non_vetoed_rate_error_hz=np.nan,
        stable_any_vetoed_fraction=np.nan,
        any_vetoed_prescale_factor=(
            prescale.correction_factor if prescale is not None else np.nan
        ),
        any_vetoed_pass_fraction=(
            prescale.pass_fraction if prescale is not None else np.nan
        ),
        faser_calib_rate=(
            prescale.calib_rate if prescale is not None else np.nan
        ),
        veto_trigger_stretch=(
            prescale.veto_stretch if prescale is not None else np.nan
        ),
        stable_luminosity_mean_ub_inv_s=np.nan,
        stable_luminosity_error_ub_inv_s=np.nan,
        n_trigger_rate_bins=0, n_luminosity_bins=0,
        rate_source="", luminosity_source="", status="missing_deadtime_csv",
    )
    has_deadtime_csv = dt_path.is_file()
    dt = deadtimes(dt_path) if has_deadtime_csv else []
    row["n_deadtime_intervals"] = len(dt)
    for i, value in enumerate(dt, 1):
        row[f"deadtime_{i}_ms"] = float(value["deadtime_ms"])
        row[f"deadtime_{i}_error_ms"] = float(
            value["deadtime_total_error_ms"]
        )
        sigma_text = value.get("deadtime_transition_width_ms", "")
        sigma = float(sigma_text) if sigma_text else np.nan
        row[f"deadtime_{i}_sigma_ms"] = sigma
        if np.isfinite(sigma):
            row[f"deadtime_{i}_display_error_ms"] = (
                row[f"deadtime_{i}_error_ms"] + sigma
            )
    if len(dt) >= TRUE_RATE_DEADTIME_COUNT:
        update_deadtime_mean_fields(row)
    bins_data, source = rate_bins(
        job_dir, run, trigger_rate_dir=trigger_rate_dir
    )
    intervals = [
        (float(v["start_utime"]), float(v["stop_utime"]))
        for v in run_info["stable_list"]
    ]
    if mean_elapsed_range_s is not None:
        if bins_data:
            time_origin = min(item["start"] for item in bins_data)
        else:
            try:
                time_origin = float(run_info["start_utime"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"run {run} needs start_utime when an elapsed mean "
                    "range is specified without trigger-rate bins"
                ) from error
        row["mean_time_origin_utime"] = time_origin
        intervals = restrict_intervals_to_elapsed_range(
            intervals, mean_elapsed_range_s, time_origin
        )
    row["mean_selected_stable_time_s"] = sum(
        stop - start for start, stop in intervals
    )
    total_bins = rate_field_bins(bins_data, "total_rate")
    any_bins = rate_field_bins(bins_data, "any_vetoed_rate")
    non_bins = rate_field_bins(bins_data, "non_vetoed_rate")
    mean, error, count = stable_rate(total_bins, intervals)
    row["stable_trigger_rate_mean_hz"] = mean
    row["stable_trigger_rate_error_hz"] = error
    any_mean, any_error, any_count = stable_rate(any_bins, intervals)
    non_mean, non_error, non_count = stable_rate(non_bins, intervals)
    row["stable_any_vetoed_rate_mean_hz"] = any_mean
    row["stable_any_vetoed_rate_error_hz"] = any_error
    row["stable_non_vetoed_rate_mean_hz"] = non_mean
    row["stable_non_vetoed_rate_error_hz"] = non_error
    reco_veto_total = any_mean + non_mean
    if (
        np.isfinite(any_mean)
        and np.isfinite(non_mean)
        and reco_veto_total > 0.0
    ):
        row["stable_any_vetoed_fraction"] = any_mean / reco_veto_total
    update_raw_rate_statistics(row, bins_data, intervals)

    csv_factors = np.asarray(
        [item["prescale_factor"] for item in bins_data], dtype=float
    )
    csv_factors = csv_factors[np.isfinite(csv_factors)]
    if prescale is None and len(csv_factors):
        if not np.allclose(csv_factors, csv_factors[0]):
            # Prescale is retained only as optional metadata, never a correction.
            csv_factors = np.asarray([np.nan])
        row["any_vetoed_prescale_factor"] = float(csv_factors[0])
        row["any_vetoed_pass_fraction"] = 1.0 / float(csv_factors[0])
    if (
        prescale is None
        and not np.isfinite(row["any_vetoed_prescale_factor"])
        and "PreScale" not in str(row["configuration"])
    ):
        row["any_vetoed_prescale_factor"] = 1.0
        row["any_vetoed_pass_fraction"] = 1.0
    row["n_trigger_rate_bins"] = count
    row["rate_source"] = str(source or "")
    lumi_data, lumi_source = luminosity_bins(job_dir, run)
    lumi_mean, lumi_error, lumi_count = stable_rate(lumi_data, intervals)
    row["stable_luminosity_mean_ub_inv_s"] = lumi_mean
    row["stable_luminosity_error_ub_inv_s"] = lumi_error
    row["n_luminosity_bins"] = lumi_count
    row["luminosity_source"] = str(lumi_source or "")
    if mean_elapsed_range_s is not None and not intervals:
        row["status"] = "no_mean_window_stable_overlap"
    elif not bins_data:
        row["status"] = "missing_trigger_rate"
    elif not any_bins or not non_bins:
        row["status"] = "missing_raw_veto_rates"
    elif deadtime_required and not has_deadtime_csv:
        row["status"] = "missing_deadtime_csv"
    elif deadtime_required and len(dt) < TRUE_RATE_DEADTIME_COUNT:
        row["status"] = "missing_deadtime_intervals"
    elif not np.isfinite(mean):
        row["status"] = "no_stable_trigger_rate_bins"
    else:
        row["status"] = "ok"
    return row


def aggregate_label(rows, yfield, label):
    """Add the across-run mean and sample standard deviation to a label."""
    values = np.asarray(
        [float(row[yfield]) for row in rows if np.isfinite(float(row[yfield]))],
        dtype=float,
    )
    if not len(values):
        return label
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return (
        f"{label}: mean={mean:.3f} ms, "
        f"σ={std:.3f} ms"
    )


def plot(rows, xfield, series, path, xlabel, ylabel, title, xerr=False):
    fig, ax = plt.subplots(figsize=(9, 6))
    for yfield, errfield, label, color, marker in series:
        selected = [
            r for r in rows
            if np.isfinite(float(r[xfield])) and np.isfinite(float(r[yfield]))
        ]
        if not selected:
            continue
        x = np.asarray([r[xfield] for r in selected], float)
        y = np.asarray([r[yfield] for r in selected], float)
        xe = (
            np.asarray(
                [r["stable_trigger_rate_error_hz"] for r in selected],
                dtype=float,
            )
            if xerr else None
        )
        ye = (
            np.asarray([r[errfield] for r in selected], float)
            if errfield else None
        )
        ax.errorbar(
            x, y, xerr=xe, yerr=ye, fmt=marker, color=color,
            capsize=2, linestyle="none",
            label=aggregate_label(rows, yfield, label),
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9,1.9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


CONFIGURATION_STYLES = {
    "AHCALPhysics_NoVeto": ("C0", "o"),
    "AHCALPhysics_Veto_PreScale": ("C1", "s"),
    "AHCALPhysics_Veto_NoPreScale": ("C2", "^"),
    "AHCALOperationUJ12_VetoOnly": ("C3", "D"),
    "TLU_VetoOnly": ("C4", "v"),
    "Unknown": ("C7", "X"),
}


def configuration_groups(rows):
    """Return config groups with fixed colors and markers across plots."""
    grouped = {}
    for row in rows:
        configuration = str(row.get("configuration") or "Unknown")
        grouped.setdefault(configuration, []).append(row)
    return [
        (
            configuration,
            group,
            CONFIGURATION_STYLES.get(configuration, ("C8", "P"))[0],
            CONFIGURATION_STYLES.get(configuration, ("C8", "P"))[1],
        )
        for configuration, group in sorted(grouped.items())
    ]


def plot_run_vs_mean_trigger_rate(
    rows, path, min_events=0, xlim=None, ylim=None
):
    """Plot raw recorded AnyVetoed rate by run and configuration."""
    selected = [
        row for row in rows
        if np.isfinite(float(row["run_number"]))
        and np.isfinite(float(row["stable_any_vetoed_rate_mean_hz"]))
        and float(row["n_deadtime_samples"]) >= min_events
    ]
    selected.sort(key=lambda row: float(row["run_number"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for configuration, group, color, marker in configuration_groups(selected):
        run_numbers = np.asarray([row["run_number"] for row in group], float)
        mean_rates = np.asarray(
            [row["stable_any_vetoed_rate_mean_hz"] for row in group], float
        )
        rate_errors = np.asarray(
            [row["stable_any_vetoed_rate_error_hz"] for row in group], float
        )
        yerr = rate_errors if np.all(np.isfinite(rate_errors)) else None
        ax.errorbar(
            run_numbers,
            mean_rates,
            yerr=yerr,
            fmt=marker,
            color=color,
            capsize=2,
            linestyle="none",
            label=f"{configuration} ({len(group)} runs)",
        )
    if selected:
        ax.legend(fontsize=8)

    title = "Run number vs recorded AnyVetoed rate"
    if min_events > 0:
        title += f" (N events >= {min_events:,})"
    ax.set(
        xlabel="Run number",
        ylabel="Recorded AnyVetoed rate in stable intervals [Hz]",
        title=title,
    )
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_run_vs_true_trigger_rate(
    rows,
    path,
    min_events=0,
    title="Run number vs true AnyVetoed rate",
    ylabel="True AnyVetoed rate in stable intervals [Hz]",
):
    """Plot deadtime-corrected rates by run/configuration."""
    selected = [
        row for row in rows
        if np.isfinite(float(row["stable_trigger_rate_true_hz"]))
        and float(row["n_deadtime_samples"]) >= min_events
    ]
    selected.sort(key=lambda row: float(row["run_number"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for configuration, group, color, marker in configuration_groups(selected):
        run_numbers = np.asarray([row["run_number"] for row in group], float)
        central = np.asarray(
            [row["stable_trigger_rate_true_hz"] for row in group], float
        )
        rate_minus = np.asarray(
            [
                row.get("stable_trigger_rate_true_total_error_minus_hz",
                        row["stable_trigger_rate_true_deadtime_sys_minus_hz"])
                for row in group
            ],
            float,
        )
        rate_plus = np.asarray(
            [
                row.get("stable_trigger_rate_true_total_error_plus_hz",
                        row["stable_trigger_rate_true_deadtime_sys_plus_hz"])
                for row in group
            ],
            float,
        )
        yerr = (
            np.vstack((rate_minus, rate_plus))
            if np.all(np.isfinite(rate_minus))
            and np.all(np.isfinite(rate_plus))
            else None
        )
        ax.errorbar(
            run_numbers,
            central,
            yerr=yerr,
            fmt=marker,
            color=color,
            capsize=2,
            linestyle="none",
            label=f"{configuration} ({len(group)} runs)",
        )
    if selected:
        ax.legend(fontsize=8)

    if min_events > 0:
        title += f" (N events >= {min_events:,})"
    ax.set(
        xlabel="Run number",
        ylabel=ylabel,
        title=title,
    )
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_luminosity_vs_recorded_trigger_rate(rows, path, min_events=0):
    """Plot raw recorded AnyVetoed rate against luminosity by config."""
    selected = [
        row for row in rows
        if np.isfinite(float(row["stable_any_vetoed_rate_mean_hz"]))
        and np.isfinite(float(row["stable_luminosity_mean_ub_inv_s"]))
        and float(row["n_deadtime_samples"]) >= min_events
    ]
    selected.sort(key=lambda row: float(row["stable_luminosity_mean_ub_inv_s"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for configuration, group, color, marker in configuration_groups(selected):
        luminosities = np.asarray(
            [row["stable_luminosity_mean_ub_inv_s"] for row in group], float
        )
        recorded_rates = np.asarray(
            [row["stable_any_vetoed_rate_mean_hz"] for row in group], float
        )
        luminosity_errors = np.asarray(
            [row["stable_luminosity_error_ub_inv_s"] for row in group], float
        )
        rate_errors = np.asarray(
            [row["stable_any_vetoed_rate_error_hz"] for row in group], float
        )
        xerr = (
            luminosity_errors
            if np.all(np.isfinite(luminosity_errors))
            else None
        )
        yerr = rate_errors if np.all(np.isfinite(rate_errors)) else None
        ax.errorbar(
            luminosities,
            recorded_rates,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            color=color,
            capsize=2,
            linestyle="none",
            label=f"{configuration} ({len(group)} runs)",
        )
    if selected:
        ax.legend(fontsize=8)

    title = "Recorded AnyVetoed rate vs luminosity"
    if min_events > 0:
        title += f" (N events >= {min_events:,})"
    ax.set(
        xlabel=r"Mean luminosity in stable intervals [$\mu$b$^{-1}$ s$^{-1}$]",
        ylabel="Recorded AnyVetoed rate in stable intervals [Hz]",
        title=title,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_luminosity_vs_true_trigger_rate(
    rows,
    path,
    min_events=0,
    title="True AnyVetoed rate vs luminosity",
    ylabel="True AnyVetoed rate in stable intervals [Hz]",
):
    """Plot stable-beam mean luminosity by run configuration."""
    selected = [
        row for row in rows
        if np.isfinite(float(row["stable_trigger_rate_true_hz"]))
        and np.isfinite(float(row["stable_luminosity_mean_ub_inv_s"]))
        and float(row["n_deadtime_samples"]) >= min_events
    ]
    selected.sort(key=lambda row: float(row["stable_trigger_rate_true_hz"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    if selected:
        for configuration, group, color, marker in configuration_groups(selected):
            true_rates = np.asarray(
                [row["stable_trigger_rate_true_hz"] for row in group], float
            )
            luminosities = np.asarray(
                [row["stable_luminosity_mean_ub_inv_s"] for row in group], float
            )
            luminosity_errors = np.asarray(
                [row["stable_luminosity_error_ub_inv_s"] for row in group],
                float,
            )
            xerr = (
                luminosity_errors
                if np.all(np.isfinite(luminosity_errors))
                else None
            )
            ax.errorbar(
                luminosities,
                true_rates,
                xerr=xerr,
                yerr=(np.asarray([
                    [r.get("stable_trigger_rate_true_total_error_minus_hz", np.nan) for r in group],
                    [r.get("stable_trigger_rate_true_total_error_plus_hz", np.nan) for r in group],
                ]) if all(np.isfinite(r.get("stable_trigger_rate_true_total_error_minus_hz", np.nan))
                          and np.isfinite(r.get("stable_trigger_rate_true_total_error_plus_hz", np.nan))
                          for r in group) else None),
                fmt=marker,
                color=color,
                capsize=2,
                linestyle="none",
                label=f"{configuration} ({len(group)} runs)",
            )
        ax.legend(fontsize=8)

    if min_events > 0:
        title += f" (N events >= {min_events:,})"
    ax.set(
        xlabel=r"Mean luminosity in stable intervals [$\mu$b$^{-1}$ s$^{-1}$]",
        ylabel=ylabel,
        title=title,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_plots(rows, out, dpi, run_rate_min_events=0):
    events = "n_deadtime_samples"
    events_label = "Number of TLU time-difference samples"
    plt.rcParams["figure.dpi"] = dpi
    rate = "stable_any_vetoed_rate_mean_hz"
    rate_label = "Mean recorded AnyVetoed rate in stable intervals [Hz]"
    jobs = [
        ("trigger_rate_vs_deadtime_1.png", rate,
         [("deadtime_1_ms", "deadtime_1_display_error_ms", "Deadtime 1: D1 +/- (delta_mu,1 + width_1)", "C0", "o")],
         rate_label, "Deadtime 1 [ms]", "Trigger rate vs deadtime 1", False),
        ("trigger_rate_vs_deadtime_2_3_4.png", rate,
         [("deadtime_2_ms", "deadtime_2_display_error_ms", "Deadtime 2: D2 +/- (delta_mu,2 + width_2)", "C1", "o"),
          ("deadtime_3_ms", "deadtime_3_display_error_ms", "Deadtime 3: D3 +/- (delta_mu,3 + width_3)", "C2", "s"),
          ("deadtime_4_ms", "deadtime_4_display_error_ms", "Deadtime 4: D4 +/- (delta_mu,4 + width_4)", "C9", "^")],
         rate_label, "Deadtime [ms]", "Trigger rate vs deadtimes 2, 3, and 4", False),
        ("trigger_rate_vs_deadtime_mean.png", rate,
         [("deadtime_mean_ms", "deadtime_mean_display_error_ms",
           "Mean deadtime: D +/- (delta_mu + width)", "C3", "o")],
         rate_label, "Mean deadtime [ms]", "Trigger rate vs mean deadtime", False),
        ("run_number_vs_deadtime_mean.png", "run_number",
         [("deadtime_mean_ms", "deadtime_mean_display_error_ms",
           "Mean deadtime: D +/- (delta_mu + width)", "C4", "o")],
         "Run number", "Mean deadtime [ms]", "Stable run number vs mean deadtime", False),
        ("run_number_vs_deadtime_error.png", "run_number",
         [("deadtime_mean_error_ms", None, "Mean deadtime error", "C5", "o")],
         "Run number", "Mean deadtime error [ms]",
         "Run number vs mean deadtime error", False),
        ("trigger_rate_vs_deadtime_error.png", rate,
         [("deadtime_mean_error_ms", None, "Mean deadtime error", "C6", "o")],
         rate_label, "Mean deadtime error [ms]",
         "Trigger rate vs mean deadtime error", True),
         ("events_vs_deadtime_mean.png", events,
         [("deadtime_mean_ms", "deadtime_mean_display_error_ms",
            "Mean deadtime: D +/- (delta_mu + width)", "C7", "o")],
          events_label, "Mean deadtime [ms]", "Events vs mean deadtime", False),
        ("events_vs_deadtime_mean_1.png", events,
         [("deadtime_1_ms", "deadtime_1_display_error_ms", "Deadtime 1: D1 +/- (delta_mu,1 + width_1)", "C0", "o"),],
          events_label, "Mean deadtime [ms]", "Events vs mean deadtime", False),  
        ("events_vs_deadtime_mean2_3_4.png", events,
         [("deadtime_2_ms", "deadtime_2_display_error_ms", "Deadtime 2: D2 +/- (delta_mu,2 + width_2)", "C1", "o"),
          ("deadtime_3_ms", "deadtime_3_display_error_ms", "Deadtime 3: D3 +/- (delta_mu,3 + width_3)", "C2", "s"),
          ("deadtime_4_ms", "deadtime_4_display_error_ms", "Deadtime 4: D4 +/- (delta_mu,4 + width_4)", "C9", "^")],
          events_label, "Deadtime [ms]", "Events vs deadtimes 2, 3, and 4", False),
    ]
    for name, x, series, xlabel, ylabel, title, use_xerr in jobs:
        plot(rows, x, series, out / name, xlabel, ylabel, title, use_xerr)
    plot_run_vs_mean_trigger_rate(
        rows,
        out / "run_number_vs_trigger_rate_mean.png",
        min_events=run_rate_min_events,
        xlim=(22050,23300),
        ylim=(0,650),
    )
    plot_run_vs_true_trigger_rate(
        rows,
        out / "run_number_vs_true_trigger_rate.png",
        min_events=run_rate_min_events,
    )
    plot_luminosity_vs_recorded_trigger_rate(
        rows,
        out / "recorded_trigger_rate_vs_luminosity.png",
        min_events=run_rate_min_events,
    )
    plot_luminosity_vs_true_trigger_rate(
        rows,
        out / "luminosity_vs_true_trigger_rate.png",
        min_events=run_rate_min_events,
    )


def write_summary_files(rows, model, output_dir):
    """Persist the same summary/model/error schema for either entry point."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "deadtime_trigger_rate_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    model_path = output_dir / "deadtime_true_rate_model.json"
    model_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    error_path = output_dir / "true_rate_error_from_deadtime_sigma.csv"
    write_true_rate_error_from_deadtime_sigma(rows, model, error_path)
    return summary, model_path, error_path


def main():
    args = arguments()
    if args.min_events < 0:
        raise ValueError("--min-events must be non-negative")
    if args.run_rate_min_events < 0:
        raise ValueError("--run-rate-min-events must be non-negative")
    if (
        not math.isfinite(args.assumed_auto_triggered_ms)
        or args.assumed_auto_triggered_ms <= 0.0
    ):
        raise ValueError("--assumed-auto-triggered-ms must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overrides = run_bin_overrides(args.run_bins)
    selections = [(run, None, False) for run in args.run]
    if args.run_csv is not None:
        selections.extend(
            (run, elapsed_range, False)
            for run, elapsed_range in run_selection_rows_from_csv(args.run_csv)
        )
    if args.true_rate_run_csv is not None:
        selections.extend(
            (run, elapsed_range, True)
            for run, elapsed_range in run_selection_rows_from_csv(
                args.true_rate_run_csv
            )
        )
    selected_runs = [run for run, _elapsed_range, _true_only in selections]
    selected_run_info = stable_runs(
        args.run_json, args.limit, args.mintimes, selected_runs
    )
    ratios = pre_over_post_ratios(
        [args.prepost_ratio_csv, *args.additional_prepost_ratio_csv],
        [item["run_number"] for item in selected_run_info],
    )
    if selections:
        run_info_by_number = {
            int(item["run_number"]): item for item in selected_run_info
        }
        collection_selections = [
            (run_info_by_number[run], elapsed_range, true_rate_only)
            for run, elapsed_range, true_rate_only in selections
        ]
    else:
        collection_selections = [
            (run_info, None, False) for run_info in selected_run_info
        ]
    rows = []
    for run_info, elapsed_range, true_rate_only in collection_selections:
        run = int(run_info["run_number"])
        row = collect(
            run_info,
            args.batch_dir,
            overrides.get(run, args.bins),
            trigger_rate_dir=args.trigger_rate_dir,
            mean_elapsed_range_s=elapsed_range,
            deadtime_required=not true_rate_only,
        )
        row["pre_over_post_ratio"] = ratios[run]
        row["used_for_deadtime_model"] = not true_rate_only
        rows.append(row)
    rows = [
        row for row in rows
        if not row["used_for_deadtime_model"]
        or row["n_deadtime_samples"] >= args.min_events
    ]
    if not rows:
        raise RuntimeError(
            "no runs satisfy the time and deadtime-sample requirements"
        )
    deadtime_model_rows = [
        row for row in rows if row["used_for_deadtime_model"]
    ]
    true_rate_model = add_true_rate_corrections(
        rows,
        args.assumed_auto_triggered_ms,
        deadtime_model_rows=deadtime_model_rows,
    )
    output, model_output, error_output = write_summary_files(
        rows, true_rate_model, args.output_dir
    )
    plot_true_rate_error_from_deadtime_sigma(
        rows,
        args.output_dir / "true_rate_error_from_deadtime_sigma.png",
        title="Deadtime sigma contribution to true AnyVetoed-rate uncertainty",
        rate_label="True AnyVetoed rate [Hz]",
    )
    make_plots(
        rows,
        args.output_dir,
        args.dpi,
        run_rate_min_events=args.run_rate_min_events,
    )
    print(f"Wrote {output}")
    print(f"Wrote {model_output}")
    print(f"Wrote {error_output}")
    print(f"Complete runs: {sum(r['status'] == 'ok' for r in rows)}/{len(rows)}")


def refresh_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--prepost-ratio-csv", type=Path, default=DEFAULT_PREPOST)
    parser.add_argument("--additional-prepost-ratio-csv", type=Path, action="append", default=[])
    parser.add_argument("--trigger-rate-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assumed-auto-triggered-ms", type=float, default=3.9)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    with args.input_summary.open(encoding="utf-8", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    if not raw_rows:
        raise ValueError("input summary contains no runs")
    ratios = pre_over_post_ratios(
        [args.prepost_ratio_csv, *args.additional_prepost_ratio_csv],
        [int(row["run_number"]) for row in raw_rows]
    )
    rows = []
    run_info = {int(run["run_number"]): run for run in json.loads(args.run_json.read_text())}
    has_any_veto_fields = all(
        field in raw_rows[0]
        for field in (
            "faser_calib_rate",
            "veto_trigger_stretch",
            "stable_any_vetoed_fraction",
        )
    )
    for raw in raw_rows:
        row = {}
        for key, value in raw.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = value
        run = int(row["run_number"])
        row["run_number"] = run
        row["pre_over_post_ratio"] = ratios[run]
        update_deadtime_mean_fields(row)
        if not has_any_veto_fields:
            row.pop("stable_trigger_rate_deadtime_corrected_post_veto_hz", None)
        if (
            row.get("status") == "missing_deadtime_intervals"
            and all(
                math.isfinite(float(row[f"deadtime_{index}_ms"]))
                for index in range(1, 4)
            )
        ):
            row["status"] = "ok"
        if args.trigger_rate_dir is not None:
            rate_path = args.trigger_rate_dir / f"run_{run}_trigger_rate.csv"
            if not rate_path.is_file():
                raise FileNotFoundError(rate_path)
            row["rate_source"] = str(rate_path)
        # Recompute Poisson errors from raw bins, never from old scatter errors.
        source = Path(row.get("rate_source", ""))
        if not source.is_file():
            raise FileNotFoundError(f"run {run}: raw rate_source is required: {source}")
        bins, _ = rate_bins(source.parent, run, trigger_rate_dir=source.parent)
        intervals = [(x["start_utime"], x["stop_utime"])
                     for x in run_info[run]["stable_list"]]
        start = row.get("mean_elapsed_start_s", math.nan) or math.nan
        stop = row.get("mean_elapsed_stop_s", math.nan) or math.nan
        if math.isfinite(start) and math.isfinite(stop):
            intervals = restrict_intervals_to_elapsed_range(
                intervals, (start, stop), row["mean_time_origin_utime"])
        update_raw_rate_statistics(row, bins, intervals)
        rows.append(row)

    model_rows = [row for row in rows if str(row.get("used_for_deadtime_model", "True")).lower()
                  not in ("false", "0", "0.0")]
    model = add_true_rate_corrections(rows, args.assumed_auto_triggered_ms,
                                      deadtime_model_rows=model_rows)
    if not has_any_veto_fields:
        model["output_rate_definition"] = (
            "deadtime-corrected pre-veto total trigger rate"
        )
        model["prescale_source"] = "not applied"
    summary, model_path, error_summary = write_summary_files(rows, model, args.output_dir)
    if has_any_veto_fields:
        make_plots(rows, args.output_dir, args.dpi)
        plot_true_rate_error_from_deadtime_sigma(
            rows,
            args.output_dir / "true_rate_error_from_deadtime_sigma.png",
            title=(
                "Deadtime sigma contribution to true AnyVetoed-rate "
                "uncertainty"
            ),
            rate_label="True AnyVetoed rate [Hz]",
        )
    else:
        plot_run_vs_true_trigger_rate(
            rows,
            args.output_dir / "run_number_vs_true_trigger_rate.png",
            title="Run number vs deadtime-corrected pre-veto trigger rate",
            ylabel="Deadtime-corrected pre-veto trigger rate [Hz]",
        )
        plot_luminosity_vs_true_trigger_rate(
            rows,
            args.output_dir / "luminosity_vs_true_trigger_rate.png",
            title="Deadtime-corrected pre-veto trigger rate vs luminosity",
            ylabel="Deadtime-corrected pre-veto trigger rate [Hz]",
        )
        plot_true_rate_error_from_deadtime_sigma(
            rows,
            args.output_dir / "true_rate_error_from_deadtime_sigma.png",
            title=(
                "Deadtime sigma contribution to pre-veto true-rate "
                "uncertainty"
            ),
            rate_label="Deadtime-corrected pre-veto trigger rate [Hz]",
        )
    print(f"Wrote {summary}")
    print(f"Wrote {model_path}")
    print(f"Wrote {error_summary}")
    print(f"Complete runs: {sum(row['status'] == 'ok' for row in rows)}/{len(rows)}")

