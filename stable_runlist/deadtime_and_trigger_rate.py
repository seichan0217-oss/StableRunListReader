#!/usr/bin/env python3
"""Plot a run's TLU deadtime and recorded trigger rate side by side."""

from .config import OUTPUT_DIR, RUNLIST_JSON, RAW_RATE_DIR, FAIR_SCRIPTS_DIR

import argparse
from bisect import bisect_right
import csv
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .timestamp_diagnostics import (
    DEFAULT_RAW_PATTERN,
    PLOT_MAX_DIFFERENCE,
    TLU_CLOCK_HZ,
    _fit_vertical_edge,
    _smooth_log_counts,
    discover_raw_files,
    find_deadtime_intervals,
    read_histogram_csv,
    read_tlu_time_differences,
    write_deadtime_csv,
    write_histogram_csv,
)
from .pyroot_erf_fit import (
    draw_erf_fits_root,
    fit_erf_edges_independent_likelihood,
)

from .raw_trigger_rate import (
    RateBin, calculate_recorded_rate_rows as calculate_rate_rows, write_rate_csv,
)


SCRIPT_DIR = OUTPUT_DIR
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "deadtime_trigger_rate"
DEFAULT_LUMINOSITY_SCRIPT = FAIR_SCRIPTS_DIR / "Luminosity.py"
DEFAULT_LUMINOSITY_CSV = FAIR_SCRIPTS_DIR
DEFAULT_RUN_INTERVALS = RUNLIST_JSON


def find_initial_deadtime_interval(
    bin_centers,
    counts,
    search_max_s=0.002,
    smooth_bins=9,
    edge_fraction=0.45,
    min_stop_s=0.0002,
    max_stop_s=0.002,
    min_log_depth=0.7,
):
    all_x = np.asarray(bin_centers, dtype=float)
    all_y = np.asarray(counts, dtype=float)
    selected = (all_x >= 0.0) & (all_x <= search_max_s)
    selected_indices = np.flatnonzero(selected)
    x = all_x[selected]
    y = all_y[selected]
    if len(x) < smooth_bins + 2:
        return None

    smooth_log = _smooth_log_counts(y, smooth_bins)
    slope = np.gradient(smooth_log, x)
    rising_peaks = [
        index
        for index in range(1, len(x) - 1)
        if slope[index] >= slope[index - 1]
        and slope[index] > slope[index + 1]
        and min_stop_s <= x[index] <= max_stop_s
    ]
    if not rising_peaks:
        return None

    bin_width = float(np.median(np.diff(x)))
    lookaround_bins = max(3, int(round(0.0003 / bin_width)))
    candidates = []
    for peak in rising_peaks:
        left_level = float(np.min(smooth_log[: peak + 1]))
        right_level = float(
            np.max(smooth_log[peak : min(len(x), peak + lookaround_bins + 1)])
        )
        depth = right_level - left_level
        if depth >= min_log_depth:
            candidates.append((depth * max(slope[peak], 0.0), peak))
    if not candidates:
        return None

    _score, peak = max(candidates)
    stop_fit = _fit_vertical_edge(
        x, slope, peak, direction=1, fraction=edge_fraction
    )
    d_stop = stop_fit["position_s"]
    if not min_stop_s <= d_stop <= max_stop_s:
        return None

    range_error = stop_fit["position_error_s"]
    # Only D_stop is fitted. There is no independent uncertainty from a
    # tstart edge, so the one-edge uniform errors use sqrt(12), not sqrt(6).
    bin_error = bin_width / np.sqrt(12.0)
    smoothing_error = smooth_bins * bin_width / np.sqrt(12.0)
    total_error = float(
        np.sqrt(range_error**2 + bin_error**2 + smoothing_error**2)
    )
    missing = float("nan")
    return {
        "tstart_s": missing,
        "tstart_error_s": missing,
        "tstart_low_s": missing,
        "tstart_high_s": missing,
        "tstop_s": d_stop,
        "tstop_error_s": stop_fit["position_error_s"],
        "tstop_low_s": stop_fit["position_low_s"],
        "tstop_high_s": stop_fit["position_high_s"],
        "deadtime_s": d_stop,
        "deadtime_error_s": range_error,
        "deadtime_bin_error_s": bin_error,
        "deadtime_smoothing_error_s": smoothing_error,
        "deadtime_total_error_s": total_error,
        "deadtime_low_s": stop_fit["position_low_s"],
        "deadtime_high_s": stop_fit["position_high_s"],
        "left_fit_start_s": missing,
        "left_fit_stop_s": missing,
        "right_fit_start_s": stop_fit["fit_start_s"],
        "right_fit_stop_s": stop_fit["fit_stop_s"],
        "left_fit_bins": 0,
        "right_fit_bins": stop_fit["fit_bins"],
        "left_fit_indices": np.asarray([], dtype=int),
        "right_fit_indices": selected_indices[stop_fit["indices"]],
    }


def read_run_intervals(path, run_number, interval_set="stable_list"):
    with Path(path).open("r", encoding="utf-8") as stream:
        runs = json.load(stream)
    run = next(
        (item for item in runs if int(item.get("run_number", -1)) == run_number),
        None,
    )
    if run is None:
        raise ValueError(f"run {run_number} is not present in {path}")
    intervals = sorted(
        (float(item["start_utime"]), float(item["stop_utime"]))
        for item in run.get(interval_set, [])
        if float(item["stop_utime"]) > float(item["start_utime"])
    )
    if not intervals:
        raise ValueError(f"run {run_number} has no {interval_set} intervals")
    return intervals


def read_stable_tlu_time_differences(paths, stable_intervals, expected_run=None, merge_files=False):
    """Read TLU differences only within the same stable-beam interval."""
    starts = [item[0] for item in stable_intervals]
    stops = [item[1] for item in stable_intervals]
    differences = []
    previous_timestamp = None
    previous_interval = None
    events_in_stable = 0
    skipped = 0
    from .ahcal_raw_reader import RawFileReader

    if merge_files:
        from .raw_trigger_rate import MergedRawEventReader, TriggerDeduplicator
        streams = [MergedRawEventReader(paths)]
        deduplicator = TriggerDeduplicator()
    else:
        streams = (RawFileReader(path) for path in paths)
        deduplicator = None
    for stream in streams:
        for event in stream:
            if expected_run is not None and event.run_number != expected_run:
                raise RuntimeError(
                    f"{event.path} contains run {event.run_number}, expected {expected_run}"
                )
            if deduplicator is not None and deduplicator.process(event).duplicate:
                continue
            interval_index = bisect_right(starts, event.unix_time) - 1
            in_stable = (
                interval_index >= 0 and event.unix_time < stops[interval_index]
            )
            if not in_stable:
                previous_timestamp = None
                previous_interval = None
                continue
            events_in_stable += 1
            if (
                previous_timestamp is not None
                and previous_interval == interval_index
            ):
                delta_ticks = event.tlu_timestamp - previous_timestamp
                if delta_ticks > 0:
                    differences.append(delta_ticks / TLU_CLOCK_HZ)
                else:
                    skipped += 1
            previous_timestamp = event.tlu_timestamp
            previous_interval = interval_index
    return differences, events_in_stable, skipped


def read_rate_csv(path):
    """Read recorded fields from count CSVs, ignoring old correction columns."""
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            parsed = {}
            for field in (
                "start_utime", "stop_utime", "count", "rate_hz",
                "any_vetoed_count", "non_vetoed_count",
                "any_vetoed_rate_hz", "non_vetoed_rate_hz",
            ):
                value = row.get(field, "")
                parsed[field] = float(value) if value not in (None, "") else np.nan
            duration = parsed["stop_utime"] - parsed["start_utime"]
            if duration > 0:
                for count, rate in (("count", "rate_hz"),
                                    ("any_vetoed_count", "any_vetoed_rate_hz"),
                                    ("non_vetoed_count", "non_vetoed_rate_hz")):
                    if not np.isfinite(parsed[rate]) and np.isfinite(parsed[count]):
                        parsed[rate] = parsed[count] / duration
            any_rate = parsed["any_vetoed_rate_hz"]
            total = any_rate + parsed["non_vetoed_rate_hz"]
            parsed["any_vetoed_fraction"] = any_rate / total if total > 0 else np.nan
            rows.append(parsed)
    if not rows:
        raise ValueError(f"{path} contains no trigger-rate rows")
    return rows


def read_luminosity_csv(path, plot_stride=1):
    elapsed = []
    timestamps = []
    raw = []
    smoothed = []
    recorded_rate = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{path} contains no luminosity rows") from error
        required = (
            "timestamp_utc", "elapsed_s", "Recorded_rate_Hz",
            "luminosity_ub_inv_s", "luminosity_smoothed_ub_inv_s",
        )
        missing = [field for field in required if field not in header]
        if missing:
            raise ValueError(f"{path} does not contain {', '.join(missing)}")
        indices = {field: header.index(field) for field in required}
        time_offset = None
        for row_index, row in enumerate(reader):
            if row_index % plot_stride:
                continue
            try:
                elapsed_value = float(row[indices["elapsed_s"]])
            except (IndexError, ValueError):
                elapsed_value = np.nan
            if time_offset is None and np.isfinite(elapsed_value):
                try:
                    absolute = datetime.fromisoformat(
                        row[indices["timestamp_utc"]].replace("Z", "+00:00")
                    ).timestamp()
                    time_offset = absolute - elapsed_value
                except (IndexError, ValueError):
                    pass
            try:
                recorded_value = float(row[indices["Recorded_rate_Hz"]])
            except (IndexError, ValueError):
                recorded_value = np.nan
            try:
                raw_value = float(row[indices["luminosity_ub_inv_s"]])
            except (IndexError, ValueError):
                raw_value = np.nan
            try:
                smooth_value = float(
                    row[indices["luminosity_smoothed_ub_inv_s"]]
                )
            except (IndexError, ValueError):
                smooth_value = np.nan
            elapsed.append(elapsed_value)
            recorded_rate.append(recorded_value)
            raw.append(raw_value)
            smoothed.append(smooth_value)
        if time_offset is None:
            timestamps = np.full(len(elapsed), np.nan)
        else:
            timestamps = np.asarray(elapsed) + time_offset
    if not raw:
        raise ValueError(f"{path} contains no luminosity rows")
    return {
        "elapsed_s": np.asarray(elapsed),
        "timestamp_s": np.asarray(timestamps),
        "recorded_rate_hz": np.asarray(recorded_rate),
        "raw": np.asarray(raw),
        "smoothed": np.asarray(smoothed),
    }


def obtain_luminosity(args):
    path = args.luminosity_csv
    if path is None:
        candidate = DEFAULT_LUMINOSITY_CSV / f"run_{args.run}_timeseries.csv"
        path = candidate if candidate.exists() else None
    if path is None:
        command = [
            sys.executable,
            str(args.luminosity_script),
            "--run", str(args.run),
            "--bin-seconds", str(args.lumi_bin_seconds),
            "--lumi-plot-smooth-bins", str(args.lumi_smooth_bins),
            "--output-dir", str(args.output_dir),
        ]
        if args.lumi_measurement:
            command.extend(["--lumi-measurement", args.lumi_measurement])
        if args.lumi_field:
            command.extend(["--lumi-field", args.lumi_field])
        print("No luminosity CSV found; running:", " ".join(command))
        subprocess.run(command, check=True)
        path = args.output_dir / f"run_{args.run}_timeseries.csv"
    return read_luminosity_csv(path, args.lumi_plot_stride), path


def recorded_rate_from_luminosity_csv(path, args):
    samples = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        elapsed_index = header.index("elapsed_s")
        timestamp_index = header.index("timestamp_utc")
        rate_index = header.index("Recorded_rate_Hz")
        time_offset = None
        for row in reader:
            try:
                elapsed = float(row[elapsed_index])
                rate = float(row[rate_index])
            except (IndexError, ValueError):
                continue
            if not np.isfinite(rate):
                continue
            if time_offset is None:
                absolute = datetime.fromisoformat(
                    row[timestamp_index].replace("Z", "+00:00")
                ).timestamp()
                time_offset = absolute - elapsed
            samples.append((time_offset + elapsed, rate))
    if not samples:
        raise ValueError(f"{path} contains no valid Recorded_rate_Hz values")
    timestamps = np.asarray([item[0] for item in samples])
    spacings = np.diff(timestamps)
    positive_spacings = spacings[spacings > 0]
    bin_seconds = (
        float(np.median(positive_spacings))
        if len(positive_spacings)
        else args.lumi_bin_seconds
    )
    recorded_bins = [
        RateBin(
            start_utime=timestamp,
            stop_utime=timestamp + bin_seconds,
            count=max(0, int(round(rate * bin_seconds))),
            rate_hz=rate,
        )
        for timestamp, rate in samples
    ]
    return recorded_bins


def interval_legend(intervals):
    mean_ms = np.mean([item["deadtime_s"] * 1000.0 for item in intervals])
    range_ms = max(item["deadtime_error_s"] * 1000.0 for item in intervals)
    bin_ms = max(item["deadtime_bin_error_s"] * 1000.0 for item in intervals)
    average_ms = max(
        item["deadtime_smoothing_error_s"] * 1000.0 for item in intervals
    )
    total_ms = np.sqrt(range_ms**2 + bin_ms**2 + average_ms**2)
    return mean_ms, total_ms, range_ms, bin_ms, average_ms


def plot_deadtime(axis, centers, counts, intervals):
    populated = counts > 0
    axis.scatter(
        centers[populated], counts[populated], s=7, color="tab:blue",
        alpha=0.7, edgecolors="none",
    )
    for interval in intervals:
        edge_ranges = [("tstop_low_s", "tstop_high_s")]
        if np.isfinite(interval["tstart_s"]):
            edge_ranges.insert(0, ("tstart_low_s", "tstart_high_s"))
        for low_key, high_key in edge_ranges:
            axis.axvspan(
                interval[low_key], interval[high_key],
                color="tab:red", alpha=0.18,
            )
        if np.isfinite(interval["tstart_s"]):
            axis.axvline(
                interval["tstart_s"],
                color="tab:red", linestyle="--", linewidth=1,
            )
        axis.axvline(
            interval["tstop_s"], color="tab:red", linestyle="--", linewidth=1,
        )
    if intervals:
        handles = []
        for i, interval in enumerate(intervals, start=1):
            deadtime_ms = interval["deadtime_s"] * 1000.0
            error_ms = interval["deadtime_error_s"] * 1000.0
            sigma_ms = interval.get("deadtime_transition_width_s", 0.0) * 1000.0
            if np.isfinite(interval["tstart_s"]):
                label = (
                    f"Deadtime interval {i}: "
                    f"({deadtime_ms:.4f} +/- {error_ms:.4f} "
                    f"+/- {sigma_ms:.4f}) ms"
                )
            else:
                label = (
                    f"Initial deadtime:"
                    f"({deadtime_ms:.4f} +/- {error_ms:.4f} "
                    f"+/- {sigma_ms:.4f}) ms"
                )
            handles.append(
                Line2D(
                    [], [], color="tab:red", linestyle="--", linewidth=1,
                    label=label,
                )
            )
        
        handles.append(
            Patch(
                facecolor="tab:red",
                alpha=0.18,
                label="Edge range: mu +/- delta_mu +/- sigma",
            )
        )
        axis.legend(handles=handles, loc="upper right", fontsize=9)
      
    axis.set_yscale("log")
    axis.set_ylim(1e-1, max(1e5, float(np.max(counts)) * 2.0))
    axis.set_xlabel("Time difference between consecutive events [s]")
    axis.set_ylabel("Number of event intervals")
    axis.set_title("TLU deadtime (stable-beam intervals)")
    axis.grid(True, alpha=0.3)


def parse_manual_erf_ranges(values):
    """Convert repeated LEFT_MS RIGHT_MS DIRECTION arguments to seconds."""
    if not values:
        return None
    direction_values = {
        "rising": 1,
        "falling": -1,
        "+1": 1,
        "-1": -1,
    }
    ranges = []
    for index, (left_text, right_text, direction_text) in enumerate(
        values, start=1
    ):
        try:
            left_ms = float(left_text)
            right_ms = float(right_text)
        except ValueError as error:
            raise ValueError(
                f"invalid --erf-fit-range {index}: boundaries must be numbers"
            ) from error
        direction_key = direction_text.lower()
        if direction_key not in direction_values:
            raise ValueError(
                f"invalid --erf-fit-range {index}: direction must be "
                "rising, falling, +1, or -1"
            )
        if (
            not np.isfinite(left_ms)
            or not np.isfinite(right_ms)
            or left_ms >= right_ms
        ):
            raise ValueError(
                f"invalid --erf-fit-range {index}: LEFT_MS must be smaller "
                "than RIGHT_MS"
            )
        ranges.append(
            (left_ms / 1000.0, right_ms / 1000.0, direction_values[direction_key])
        )
    return ranges


def plot_erf_edge_fits(
    centers,
    counts,
    intervals,
    output,
    dpi=150,
    manual_ranges=None,
    statistic="poisson",
    run_number=None,
):
    """Plot manually selected or slope-selected linear-count erf fits."""
    edges = []
    if manual_ranges is not None:
        for fit_index, (fit_left, fit_right, direction) in enumerate(
            manual_ranges, start=1
        ):
            side = "rising" if direction == 1 else "falling"
            edges.append(
                (
                    fit_index,
                    side,
                    (fit_left + fit_right) / 2.0,
                    direction,
                    fit_left,
                    fit_right,
                )
            )
        title_prefix = "Manual"
    else:
        for interval_index, interval in enumerate(intervals, start=1):
            if np.isfinite(interval["tstart_s"]):
                edges.append(
                    (
                        interval_index,
                        "start",
                        interval["tstart_s"],
                        -1,
                        interval["left_fit_start_s"],
                        interval["left_fit_stop_s"],
                    )
                )
            edges.append(
                (
                    interval_index,
                    "stop",
                    interval["tstop_s"],
                    1,
                    interval["right_fit_start_s"],
                    interval["right_fit_stop_s"],
                )
            )
        title_prefix = "Deadtime"

    if run_number is not None:
        title_prefix = f"Run {run_number}: {title_prefix}"

    if not edges:
        return []

    fit_specs = [
        {
            "center_guess": edge[2],
            "direction": edge[3],
            "fit_left": edge[4],
            "fit_right": edge[5],
        }
        for edge in edges
    ]
    try:
        (
            all_x,
            all_y,
            prepared_edges,
            independent_fits,
            fit_summary,
        ) = fit_erf_edges_independent_likelihood(
            centers, counts, fit_specs, statistic=statistic
        )
        draw_erf_fits_root(
            all_x,
            all_y,
            prepared_edges,
            independent_fits,
            fit_summary,
            output,
            title_prefix=title_prefix,
        )
        for edge, fit in zip(edges, independent_fits):
            warnings = []
            if not fit["fit_converged"]:
                warnings.append(
                    f"Minuit status {fit['minuit_status']}, EDM={fit['edm']:.3g}"
                )
            if fit["center_at_boundary"]:
                warnings.append("center at fit-range boundary")
            warning_text = (
                f" [WARNING: {', '.join(warnings)}]" if warnings else ""
            )
            print(
                f"Independent erf fit {edge[0]} {edge[1]}: "
                f"center = {fit['position_s'] * 1000.0:.6f} ms, "
                f"width = {fit['transition_width_s'] * 1.0e6:.3f} +/- "
                f"{fit['transition_width_error_s'] * 1.0e6:.3f} us, "
                f"{fit['statistic_label']}/ndf = "
                f"{fit['fit_statistic_ndf']:.2f}"
                f"{warning_text}"
            )
        return [
            {
                "interval_index": edge[0],
                "edge": edge[1],
                "status": "ok",
                **fit,
            }
            for edge, fit in zip(edges, independent_fits)
        ]
    except (RuntimeError, ValueError) as error:
        print(f"Independent error-function fit failed: {error}")
        return [
            {
                "interval_index": edge[0],
                "edge": edge[1],
                "status": "failed",
                "error": str(error),
            }
            for edge in edges
        ]


def _validate_erf_edge_fit(fit, expected_direction):
    """Validate one fitted edge before using it as a dead-time boundary."""
    if fit.get("status") != "ok":
        raise RuntimeError(
            f"erf fit {fit.get('interval_index', '?')} "
            f"{fit.get('edge', '?')} failed: {fit.get('error', 'unknown error')}"
        )
    if fit.get("direction") != expected_direction:
        expected = "rising" if expected_direction == 1 else "falling"
        raise ValueError(
            f"erf fit {fit.get('interval_index', '?')} must be {expected}"
        )
    if not fit.get("fit_converged", False):
        raise RuntimeError(
            f"erf fit {fit.get('interval_index', '?')} did not converge "
            f"(Minuit status {fit.get('minuit_status', '?')})"
        )
    position = float(fit["position_s"])
    error = float(fit["position_error_s"])
    if not np.isfinite(position) or not np.isfinite(error) or error < 0.0:
        raise RuntimeError(
            f"erf fit {fit.get('interval_index', '?')} returned an invalid center"
        )


def _deadtime_interval_from_erf_fits(start_fit, stop_fit):
    """Build one dead-time interval from fitted erf center positions."""
    _validate_erf_edge_fit(stop_fit, 1)
    missing = float("nan")
    stop = float(stop_fit["position_s"])
    stop_error = float(stop_fit["position_error_s"])
    stop_width = float(stop_fit["transition_width_s"])

    if start_fit is None:
        start = missing
        start_error = missing
        deadtime = stop
        deadtime_error = stop_error
        deadtime_width = stop_width
        start_width = missing
        left_fit_start = missing
        left_fit_stop = missing
        left_fit_bins = 0
        start_statistic_ndf = missing
        start_converged = ""
    else:
        _validate_erf_edge_fit(start_fit, -1)
        start = float(start_fit["position_s"])
        start_error = float(start_fit["position_error_s"])
        deadtime = stop - start
        deadtime_error = float(np.hypot(start_error, stop_error))
        start_width = float(start_fit["transition_width_s"])
        # The absolute edge widths are cumulative.  The incremental physical
        # dead-time variance is therefore the later (rising) edge variance
        # minus the earlier (falling) edge variance.  A negative result is
        # inconsistent with this model and must not be folded positive.
        deadtime_variance = stop_width**2 - start_width**2
        deadtime_width = (
            float(np.sqrt(deadtime_variance))
            if deadtime_variance >= 0.0
            else missing
        )
        left_fit_start = float(start_fit["fit_start_s"])
        left_fit_stop = float(start_fit["fit_stop_s"])
        left_fit_bins = int(start_fit["fit_bins"])
        start_statistic_ndf = float(start_fit["fit_statistic_ndf"])
        start_converged = bool(start_fit["fit_converged"])

    if deadtime <= 0.0:
        raise RuntimeError(f"erf fits returned non-positive dead time {deadtime} s")

    return {
        "deadtime_method": "erf_fit",
        "fit_statistic": stop_fit["statistic"],
        "tstart_s": start,
        "tstart_error_s": start_error,
        "tstart_transition_width_s": start_width,
        "tstart_low_s": start - start_error - start_width,
        "tstart_high_s": start + start_error + start_width,
        "tstop_s": stop,
        "tstop_error_s": stop_error,
        "tstop_transition_width_s": stop_width,
        "tstop_low_s": stop - stop_error - stop_width,
        "tstop_high_s": stop + stop_error + stop_width,
        "deadtime_s": deadtime,
        "deadtime_error_s": deadtime_error,
        "deadtime_transition_width_s": deadtime_width,
        "deadtime_bin_error_s": 0.0,
        "deadtime_smoothing_error_s": 0.0,
        "deadtime_total_error_s": deadtime_error,
        "deadtime_low_s": deadtime - deadtime_error - deadtime_width,
        "deadtime_high_s": deadtime + deadtime_error + deadtime_width,
        "left_fit_start_s": left_fit_start,
        "left_fit_stop_s": left_fit_stop,
        "right_fit_start_s": float(stop_fit["fit_start_s"]),
        "right_fit_stop_s": float(stop_fit["fit_stop_s"]),
        "left_fit_bins": left_fit_bins,
        "right_fit_bins": int(stop_fit["fit_bins"]),
        "left_fit_indices": np.asarray([], dtype=int),
        "right_fit_indices": np.asarray([], dtype=int),
        "tstart_fit_statistic_ndf": start_statistic_ndf,
        "tstop_fit_statistic_ndf": float(stop_fit["fit_statistic_ndf"]),
        "tstart_fit_converged": start_converged,
        "tstop_fit_converged": bool(stop_fit["fit_converged"]),
    }


def deadtime_intervals_from_erf_fits(erf_fits, manual_ranges=None):
    """Convert fitted edges to an initial interval plus falling/rising pairs."""
    if not erf_fits:
        raise RuntimeError("no erf edge fits are available for dead-time calculation")

    if manual_ranges is not None:
        directions = [item[2] for item in manual_ranges]
        offset = 0
        intervals = []
        if directions[0] == 1:
            intervals.append(_deadtime_interval_from_erf_fits(None, erf_fits[0]))
            offset = 1
        remaining = len(erf_fits) - offset
        if remaining % 2:
            raise ValueError(
                "manual erf ranges must be an optional initial rising edge "
                "followed by falling/rising pairs"
            )
        for index in range(offset, len(erf_fits), 2):
            intervals.append(
                _deadtime_interval_from_erf_fits(
                    erf_fits[index], erf_fits[index + 1]
                )
            )
        return intervals

    grouped = {}
    for fit in erf_fits:
        grouped.setdefault(fit["interval_index"], {})[fit["edge"]] = fit
    intervals = []
    for interval_index in sorted(grouped):
        edges = grouped[interval_index]
        if "stop" not in edges:
            raise RuntimeError(f"interval {interval_index} has no rising stop fit")
        intervals.append(
            _deadtime_interval_from_erf_fits(edges.get("start"), edges["stop"])
        )
    return intervals


def limit_deadtime_intervals(intervals, max_intervals, manual_ranges=None):
    """Limit auto-detected intervals while preserving every manual fit pair."""
    if manual_ranges is not None:
        return list(intervals)
    return list(intervals[:max_intervals])


def _rate_value(item, field):
    if isinstance(item, dict):
        value = item.get(field, np.nan)
    else:
        value = getattr(item, field, np.nan)
    return np.nan if value is None else value


def _trigger_rate_arrays(bins):
    start = np.asarray([_rate_value(item, "start_utime") for item in bins])
    time_origin = start[0]
    elapsed = start - time_origin
    series = (
        ("rate_hz", "tab:blue", "Recorded total rate"),
        ("any_vetoed_rate_hz", "tab:green", "Recorded AnyVetoed rate"),
        ("non_vetoed_rate_hz", "tab:orange", "Recorded NonVetoed rate"),
    )
    return elapsed, time_origin, series


def _plot_trigger_rate_series(axis, bins, elapsed, series):
    for field, color, label in series:
        values = np.asarray([_rate_value(item, field) for item in bins])
        valid = np.isfinite(values)
        if np.any(valid):
            axis.plot(
                elapsed[valid], values[valid], color=color,
                linewidth=1.3, label=label,
            )


def plot_trigger_rate(axis, bins, luminosity, lumi_smooth_bins):
    """Plot recorded total/veto-class rates and luminosity against time."""
    elapsed, time_origin, series = _trigger_rate_arrays(bins)
    _plot_trigger_rate_series(axis, bins, elapsed, series)

    lumi_axis = axis.twinx()
    lumi_elapsed = luminosity["timestamp_s"] - time_origin
    if not np.any(np.isfinite(lumi_elapsed)):
        lumi_elapsed = luminosity["elapsed_s"]
    smooth_valid = (
        np.isfinite(lumi_elapsed)
        & np.isfinite(luminosity["smoothed"])
        & (luminosity["smoothed"] > 0)
    )
    lumi_axis.plot(
        lumi_elapsed[smooth_valid], luminosity["smoothed"][smooth_valid],
        color="tab:red", linewidth=1.8, alpha=0.95,
        label=f"Luminosity ({lumi_smooth_bins}-bin moving average)",
    )
    lumi_axis.set_yscale("linear")
    lumi_axis.set_ylim(0.0, 40000.0)

    axis.set_xlabel("Elapsed time [s]")
    axis.set_ylabel("Trigger rate [Hz]", color="tab:blue")
    lumi_axis.set_ylabel("Luminosity [ub^-1 s^-1]", color="tab:red")
    axis.tick_params(axis="y", colors="tab:blue")
    lumi_axis.tick_params(axis="y", colors="tab:red")
    axis.set_title("Trigger rate and luminosity")
    axis.grid(True, alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    lumi_handles, lumi_labels = lumi_axis.get_legend_handles_labels()
    axis.legend(
        handles + lumi_handles,
        labels + lumi_labels,
        loc="upper right",
    )


def plot_trigger_rate_and_any_vetoed_fraction(rate_axis, ratio_axis, bins):
    """Plot recorded rates above their raw-derived AnyVetoed fraction."""
    elapsed, _time_origin, series = _trigger_rate_arrays(bins)
    _plot_trigger_rate_series(rate_axis, bins, elapsed, series)

    fraction = np.asarray(
        [_rate_value(item, "any_vetoed_fraction") for item in bins],
        dtype=float,
    )
    fraction_valid = np.isfinite(fraction)
    if np.any(fraction_valid):
        ratio_axis.plot(
            elapsed[fraction_valid],
            fraction[fraction_valid],
            color="tab:brown",
            linewidth=1.4,
            label="AnyVetoed / (AnyVetoed + NonVetoed)",
        )
        ratio_top = min(
            1.0,
            max(0.05, 1.15 * float(np.nanpercentile(fraction, 99.0))),
        )
        ratio_axis.set_ylim(0.0, ratio_top)

    rate_axis.set_ylabel("Trigger rate [Hz]")
    rate_axis.set_title("Trigger rates")
    rate_axis.grid(True, alpha=0.3)
    rate_axis.legend(loc="upper right")
    ratio_axis.set_xlabel("Elapsed time [s]")
    ratio_axis.set_ylabel("AnyVetoed fraction")
    ratio_axis.set_title("AnyVetoed ratio")
    ratio_axis.grid(True, alpha=0.3)
    if np.any(fraction_valid):
        ratio_axis.legend(loc="upper right")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "-r", type=int, required=True)
    parser.add_argument("--raw-pattern", action="append")
    parser.add_argument("--histogram-csv", type=Path)
    parser.add_argument(
        "--run-intervals-json", type=Path, default=DEFAULT_RUN_INTERVALS,
    )
    parser.add_argument(
        "--all-run-deadtime", action="store_true",
        help="use the full run instead of stable_list for the TLU histogram",
    )
    parser.add_argument("--trigger-rate-csv", type=Path)
    parser.add_argument("--luminosity-csv", type=Path)
    parser.add_argument(
        "--luminosity-script", type=Path, default=DEFAULT_LUMINOSITY_SCRIPT,
    )
    parser.add_argument("--lumi-bin-seconds", type=float, default=0.2)
    parser.add_argument("--lumi-smooth-bins", type=int, default=11)
    parser.add_argument(
        "--lumi-plot-stride", type=int, default=20,
        help="plot every Nth luminosity row; calculations in Luminosity.py are unchanged",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--lumi-measurement")
    parser.add_argument("--lumi-field", default="Luminosity")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bins", type=int,
        help=(
            "number of deadtime histogram bins; when specified, rebuild the "
            "histogram from raw files instead of reusing an existing CSV "
            "(default when building: 5000)"
        ),
    )
    parser.add_argument("--max-difference", type=float, default=PLOT_MAX_DIFFERENCE)
    parser.add_argument("--deadtime-search-min-ms", type=float, default=2.0)
    parser.add_argument("--deadtime-search-max-ms", type=float, default=20.0)
    parser.add_argument("--deadtime-smooth-bins", type=int, default=9)
    parser.add_argument("--deadtime-edge-fraction", type=float, default=0.45)
    parser.add_argument("--deadtime-min-width-ms", type=float, default=0.2)
    parser.add_argument("--deadtime-max-width-ms", type=float, default=2.0)
    parser.add_argument("--deadtime-min-log-depth", type=float, default=0.7)
    parser.add_argument(
        "--erf-fit-range",
        action="append",
        nargs=3,
        metavar=("LEFT_MS", "RIGHT_MS", "DIRECTION"),
        help=(
            "manually fit one linear-count edge over LEFT_MS..RIGHT_MS; "
            "DIRECTION is rising or falling; may be repeated"
        ),
    )
    parser.add_argument(
        "--erf-fit-statistic",
        choices=("poisson", "chi2"),
        default="poisson",
        help=(
            "objective used for dead-time erf fits: binned Poisson likelihood "
            "or Pearson chi-square (default: poisson)"
        ),
    )
    parser.add_argument(
        "--deadtime-max-intervals", type=int, default=3,
        help=(
            "maximum number of automatically detected deadtime intervals to "
            "keep (default: 3); manual --erf-fit-range intervals are all kept"
        ),
    )
    parser.add_argument("--bin-seconds", type=float, default=2.0)
    parser.add_argument(
        "--trigger-source", choices=("auto", "raw"), default="raw",
        help=(
            "raw calculates the recorded trigger rate from raw event files "
            "(default); auto reuses a trigger CSV or Luminosity.py "
            "Recorded_rate_Hz"
        ),
    )
    parser.add_argument(
        "--trigger-order", choices=("merged", "external-sort"),
        default="merged",
        help=(
            "merged heap-merges timestamp-ordered raw files (fast); "
            "external-sort performs a full disk sort (slow fallback)"
        ),
    )
    parser.add_argument("--recent-id-limit", type=int, default=100_000)
    parser.add_argument("--reorder-tolerance", type=int, default=1024)
    return parser.parse_args(argv)


def main(args=None):
    args = parse_args() if args is None else args
    selection_intervals = getattr(args, "selection_intervals", None)
    manual_erf_ranges = parse_manual_erf_ranges(args.erf_fit_range)
    if args.bins is not None and args.bins <= 0:
        raise ValueError("--bins must be positive")
    if args.bins is not None and args.histogram_csv is not None:
        raise ValueError(
            "--bins cannot be combined with --histogram-csv because a "
            "binned histogram cannot be exactly rebinned"
        )
    if args.lumi_plot_stride <= 0:
        raise ValueError("--lumi-plot-stride must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.deadtime_max_intervals <= 0:
        raise ValueError("--deadtime-max-intervals must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patterns = args.raw_pattern or [DEFAULT_RAW_PATTERN]
    paths = None

    rebuild_histogram = args.bins is not None
    histogram_csv = args.histogram_csv
    if histogram_csv is None and not rebuild_histogram:
        cache_name = (
            f"run_{args.run}_deadtime_histogram.csv"
            if args.all_run_deadtime
            else f"run_{args.run}_stable_deadtime_histogram.csv"
        )
        candidate = args.output_dir / cache_name
        histogram_csv = candidate if candidate.exists() else None
    if histogram_csv is not None:
        centers, counts = read_histogram_csv(histogram_csv)
        print(f"Reusing deadtime histogram: {histogram_csv}")
    else:
        paths = discover_raw_files(args.run, patterns)
        if not paths:
            raise RuntimeError(f"no raw files found for run {args.run}")
        if args.all_run_deadtime:
            differences, event_count, skipped = read_tlu_time_differences(
                paths, expected_run=args.run
            )
            interval_description = "full run"
        else:
            stable_intervals = selection_intervals if selection_intervals is not None else read_run_intervals(
                args.run_intervals_json, args.run, "stable_list"
            )
            differences, event_count, skipped = read_stable_tlu_time_differences(
                paths, stable_intervals, expected_run=args.run,
                merge_files=selection_intervals is not None,
            )
            interval_description = (
                f"{len(stable_intervals)} stable interval(s)"
            )
        if not differences:
            raise RuntimeError(
                f"no positive TLU differences found in {interval_description}"
            )
        print(
            f"Selected {event_count} TLU events in {interval_description}; "
            f"skipped {skipped} non-positive differences"
        )
        histogram_bins = args.bins if args.bins is not None else 5000
        counts, edges = np.histogram(
            differences, bins=histogram_bins, range=(0, args.max_difference),
        )
        centers = (edges[:-1] + edges[1:]) / 2.0
        histogram_name = (
            f"run_{args.run}_deadtime_histogram.csv"
            if args.all_run_deadtime
            else f"run_{args.run}_stable_deadtime_histogram.csv"
        )
        histogram_csv = args.output_dir / histogram_name
        write_histogram_csv(histogram_csv, centers, counts)
        print(
            f"Rebuilt deadtime histogram with {histogram_bins} bins: "
            f"{histogram_csv}"
        )

    initial_interval = find_initial_deadtime_interval(
        centers, counts,
        search_max_s=args.deadtime_search_min_ms / 1000.0,
        smooth_bins=args.deadtime_smooth_bins,
        edge_fraction=args.deadtime_edge_fraction,
        min_stop_s=args.deadtime_min_width_ms / 1000.0,
        max_stop_s=args.deadtime_search_min_ms / 1000.0,
        min_log_depth=args.deadtime_min_log_depth,
    )
    intervals = find_deadtime_intervals(
        centers, counts,
        search_min_s=args.deadtime_search_min_ms / 1000.0,
        search_max_s=args.deadtime_search_max_ms / 1000.0,
        smooth_bins=args.deadtime_smooth_bins,
        edge_fraction=args.deadtime_edge_fraction,
        min_width_s=args.deadtime_min_width_ms / 1000.0,
        max_width_s=args.deadtime_max_width_ms / 1000.0,
        min_log_depth=args.deadtime_min_log_depth,
    )
    if initial_interval is not None:
        intervals.insert(0, initial_interval)
    intervals = limit_deadtime_intervals(
        intervals,
        args.deadtime_max_intervals,
        manual_ranges=manual_erf_ranges,
    )
    erf_plot = args.output_dir / f"run_{args.run}_deadtime_erf_fit.png"
    erf_fits = plot_erf_edge_fits(
        centers,
        counts,
        intervals,
        erf_plot,
        dpi=args.dpi,
        manual_ranges=manual_erf_ranges,
        statistic=args.erf_fit_statistic,
        run_number=args.run,
    )
    failed_erf_fits = [item for item in erf_fits if item["status"] != "ok"]
    if failed_erf_fits:
        print(
            f"Error-function fits failed for {len(failed_erf_fits)} "
            f"of {len(erf_fits)} edges; fit plot was not updated"
        )
    else:
        print(f"Saved error-function edge fits to {erf_plot}")
        unconverged_erf_fits = [
            item for item in erf_fits if not item.get("fit_converged", True)
        ]
        if unconverged_erf_fits:
            print(
                f"Warning: {len(unconverged_erf_fits)} of {len(erf_fits)} "
                "edge fits have a non-zero Minuit status"
            )

    intervals = limit_deadtime_intervals(
        deadtime_intervals_from_erf_fits(
            erf_fits, manual_ranges=manual_erf_ranges
        ),
        args.deadtime_max_intervals,
        manual_ranges=manual_erf_ranges,
    )
    deadtime_csv = args.output_dir / f"run_{args.run}_deadtime.csv"
    write_deadtime_csv(deadtime_csv, args.run, intervals)
    print(
        "Using erf-fit centers for dead-time intervals: "
        + ", ".join(f"{item['deadtime_s'] * 1000.0:.6f} ms" for item in intervals)
    )

    luminosity, luminosity_csv = obtain_luminosity(args)

    rate_csv = args.trigger_rate_csv
    if rate_csv is None and args.trigger_source == "auto":
        candidate = RAW_RATE_DIR / f"run{args.run}_trigger_rate.csv"
        rate_csv = candidate if candidate.exists() else None
    if rate_csv is not None:
        rate_bins = read_rate_csv(rate_csv)
        rate_stats = None
        rate_source = str(rate_csv)
    elif args.trigger_source == "auto":
        rate_bins = recorded_rate_from_luminosity_csv(luminosity_csv, args)
        rate_stats = None
        rate_source = (
            f"{luminosity_csv}:Recorded_rate_Hz"
        )
    else:
        if paths is None:
            paths = discover_raw_files(args.run, patterns)
        if not paths:
            raise RuntimeError(f"no raw files found for run {args.run}")
        rate_bins, rate_stats = calculate_rate_rows(paths, args, intervals=selection_intervals)
        if not rate_bins:
            raise RuntimeError("no raw trigger-rate coverage in the selected intervals")
        rate_csv = args.output_dir / f"run_{args.run}_trigger_rate.csv"
        write_rate_csv(rate_csv, rate_bins)
        rate_source = str(rate_csv)

    figure, axes = plt.subplots(2, 1, figsize=(12, 12))
    plot_deadtime(axes[0], centers, counts, intervals)
    axes[0].set_xlim(0, args.deadtime_search_max_ms / 1000.0)
    plot_trigger_rate(axes[1], rate_bins, luminosity, args.lumi_smooth_bins)
    figure.suptitle(
        f"AHCAL run {args.run}: deadtime, trigger rate, and luminosity",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output = args.output_dir / f"run_{args.run}_deadtime_trigger_rate_luminosity.png"
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)
    print(f"Saved combined plot to {output}")

    rate_figure, rate_axes = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    plot_trigger_rate_and_any_vetoed_fraction(
        rate_axes[0], rate_axes[1], rate_bins,
    )
    rate_figure.suptitle(
        f"AHCAL run {args.run}: trigger rates and AnyVetoed ratio",
        fontsize=15,
    )
    rate_figure.tight_layout(rect=(0, 0, 1, 0.96))
    rate_ratio_output = (
        args.output_dir / f"run_{args.run}_trigger_rate_any_vetoed_ratio.png"
    )
    rate_figure.savefig(rate_ratio_output, dpi=args.dpi)
    plt.close(rate_figure)
    print(f"Saved trigger-rate and AnyVetoed-ratio plot to {rate_ratio_output}")
    print(f"Saved deadtime intervals to {deadtime_csv}")
    print(f"Trigger-rate source: {rate_source}")
    print(f"Luminosity input/output: {luminosity_csv}")
    if rate_stats is not None:
        print(
            f"Processed {rate_stats.unique_triggers} unique and "
            f"{rate_stats.duplicate_triggers} duplicate triggers"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
