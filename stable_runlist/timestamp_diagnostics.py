#!/usr/bin/env python3
"""TLU/CPU timestamp diagnostics and deadtime-edge analysis from AHCAL raw files."""

from .config import DIAGNOSTICS_DIR

import argparse
import csv
import glob
import sys
from pathlib import Path
from scipy.special import erf
from scipy.optimize import curve_fit, least_squares

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


from .ahcal_raw_reader import RawFileReader  # noqa: E402


TLU_CLOCK_HZ = 40_000_000.0
PLOT_MAX_DIFFERENCE = 5e-2
DEFAULT_RAW_PATTERN = (
    "/eos/experiment/faser/raw/*/{run06}/"
    "FaserAHCAL-Physics-{run06}-*.raw"
)


def discover_raw_files(run, patterns):
    paths = []
    for pattern in patterns:
        try:
            expanded = pattern.format(run=run, run06=f"{run:06d}")
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(
                f"Invalid raw pattern {pattern!r}; only {{run}} and {{run06}} "
                "are supported"
            ) from error
        paths.extend(Path(path) for path in glob.glob(expanded))
    return sorted({path.resolve() for path in paths if path.is_file()})


def _read_time_differences(paths, attribute, clock_hz, expected_run=None, verbose=False):
    """Read consecutive positive differences in a raw timestamp clock."""
    differences = []
    previous_timestamp = None
    events = 0
    skipped = 0

    for path in paths:
        reader = RawFileReader(path)
        for event in reader:
            if expected_run is not None and event.run_number != expected_run:
                raise RuntimeError(
                    f"{path} contains run {event.run_number}, expected {expected_run}"
                )
            events += 1
            if previous_timestamp is not None:
                delta_ticks = getattr(event, attribute) - previous_timestamp
                if delta_ticks > 0:
                    differences.append(delta_ticks / clock_hz)
                else:
                    # Do not turn duplicated/out-of-order timestamps into
                    # non-physical negative time differences.
                    skipped += 1
            previous_timestamp = getattr(event, attribute)
        if verbose:
            print(
                f"Read {path}: {reader.stats.events_with_tlu} events with TLU, "
                f"{reader.stats.events_without_tlu} without TLU"
            )

    return differences, events, skipped


def read_tlu_time_differences(paths, expected_run=None, verbose=False):
    """Read positive consecutive TLU timestamp differences in seconds."""
    return _read_time_differences(paths, "tlu_timestamp", TLU_CLOCK_HZ, expected_run, verbose)


def read_cpu_time_differences(paths, expected_run=None, verbose=False):
    """Read positive consecutive CPU timestamp differences in seconds."""
    return _read_time_differences(paths, "cpu_timestamp_us", 1_000_000.0, expected_run, verbose)


def read_tlu_time_differences_csv(path, expected_run=None):
    """Read consecutive TLU timestamp differences from an event-level CSV."""
    differences = []
    previous_timestamp = None
    events = 0
    skipped = 0
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fields = set(reader.fieldnames or ())
        if "tlu_timestamp" not in fields:
            raise ValueError(
                f"{path} has no tlu_timestamp column; a trigger-rate CSV cannot "
                "be used because it does not contain event-level timestamps"
            )
        for row in reader:
            if expected_run is not None and row.get("run_number"):
                run_number = int(row["run_number"])
                if run_number != expected_run:
                    raise RuntimeError(
                        f"{path} contains run {run_number}, expected {expected_run}"
                    )
            timestamp = int(row["tlu_timestamp"])
            events += 1
            if previous_timestamp is not None:
                delta_ticks = timestamp - previous_timestamp
                if delta_ticks > 0:
                    differences.append(delta_ticks / TLU_CLOCK_HZ)
                else:
                    skipped += 1
            previous_timestamp = timestamp
    return differences, events, skipped


def write_histogram_csv(path, bin_centers, counts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("time_difference_s", "count"))
        writer.writerows(zip(bin_centers, counts))


def read_histogram_csv(path):
    centers = []
    counts = []
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fields = set(reader.fieldnames or ())
        required = {"time_difference_s", "count"}
        if not required.issubset(fields):
            raise ValueError(
                f"{path} must contain time_difference_s and count columns"
            )
        for row in reader:
            centers.append(float(row["time_difference_s"]))
            counts.append(int(row["count"]))
    if not centers:
        raise ValueError(f"{path} contains no histogram rows")
    return np.asarray(centers), np.asarray(counts)


def _smooth_log_counts(counts, width):
    values = np.log(np.asarray(counts, dtype=float) + 1.0)
    half_width = width // 2
    padded = np.pad(values, (half_width, half_width), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _fit_vertical_edge(x, slope, peak_index, direction, fraction):
    """Bracket an edge with bins selected by a slope threshold."""
    strength = direction * slope
    threshold = fraction * strength[peak_index]
    start = peak_index
    stop = peak_index
    while start > 0 and strength[start - 1] >= threshold:
        start -= 1
    while stop + 1 < len(x) and strength[stop + 1] >= threshold:
        stop += 1

    edge_x = x[start : stop + 1]
    bin_width = float(np.median(np.diff(x)))
    position_low = float(edge_x[0])
    position_high = float(edge_x[-1])
    position = (position_low + position_high) / 2.0
    # Half-width of the slope-selected interval, not a statistical deviation.
    position_error = (position_high - position_low) / 2.0
    return {
        "position_s": position,
        "position_error_s": float(position_error),
        "position_low_s": position_low,
        "position_high_s": position_high,
        "fit_start_s": float(edge_x[0]),
        "fit_stop_s": float(edge_x[-1]),
        "fit_bins": int(len(edge_x)),
        "indices": np.arange(start, stop + 1),
    }


def fit_erf_edge(
    x,
    y,
    center_guess,
    direction,
    fit_half_width=0.00003,
    fit_left=None,
    fit_right=None,
):
    """Fit one histogram edge with an error function on linear counts.

    Direction is +1 for a rising edge and -1 for a falling edge. Counts are
    assigned Poisson uncertainties sqrt(max(count, 1)). If fit_left and
    fit_right are given, they define the fit range; otherwise the range is
    center_guess +/- fit_half_width.
    """
    all_x = np.asarray(x, dtype=float)
    all_y = np.asarray(y, dtype=float)
    if all_x.ndim != 1 or all_y.ndim != 1 or all_x.shape != all_y.shape:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 (rising) or -1 (falling)")
    if not np.isfinite(center_guess):
        raise ValueError("center_guess must be finite")
    if (fit_left is None) != (fit_right is None):
        raise ValueError("fit_left and fit_right must be given together")
    if fit_left is None:
        if not np.isfinite(fit_half_width) or fit_half_width <= 0:
            raise ValueError("fit_half_width must be positive and finite")
        range_left = center_guess - fit_half_width
        range_right = center_guess + fit_half_width
    else:
        range_left = float(fit_left)
        range_right = float(fit_right)
        if (
            not np.isfinite(range_left)
            or not np.isfinite(range_right)
            or range_left >= range_right
        ):
            raise ValueError("fit_left and fit_right must define a finite range")

    selected = (
        np.isfinite(all_x)
        & np.isfinite(all_y)
        & (all_y >= 0.0)
        & (all_x >= range_left)
        & (all_x <= range_right)
    )
    selected_indices = np.flatnonzero(selected)
    fit_x = all_x[selected]
    fit_y = all_y[selected]
    if len(fit_x) < 5:
        raise ValueError(
            "an error-function edge fit needs at least 5 histogram bins; "
            f"selected {len(fit_x)}"
        )
    if np.any(np.diff(fit_x) <= 0.0):
        raise ValueError("x values in the fit range must be strictly increasing")

    bin_width = float(np.median(np.diff(fit_x)))
    side_bins = max(1, min(3, len(fit_x) // 3))
    left_level = float(np.median(fit_y[:side_bins]))
    right_level = float(np.median(fit_y[-side_bins:]))
    if direction == 1:
        baseline_guess = left_level
        amplitude_guess = right_level - left_level
    else:
        baseline_guess = right_level
        amplitude_guess = left_level - right_level
    baseline_guess = max(0.0, baseline_guess)
    amplitude_guess = max(amplitude_guess, float(np.ptp(fit_y)), 1.0)

    # Fit the center as an offset from the guess for stable conditioning.
    relative_x = fit_x - center_guess

    def model(relative_time, baseline, amplitude, center_offset, edge_width):
        argument = (relative_time - center_offset) / (
            np.sqrt(2.0) * edge_width
        )
        return baseline + 0.5 * amplitude * (
            1.0 + direction * erf(argument)
        )

    fit_range_width = range_right - range_left
    initial_width = max(bin_width, fit_range_width / 6.0)
    count_errors = np.sqrt(np.maximum(fit_y, 1.0))
    parameters, covariance = curve_fit(
        model,
        relative_x,
        fit_y,
        p0=(baseline_guess, amplitude_guess, 0.0, initial_width),
        sigma=count_errors,
        absolute_sigma=True,
        bounds=(
            (0.0, 0.0, relative_x[0], bin_width / 100.0),
            (np.inf, np.inf, relative_x[-1], fit_range_width),
        ),
        maxfev=20_000,
    )

    baseline, amplitude, center_offset, edge_width = parameters
    center = float(center_guess + center_offset)
    center_error = float(np.sqrt(max(covariance[2, 2], 0.0)))
    fitted_y = model(relative_x, *parameters)
    residuals = (fit_y - fitted_y) / count_errors
    chi2 = float(np.sum(residuals**2))
    ndf = int(len(fit_x) - len(parameters))
    return {
        "position_s": center,
        "position_error_s": center_error,
        "position_low_s": center - center_error,
        "position_high_s": center + center_error,
        "transition_width_s": float(edge_width),
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "chi2": chi2,
        "ndf": ndf,
        "chi2_ndf": chi2 / ndf if ndf > 0 else float("nan"),
        "fit_start_s": float(fit_x[0]),
        "fit_stop_s": float(fit_x[-1]),
        "fit_bins": int(len(fit_x)),
        "indices": selected_indices,
        "fit_x": fit_x,
        "fit_y": fitted_y,
        "covariance": covariance,
    }


def fit_erf_edges_shared_width(x, y, edge_specs):
    """Fit several error-function edges with one shared transition width."""
    all_x = np.asarray(x, dtype=float)
    all_y = np.asarray(y, dtype=float)
    if all_x.ndim != 1 or all_y.ndim != 1 or all_x.shape != all_y.shape:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if not edge_specs:
        raise ValueError("at least one edge is required for a shared-width fit")

    prepared = []
    bin_widths = []
    range_widths = []
    initial_parameters = [0.0]
    lower_bounds = [0.0]
    upper_bounds = [0.0]

    for edge_index, spec in enumerate(edge_specs, start=1):
        center_guess = float(spec["center_guess"])
        direction = int(spec["direction"])
        fit_left = float(spec["fit_left"])
        fit_right = float(spec["fit_right"])
        if direction not in (-1, 1):
            raise ValueError(f"edge {edge_index}: direction must be +1 or -1")
        if (
            not np.isfinite(center_guess)
            or not np.isfinite(fit_left)
            or not np.isfinite(fit_right)
            or fit_left >= fit_right
        ):
            raise ValueError(f"edge {edge_index}: invalid center or fit range")

        selected = (
            np.isfinite(all_x)
            & np.isfinite(all_y)
            & (all_y >= 0.0)
            & (all_x >= fit_left)
            & (all_x <= fit_right)
        )
        indices = np.flatnonzero(selected)
        fit_x = all_x[selected]
        fit_y = all_y[selected]
        if len(fit_x) < 5:
            raise ValueError(
                f"edge {edge_index}: shared-width fit needs at least 5 bins; "
                f"selected {len(fit_x)}"
            )
        if np.any(np.diff(fit_x) <= 0.0):
            raise ValueError(f"edge {edge_index}: x must be strictly increasing")

        bin_width = float(np.median(np.diff(fit_x)))
        side_bins = max(1, min(3, len(fit_x) // 3))
        left_level = float(np.median(fit_y[:side_bins]))
        right_level = float(np.median(fit_y[-side_bins:]))
        if direction == 1:
            baseline_guess = left_level
            amplitude_guess = right_level - left_level
        else:
            baseline_guess = right_level
            amplitude_guess = left_level - right_level
        baseline_guess = max(baseline_guess, 1.0e-9)
        amplitude_guess = max(amplitude_guess, float(np.ptp(fit_y)), 1.0)

        relative_x = fit_x - center_guess
        prepared.append(
            {
                "center_guess": center_guess,
                "direction": direction,
                "fit_x": fit_x,
                "fit_y": fit_y,
                "relative_x": relative_x,
                "indices": indices,
                "fit_left": fit_left,
                "fit_right": fit_right,
            }
        )
        bin_widths.append(bin_width)
        range_widths.append(fit_right - fit_left)
        initial_parameters.extend((baseline_guess, amplitude_guess, 0.0))
        lower_bounds.extend((0.0, 0.0, relative_x[0]))
        upper_bounds.extend((np.inf, np.inf, relative_x[-1]))

    minimum_bin_width = min(bin_widths)
    maximum_shared_width = min(range_widths)
    initial_shared_width = max(
        minimum_bin_width,
        float(np.median(range_widths)) / 6.0,
    )
    initial_shared_width = min(
        initial_shared_width,
        0.9 * maximum_shared_width,
    )
    initial_parameters[0] = initial_shared_width
    lower_bounds[0] = minimum_bin_width / 100.0
    upper_bounds[0] = maximum_shared_width

    def edge_model(relative_x, baseline, amplitude, center_offset, width, direction):
        argument = (relative_x - center_offset) / (np.sqrt(2.0) * width)
        return baseline + 0.5 * amplitude * (1.0 + direction * erf(argument))

    def poisson_deviance_residual(observed, expected):
        expected = np.maximum(expected, 1.0e-12)
        term = expected - observed
        positive = observed > 0.0
        term[positive] += observed[positive] * np.log(
            observed[positive] / expected[positive]
        )
        return np.sign(observed - expected) * np.sqrt(
            2.0 * np.maximum(term, 0.0)
        )

    def residuals(parameters):
        width = parameters[0]
        result = []
        for index, edge in enumerate(prepared):
            offset = 1 + 3 * index
            baseline, amplitude, center_offset = parameters[offset : offset + 3]
            expected = edge_model(
                edge["relative_x"],
                baseline,
                amplitude,
                center_offset,
                width,
                edge["direction"],
            )
            result.append(poisson_deviance_residual(edge["fit_y"], expected))
        return np.concatenate(result)

    result = least_squares(
        residuals,
        initial_parameters,
        bounds=(lower_bounds, upper_bounds),
        x_scale="jac",
        max_nfev=50_000,
    )
    if not result.success:
        raise RuntimeError(f"shared-width error-function fit failed: {result.message}")

    parameters = result.x
    jacobian_information = result.jac.T @ result.jac
    covariance = np.linalg.pinv(jacobian_information)
    shared_width = float(parameters[0])
    shared_width_error = float(np.sqrt(max(covariance[0, 0], 0.0)))
    deviance = float(np.sum(result.fun**2))
    ndf = int(sum(len(edge["fit_x"]) for edge in prepared) - len(parameters))
    deviance_ndf = deviance / ndf if ndf > 0 else float("nan")

    fits = []
    for index, edge in enumerate(prepared):
        offset = 1 + 3 * index
        baseline, amplitude, center_offset = parameters[offset : offset + 3]
        center = float(edge["center_guess"] + center_offset)
        center_error = float(np.sqrt(max(covariance[offset + 2, offset + 2], 0.0)))
        fitted_y = edge_model(
            edge["relative_x"],
            baseline,
            amplitude,
            center_offset,
            shared_width,
            edge["direction"],
        )
        fits.append(
            {
                "position_s": center,
                "position_error_s": center_error,
                "position_low_s": center - center_error,
                "position_high_s": center + center_error,
                "transition_width_s": shared_width,
                "transition_width_error_s": shared_width_error,
                "baseline": float(baseline),
                "amplitude": float(amplitude),
                "fit_start_s": float(edge["fit_x"][0]),
                "fit_stop_s": float(edge["fit_x"][-1]),
                "fit_bins": int(len(edge["fit_x"])),
                "indices": edge["indices"],
                "fit_x": edge["fit_x"],
                "fit_y": fitted_y,
                "covariance": covariance,
                "global_deviance": deviance,
                "global_ndf": ndf,
                "global_deviance_ndf": deviance_ndf,
            }
        )

    summary = {
        "transition_width_s": shared_width,
        "transition_width_error_s": shared_width_error,
        "deviance": deviance,
        "ndf": ndf,
        "deviance_ndf": deviance_ndf,
    }
    return fits, summary

def find_deadtime_intervals(
    bin_centers,
    counts,
    search_min_s=0.002,
    search_max_s=0.02,
    smooth_bins=9,
    edge_fraction=0.35,
    min_width_s=0.0002,
    max_width_s=0.002,
    min_log_depth=0.7,
):
    """Bracket the falling and rising sides of each deadtime valley."""
    all_x = np.asarray(bin_centers, dtype=float)
    all_y = np.asarray(counts, dtype=float)
    selected = (all_x >= search_min_s) & (all_x <= search_max_s)
    selected_indices = np.flatnonzero(selected)
    x = all_x[selected]
    y = all_y[selected]
    if len(x) < smooth_bins + 2:
        return []

    smooth_log = _smooth_log_counts(y, smooth_bins)
    slope = np.gradient(smooth_log, x)
    falling_peaks = [
        index
        for index in range(1, len(x) - 1)
        if slope[index] <= slope[index - 1] and slope[index] < slope[index + 1]
    ]
    rising_peaks = [
        index
        for index in range(1, len(x) - 1)
        if slope[index] >= slope[index - 1] and slope[index] > slope[index + 1]
    ]

    bin_width = float(np.median(np.diff(x)))
    lookaround_bins = max(3, int(round(0.0003 / bin_width)))
    candidates = []
    for left_peak in falling_peaks:
        possible_right = [
            index
            for index in rising_peaks
            if min_width_s <= x[index] - x[left_peak] <= max_width_s
        ]
        if not possible_right:
            continue
        right_peak = max(possible_right, key=lambda index: slope[index])
        valley_level = float(np.min(smooth_log[left_peak : right_peak + 1]))
        left_level = float(
            np.max(smooth_log[max(0, left_peak - lookaround_bins) : left_peak + 1])
        )
        right_level = float(
            np.max(
                smooth_log[
                    right_peak : min(len(x), right_peak + lookaround_bins + 1)
                ]
            )
        )
        if (
            left_level - valley_level < min_log_depth
            or right_level - valley_level < min_log_depth
        ):
            continue

        score = min(left_level - valley_level, right_level - valley_level)
        score *= np.sqrt(max(-slope[left_peak], 0.0) * max(slope[right_peak], 0.0))
        candidates.append((float(score), left_peak, right_peak))

    # Several noisy local extrema can describe the same valley.  Keep the
    # strongest non-overlapping falling/rising pair for each valley.
    chosen = []
    for candidate in sorted(candidates, reverse=True):
        _score, left_peak, right_peak = candidate
        overlaps = any(
            not (right_peak < other_left or left_peak > other_right)
            for _other_score, other_left, other_right in chosen
        )
        if not overlaps:
            chosen.append(candidate)

    intervals = []
    for _score, left_peak, right_peak in sorted(chosen, key=lambda item: item[1]):

        left_fit = _fit_vertical_edge(
            x, slope, left_peak, direction=-1, fraction=edge_fraction
        )
        right_fit = _fit_vertical_edge(
            x, slope, right_peak, direction=1, fraction=edge_fraction
        )
        deadtime_s = right_fit["position_s"] - left_fit["position_s"]
        if not min_width_s <= deadtime_s <= max_width_s:
            continue
        deadtime_low_s = right_fit["position_low_s"] - left_fit["position_high_s"]
        deadtime_high_s = right_fit["position_high_s"] - left_fit["position_low_s"]
        deadtime_error_s = (deadtime_high_s - deadtime_low_s) / 2.0
        # For a uniform distribution of the true edge within one bin, each
        # edge has sigma = bin_width / sqrt(12).  The deadtime is the
        # difference of two edges, hence the factor sqrt(2).  Model the
        # centered moving-average window as another uniform location range.
        deadtime_bin_error_s = bin_width / np.sqrt(6.0)
        deadtime_smoothing_error_s = (
            smooth_bins * bin_width / np.sqrt(6.0)
        )
        # Treat range, binning, and smoothing components as independent.
        deadtime_total_error_s = float(
            np.sqrt(
                deadtime_error_s**2
                + deadtime_bin_error_s**2
                + deadtime_smoothing_error_s**2
            )
        )
        intervals.append(
            {
                "tstart_s": left_fit["position_s"],
                "tstart_error_s": left_fit["position_error_s"],
                "tstart_low_s": left_fit["position_low_s"],
                "tstart_high_s": left_fit["position_high_s"],
                "tstop_s": right_fit["position_s"],
                "tstop_error_s": right_fit["position_error_s"],
                "tstop_low_s": right_fit["position_low_s"],
                "tstop_high_s": right_fit["position_high_s"],
                "deadtime_s": deadtime_s,
                "deadtime_error_s": deadtime_error_s,
                "deadtime_bin_error_s": deadtime_bin_error_s,
                "deadtime_smoothing_error_s": deadtime_smoothing_error_s,
                "deadtime_total_error_s": deadtime_total_error_s,
                "deadtime_low_s": deadtime_low_s,
                "deadtime_high_s": deadtime_high_s,
                "left_fit_start_s": left_fit["fit_start_s"],
                "left_fit_stop_s": left_fit["fit_stop_s"],
                "right_fit_start_s": right_fit["fit_start_s"],
                "right_fit_stop_s": right_fit["fit_stop_s"],
                "left_fit_bins": left_fit["fit_bins"],
                "right_fit_bins": right_fit["fit_bins"],
                "left_fit_indices": selected_indices[left_fit["indices"]],
                "right_fit_indices": selected_indices[right_fit["indices"]],
            }
        )

    return intervals


def write_deadtime_csv(
    path,
    run_number,
    intervals,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "run_number",
        "interval_index",
        "deadtime_method",
        "fit_statistic",
        "tstart_s",
        "tstart_error_s",
        "tstart_transition_width_s",
        "tstart_low_s",
        "tstart_high_s",
        "tstop_s",
        "tstop_error_s",
        "tstop_transition_width_s",
        "tstop_low_s",
        "tstop_high_s",
        "deadtime_s",
        "deadtime_error_s",
        "deadtime_transition_width_s",
        "deadtime_bin_error_s",
        "deadtime_smoothing_error_s",
        "deadtime_total_error_s",
        "deadtime_low_s",
        "deadtime_high_s",
        "tstart_ms",
        "tstart_error_ms",
        "tstart_transition_width_ms",
        "tstart_low_ms",
        "tstart_high_ms",
        "tstop_ms",
        "tstop_error_ms",
        "tstop_transition_width_ms",
        "tstop_low_ms",
        "tstop_high_ms",
        "deadtime_ms",
        "deadtime_error_ms",
        "deadtime_transition_width_ms",
        "deadtime_bin_error_ms",
        "deadtime_smoothing_error_ms",
        "deadtime_total_error_ms",
        "deadtime_low_ms",
        "deadtime_high_ms",
        "left_fit_start_s",
        "left_fit_stop_s",
        "right_fit_start_s",
        "right_fit_stop_s",
        "left_fit_bins",
        "right_fit_bins",
        "tstart_fit_statistic_ndf",
        "tstop_fit_statistic_ndf",
        "tstart_fit_converged",
        "tstop_fit_converged",
    )
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for interval_index, interval in enumerate(intervals, start=1):
            writer.writerow(
                {
                    "run_number": run_number,
                    "interval_index": interval_index,
                    **{
                        key: value
                        for key, value in interval.items()
                        if not key.endswith("_indices")
                    },
                    "tstart_ms": interval["tstart_s"] * 1000.0,
                    "tstart_error_ms": interval["tstart_error_s"] * 1000.0,
                    "tstart_transition_width_ms": interval.get(
                        "tstart_transition_width_s", float("nan")
                    ) * 1000.0,
                    "tstart_low_ms": interval["tstart_low_s"] * 1000.0,
                    "tstart_high_ms": interval["tstart_high_s"] * 1000.0,
                    "tstop_ms": interval["tstop_s"] * 1000.0,
                    "tstop_error_ms": interval["tstop_error_s"] * 1000.0,
                    "tstop_transition_width_ms": interval.get(
                        "tstop_transition_width_s", float("nan")
                    ) * 1000.0,
                    "tstop_low_ms": interval["tstop_low_s"] * 1000.0,
                    "tstop_high_ms": interval["tstop_high_s"] * 1000.0,
                    "deadtime_ms": interval["deadtime_s"] * 1000.0,
                    "deadtime_error_ms": interval["deadtime_error_s"] * 1000.0,
                    "deadtime_transition_width_ms": interval.get(
                        "deadtime_transition_width_s", float("nan")
                    ) * 1000.0,
                    "deadtime_bin_error_ms": (
                        interval["deadtime_bin_error_s"] * 1000.0
                    ),
                    "deadtime_smoothing_error_ms": (
                        interval["deadtime_smoothing_error_s"] * 1000.0
                    ),
                    "deadtime_total_error_ms": (
                        interval["deadtime_total_error_s"] * 1000.0
                    ),
                    "deadtime_low_ms": interval["deadtime_low_s"] * 1000.0,
                    "deadtime_high_ms": interval["deadtime_high_s"] * 1000.0,
                }
            )


def update_deadtime_summary_csv(path, summary):
    """Insert or replace one run in the cross-run deadtime summary CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_run = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as input_file:
            for row in csv.DictReader(input_file):
                if row.get("run_number"):
                    rows_by_run[int(row["run_number"])] = row
    rows_by_run[int(summary["run_number"])] = summary
    fields = tuple(summary.keys())
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for run_number in sorted(rows_by_run):
            row = rows_by_run[run_number]
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "-r", type=int, required=True, help="AHCAL run number")
    parser.add_argument(
        "--raw-pattern",
        action="append",
        help=(
            "Raw-file glob supporting {run} and {run06}; repeatable. "
            f"Default: {DEFAULT_RAW_PATTERN}"
        ),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Event-level CSV containing a tlu_timestamp column; skips raw input",
    )
    parser.add_argument(
        "--histogram-csv",
        type=Path,
        help="Existing histogram CSV; skips raw/event CSV reading and binning",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=1000,
        help="Number of time-difference bins (default: 1000)",
    )
    parser.add_argument(
        "--max-difference",
        type=float,
        help="Only plot differences up to this many seconds",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image (default: tlu_timestamp_difference_run_<run>.png)",
    )
    parser.add_argument(
        "--deadtime-search-min-ms",
        type=float,
        default=2.0,
        metavar="MILLISECONDS",
        help="start of the deadtime-valley search range (default: 2 ms)",
    )
    parser.add_argument(
        "--deadtime-search-max-ms",
        type=float,
        default=20.0,
        metavar="MILLISECONDS",
        help="search for deadtime intervals from 0 to this value (default: 20 ms)",
    )
    parser.add_argument(
        "--deadtime-smooth-bins",
        type=int,
        default=9,
        help="odd moving-average width used for edge selection (default: 9)",
    )
    parser.add_argument(
        "--deadtime-edge-fraction",
        type=float,
        default=0.6,
        help="fraction of peak slope included in each x=a fit (default: 0.35)",
    )
    parser.add_argument(
        "--deadtime-min-width-ms",
        type=float,
        default=0.2,
        help="minimum accepted valley width (default: 0.2 ms)",
    )
    parser.add_argument(
        "--deadtime-max-width-ms",
        type=float,
        default=2.0,
        help="maximum accepted valley width (default: 2.0 ms)",
    )
    parser.add_argument(
        "--deadtime-min-log-depth",
        type=float,
        default=0.7,
        help="minimum log(count+1) depth required on both valley sides (default: 0.7)",
    )
    parser.add_argument(
        "--deadtime-max-intervals",
        type=int,
        default=2,
        help="maximum number of deadtime intervals to keep (default: 2)",
    )
    parser.add_argument(
        "--deadtime-csv",
        type=Path,
        help="per-interval CSV (default: <output stem>_deadtime.csv)",
    )
    parser.add_argument(
        "--deadtime-summary-csv",
        type=Path,
        help="cross-run summary CSV (default: deadtime_by_run.csv beside output)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.input_csv and args.histogram_csv:
        parser.error("--input-csv and --histogram-csv cannot be used together")
    if args.raw_pattern and (args.input_csv or args.histogram_csv):
        parser.error(
            "--raw-pattern cannot be combined with --input-csv or --histogram-csv"
        )
    if args.bins <= 0:
        parser.error("--bins must be greater than zero")
    if args.max_difference is not None and args.max_difference <= 0:
        parser.error("--max-difference must be greater than zero")
    if args.deadtime_search_min_ms < 0:
        parser.error("--deadtime-search-min-ms must be zero or greater")
    if args.deadtime_search_max_ms <= args.deadtime_search_min_ms:
        parser.error("--deadtime-search-max-ms must be greater than zero")
    if args.deadtime_smooth_bins < 3 or args.deadtime_smooth_bins % 2 == 0:
        parser.error("--deadtime-smooth-bins must be an odd integer >= 3")
    if not 0 < args.deadtime_edge_fraction <= 1:
        parser.error("--deadtime-edge-fraction must be in (0, 1]")
    if args.deadtime_min_width_ms <= 0:
        parser.error("--deadtime-min-width-ms must be greater than zero")
    if args.deadtime_max_width_ms <= args.deadtime_min_width_ms:
        parser.error(
            "--deadtime-max-width-ms must exceed --deadtime-min-width-ms"
        )
    if args.deadtime_min_log_depth <= 0:
        parser.error("--deadtime-min-log-depth must be greater than zero")
    if args.deadtime_max_intervals <= 0:
        parser.error("--deadtime-max-intervals must be greater than zero")

    plot_max = (
        min(args.max_difference, PLOT_MAX_DIFFERENCE)
        if args.max_difference
        else PLOT_MAX_DIFFERENCE
    )
    output = args.output or DIAGNOSTICS_DIR / f"tlu_timestamp_difference_run_{args.run}.png"
    event_count = None
    skipped = 0
    if args.histogram_csv:
        bin_centers, counts = read_histogram_csv(args.histogram_csv)
        print(f"Loaded {len(bin_centers)} histogram bins from {args.histogram_csv}")
    else:
        if args.input_csv:
            differences, event_count, skipped = read_tlu_time_differences_csv(
                args.input_csv, expected_run=args.run
            )
            print(f"Read {event_count} events from {args.input_csv}")
        else:
            patterns = args.raw_pattern or [DEFAULT_RAW_PATTERN]
            paths = discover_raw_files(args.run, patterns)
            if not paths:
                raise RuntimeError(
                    f"No Physics raw files found for run {args.run} using: "
                    + ", ".join(patterns)
                )
            print(f"Run {args.run}: found {len(paths)} Physics raw files")
            differences, event_count, skipped = read_tlu_time_differences(
                paths, expected_run=args.run, verbose=args.verbose
            )
        if not differences:
            raise RuntimeError(
                "No positive consecutive TLU timestamp differences found"
            )
        counts, bin_edges = np.histogram(
            differences, bins=args.bins, range=(0, plot_max)
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        histogram_output = output.with_suffix(".csv")
        write_histogram_csv(histogram_output, bin_centers, counts)
        print(f"Saved reusable histogram to {histogram_output}")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    populated = counts > 0
    ax.scatter(
        bin_centers[populated],
        counts[populated],
        s=8,
        color="tab:blue",
        alpha=0.7,
        edgecolors="none",
    )
    deadtime_search_min = args.deadtime_search_min_ms / 1000.0
    deadtime_search_max = min(args.deadtime_search_max_ms / 1000.0, plot_max)
    deadtime_intervals = find_deadtime_intervals(
        bin_centers,
        counts,
        search_min_s=deadtime_search_min,
        search_max_s=deadtime_search_max,
        smooth_bins=args.deadtime_smooth_bins,
        edge_fraction=args.deadtime_edge_fraction,
        min_width_s=args.deadtime_min_width_ms / 1000.0,
        max_width_s=args.deadtime_max_width_ms / 1000.0,
        min_log_depth=args.deadtime_min_log_depth,
    )

    deadtime_intervals = deadtime_intervals[:args.deadtime_max_intervals]
    for interval_index, interval in enumerate(deadtime_intervals):
        ax.axvspan(
            interval["tstart_low_s"],
            interval["tstart_high_s"],
            color="tab:red",
            alpha=0.18,
        )
        ax.axvspan(
            interval["tstop_low_s"],
            interval["tstop_high_s"],
            color="tab:red",
            alpha=0.18,
        )
        ax.axvline(
            interval["tstart_s"],
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axvline(
            interval["tstop_s"],
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
        )
        fit_indices = np.concatenate(
            (interval["left_fit_indices"], interval["right_fit_indices"])
        )
        visible_fit_indices = fit_indices[counts[fit_indices] > 0]
        ax.scatter(
            bin_centers[visible_fit_indices],
            counts[visible_fit_indices],
            s=18,
            color="tab:red",
            zorder=3,
        )
    if deadtime_intervals:
        mean_deadtime_ms = np.mean(
            [interval["deadtime_s"] * 1000.0 for interval in deadtime_intervals]
        )
        range_error_ms = max(
            interval["deadtime_error_s"] * 1000.0
            for interval in deadtime_intervals
        )
        bin_error_ms = max(
            interval["deadtime_bin_error_s"] * 1000.0
            for interval in deadtime_intervals
        )
        smoothing_error_ms = max(
            interval["deadtime_smoothing_error_s"] * 1000.0
            for interval in deadtime_intervals
        )
        total_error_ms = np.sqrt(
            range_error_ms**2 + bin_error_ms**2 + smoothing_error_ms**2
        )
        ax.legend(
            handles=[
                Line2D(
                    [], [], color="none",
                    label=(
                        f"Deadtime = {mean_deadtime_ms:.4f} "
                        f"+\u002f- {total_error_ms:.4f} ms\n"
                        f"(range {range_error_ms:.4f} \u2295 "
                        f"bin {bin_error_ms:.4f} \u2295 "
                        f"average {smoothing_error_ms:.4f} ms)"
                    ),
                ),
                Patch(
                    facecolor="tab:red", alpha=0.18,
                    label="Edge position range",
                ),
            ],
            loc="upper right",
        )

    deadtime_csv = args.deadtime_csv or output.with_name(
        output.stem + "_deadtime.csv"
    )
    summary_csv = args.deadtime_summary_csv or output.parent / "deadtime_by_run.csv"
    write_deadtime_csv(
        deadtime_csv,
        args.run,
        deadtime_intervals,
    )
    deadtimes_ms = np.asarray(
        [interval["deadtime_s"] * 1000.0 for interval in deadtime_intervals]
    )
    deadtime_std_ms = (
        float(np.std(deadtimes_ms, ddof=1)) if len(deadtimes_ms) > 1 else float("nan")
    )
    bin_width_s = (
        float(np.median(np.diff(bin_centers))) if len(bin_centers) > 1 else float("nan")
    )
    summary = {
        "run_number": args.run,
        "intervals_found": len(deadtime_intervals),
        "median_deadtime_ms": (
            float(np.median(deadtimes_ms)) if len(deadtimes_ms) else ""
        ),
        "mean_deadtime_ms": (
            float(np.mean(deadtimes_ms)) if len(deadtimes_ms) else ""
        ),
        "deadtime_std_ms": deadtime_std_ms if len(deadtimes_ms) > 1 else "",
        "deadtime_sem_ms": (
            deadtime_std_ms / np.sqrt(len(deadtimes_ms))
            if len(deadtimes_ms) > 1
            else ""
        ),
        "deadtime_range_error_ms": (range_error_ms if len(deadtimes_ms) else ""),
        "deadtime_bin_error_ms": (bin_error_ms if len(deadtimes_ms) else ""),
        "deadtime_smoothing_error_ms": (
            smoothing_error_ms if len(deadtimes_ms) else ""
        ),
        "deadtime_total_error_ms": (total_error_ms if len(deadtimes_ms) else ""),
        "min_deadtime_ms": (
            float(np.min(deadtimes_ms)) if len(deadtimes_ms) else ""
        ),
        "max_deadtime_ms": (
            float(np.max(deadtimes_ms)) if len(deadtimes_ms) else ""
        ),
        "search_max_ms": deadtime_search_max * 1000.0,
        "search_min_ms": deadtime_search_min * 1000.0,
        "bin_width_s": bin_width_s,
        "smooth_bins": args.deadtime_smooth_bins,
        "edge_fraction": args.deadtime_edge_fraction,
        "min_width_ms": args.deadtime_min_width_ms,
        "max_width_ms": args.deadtime_max_width_ms,
        "min_log_depth": args.deadtime_min_log_depth,
    }
    update_deadtime_summary_csv(summary_csv, summary)
    print(f"Saved {len(deadtime_intervals)} deadtime intervals to {deadtime_csv}")
    print(f"Updated run summary in {summary_csv}")
    for interval_index, interval in enumerate(deadtime_intervals, start=1):
        print(
            f"  {interval_index}: tstart={interval['tstart_s'] * 1000:.6f} ms, "
            f"+/- {interval['tstart_error_s'] * 1000:.6f} ms, "
            f"tstop={interval['tstop_s'] * 1000:.6f} ms, "
            f"+/- {interval['tstop_error_s'] * 1000:.6f} ms, "
            f"deadtime={interval['deadtime_s'] * 1000:.6f} +/- "
            f"{interval['deadtime_total_error_s'] * 1000:.6f} ms "
            f"(range={interval['deadtime_error_s'] * 1000:.6f}, "
            f"bin={interval['deadtime_bin_error_s'] * 1000:.6f}, "
            f"average={interval['deadtime_smoothing_error_s'] * 1000:.6f})"
        )
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e5)
    ax.set_xlim(0, deadtime_search_max)
    ax.set_title(f"TLU timestamp differences — run {args.run}")
    ax.set_xlabel("Time difference between consecutive events [s]")
    ax.set_ylabel("Number of event intervals")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    if event_count is None:
        print(f"Saved plot to {output}")
    else:
        print(
            f"Saved {len(differences)} intervals from {event_count} "
            f"TLU events to {output}"
        )
    if skipped:
        print(f"Skipped {skipped} duplicated or out-of-order TLU timestamps")


def cpu_main():
    parser = argparse.ArgumentParser(description="Plot consecutive CPU timestamp differences read from AHCAL raw files.")
    parser.add_argument("--run", "-r", type=int, required=True, help="AHCAL run number")
    parser.add_argument(
        "--raw-pattern",
        action="append",
        help=(
            "Raw-file glob supporting {run} and {run06}; repeatable. "
            f"Default: {DEFAULT_RAW_PATTERN}"
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of time-difference bins (default: 100)",
    )
    parser.add_argument(
        "--max-difference",
        type=float,
        help="Plot differences up to this many seconds (default: 0.05)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image (default: cpu_timestamp_difference_run_<run>.png)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.bins <= 0:
        parser.error("--bins must be greater than zero")
    if args.max_difference is not None and args.max_difference <= 0:
        parser.error("--max-difference must be greater than zero")

    patterns = args.raw_pattern or [DEFAULT_RAW_PATTERN]
    paths = discover_raw_files(args.run, patterns)
    if not paths:
        raise RuntimeError(
            f"No Physics raw files found for run {args.run} using: "
            + ", ".join(patterns)
        )

    print(f"Run {args.run}: found {len(paths)} Physics raw files")
    differences, event_count, skipped = read_cpu_time_differences(
        paths, expected_run=args.run, verbose=args.verbose
    )
    if not differences:
        raise RuntimeError("No positive consecutive CPU timestamp differences found")

    plot_max = args.max_difference or PLOT_MAX_DIFFERENCE
    counts, bin_edges = np.histogram(
        differences, bins=args.bins, range=(0, plot_max)
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    populated = counts > 0

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(
        bin_centers[populated],
        counts[populated],
        s=8,
        color="tab:orange",
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e5)
    ax.set_xlim(0, plot_max)
    ax.set_title(f"CPU timestamp differences — run {args.run}")
    ax.set_xlabel("Time difference between consecutive events [s]")
    ax.set_ylabel("Number of event intervals")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output = args.output or DIAGNOSTICS_DIR / f"cpu_timestamp_difference_run_{args.run}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(
        f"Saved {len(differences)} intervals from {event_count} events to {output}"
    )
    if skipped:
        print(f"Skipped {skipped} duplicated or out-of-order CPU timestamps")

