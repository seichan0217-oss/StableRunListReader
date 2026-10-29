#!/usr/bin/env python3
"""Toy MC for repeated dead/live windows with a Poisson trigger stream."""

from .config import OUTPUT_DIR

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .pyroot_erf_fit import (
    draw_erf_fits_root,
    fit_erf_edges_independent_likelihood,
)


HERE = OUTPUT_DIR
DEFAULT_OUTPUT = HERE / "deadtime_trigger_rate" / "toy_mc"


def positive_duration(rng, mean_s, sigma_s):
    """Draw a positive Gaussian duration."""
    if sigma_s == 0.0:
        return mean_s
    value = rng.normal(mean_s, sigma_s)
    while value <= 0.0:
        value = rng.normal(mean_s, sigma_s)
    return float(value)


def simulate_recorded_intervals(
    recorded_events,
    trigger_rate_hz=100.0,
    deadtime_ms=1.3,
    deadtime_sigma_ms=0.0,
    live_ms=3.9,
    live_sigma_ms=0.0,
    seed=12345,
):
    """Simulate accepted-trigger spacings through alternating dead/live gates."""
    rng = np.random.default_rng(seed)
    dead_mean_s = deadtime_ms / 1000.0
    dead_sigma_s = deadtime_sigma_ms / 1000.0
    live_mean_s = live_ms / 1000.0
    live_sigma_s = live_sigma_ms / 1000.0
    intervals = np.empty(recorded_events, dtype=float)
    proposed = 0

    for event_index in range(recorded_events):
        elapsed = float(rng.exponential(1.0 / trigger_rate_hz))
        proposed += 1
        in_deadtime = True
        boundary = positive_duration(rng, dead_mean_s, dead_sigma_s)

        while True:
            while elapsed >= boundary:
                in_deadtime = not in_deadtime
                if in_deadtime:
                    boundary += positive_duration(
                        rng, dead_mean_s, dead_sigma_s
                    )
                else:
                    boundary += positive_duration(
                        rng, live_mean_s, live_sigma_s
                    )

            if not in_deadtime:
                intervals[event_index] = elapsed
                break

            elapsed += float(rng.exponential(1.0 / trigger_rate_hz))
            proposed += 1

    return intervals, proposed


def nominal_edge_centers(deadtime_ms, live_ms, deadtime_intervals=3):
    """Return alternating rising/falling boundary positions."""
    positions = []
    elapsed_ms = 0.0
    for interval_index in range(deadtime_intervals):
        elapsed_ms += deadtime_ms
        positions.append(elapsed_ms)
        if interval_index + 1 < deadtime_intervals:
            elapsed_ms += live_ms
            positions.append(elapsed_ms)
    return np.asarray(positions, dtype=float)


def expected_edge_sigmas(
    dead_sigma_ms, live_sigma_ms, deadtime_intervals=3
):
    """Propagate independent duration jitter to absolute boundaries."""
    sd2 = dead_sigma_ms**2
    sl2 = live_sigma_ms**2
    variances = []
    cumulative_variance = 0.0
    for interval_index in range(deadtime_intervals):
        cumulative_variance += sd2
        variances.append(cumulative_variance)
        if interval_index + 1 < deadtime_intervals:
            cumulative_variance += sl2
            variances.append(cumulative_variance)
    return np.sqrt(variances)


def fit_histogram(
    intervals,
    bins,
    max_difference_ms,
    deadtime_ms,
    deadtime_sigma_ms,
    live_ms,
    live_sigma_ms,
    fit_half_range_ms,
    statistic,
    deadtime_intervals=3,
):
    counts, edges = np.histogram(
        intervals,
        bins=bins,
        range=(0.0, max_difference_ms / 1000.0),
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    guesses_ms = nominal_edge_centers(
        deadtime_ms, live_ms, deadtime_intervals
    )
    boundary_sigmas_ms = expected_edge_sigmas(
        deadtime_sigma_ms, live_sigma_ms, deadtime_intervals
    )
    directions = tuple(1 if index % 2 == 0 else -1 for index in range(len(guesses_ms)))
    fit_specs = []

    for index, (guess_ms, sigma_ms, direction) in enumerate(
        zip(guesses_ms, boundary_sigmas_ms, directions), start=1
    ):
        half_ms = max(fit_half_range_ms, 3.0 * sigma_ms)
        fit_specs.append(
            {
                "label": f"edge_{index}",
                "center_guess": guess_ms / 1000.0,
                "direction": direction,
                "fit_left": (guess_ms - half_ms) / 1000.0,
                "fit_right": (guess_ms + half_ms) / 1000.0,
            }
        )

    fit_x, fit_y, prepared, fits, fit_summary = (
        fit_erf_edges_independent_likelihood(
            centers, counts, fit_specs, statistic=statistic
        )
    )
    positions_ms = np.asarray(
        [fit["position_s"] * 1000.0 for fit in fits]
    )
    position_errors_ms = np.asarray(
        [fit["position_error_s"] * 1000.0 for fit in fits]
    )
    widths_ms = np.asarray(
        [fit["transition_width_s"] * 1000.0 for fit in fits]
    )

    deadtimes_ms = [positions_ms[0]]
    deadtime_errors_ms = [position_errors_ms[0]]
    deadtime_sigmas_ms = [widths_ms[0]]
    for rising_index in range(2, len(positions_ms), 2):
        falling_index = rising_index - 1
        deadtimes_ms.append(
            positions_ms[rising_index] - positions_ms[falling_index]
        )
        deadtime_errors_ms.append(
            np.hypot(
                position_errors_ms[falling_index],
                position_errors_ms[rising_index],
            )
        )
        variance = (
            widths_ms[rising_index] ** 2
            - widths_ms[falling_index] ** 2
        )
        deadtime_sigmas_ms.append(
            np.sqrt(variance) if variance >= 0.0 else np.nan
        )
    deadtimes_ms = np.asarray(deadtimes_ms)
    deadtime_errors_ms = np.asarray(deadtime_errors_ms)
    deadtime_sigmas_ms = np.asarray(deadtime_sigmas_ms)
    return {
        "centers": centers,
        "counts": counts,
        "prepared_edges": prepared,
        "fits": fits,
        "fit_x": fit_x,
        "fit_y": fit_y,
        "fit_summary": fit_summary,
        "positions_ms": positions_ms,
        "position_errors_ms": position_errors_ms,
        "edge_widths_ms": widths_ms,
        "deadtimes_ms": deadtimes_ms,
        "deadtime_errors_ms": deadtime_errors_ms,
        "deadtime_sigmas_ms": deadtime_sigmas_ms,
    }


def row_from_fit(live_sigma_ms, replicate, proposed, result):
    row = {
        "live_sigma_ms": live_sigma_ms,
        "replicate": replicate,
        "proposed_triggers": proposed,
    }
    for index in range(len(result["deadtimes_ms"])):
        row[f"deadtime_{index + 1}_ms"] = result["deadtimes_ms"][index]
        row[f"deadtime_{index + 1}_fit_error_ms"] = (
            result["deadtime_errors_ms"][index]
        )
        row[f"deadtime_{index + 1}_edge_sigma_ms"] = (
            result["deadtime_sigmas_ms"][index]
        )
    for index in range(len(result["positions_ms"])):
        row[f"edge_{index + 1}_ms"] = result["positions_ms"][index]
        row[f"edge_{index + 1}_sigma_ms"] = result["edge_widths_ms"][index]
    return row


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_scan(rows, true_deadtime_ms, output):
    figure, axis = plt.subplots(figsize=(9, 6))
    colors = ("C0", "C1", "C2")
    markers = ("o", "s", "^")
    live_sigmas = sorted({float(row["live_sigma_ms"]) for row in rows})

    deadtime_count = max(
        sum(key.endswith("_edge_sigma_ms") for key in row)
        for row in rows
    )
    for deadtime_index in range(1, deadtime_count + 1):
        means = []
        spreads = []
        for live_sigma in live_sigmas:
            values = np.asarray(
                [
                    row[f"deadtime_{deadtime_index}_ms"]
                    for row in rows
                    if float(row["live_sigma_ms"]) == live_sigma
                ],
                dtype=float,
            )
            means.append(float(values.mean()))
            spreads.append(
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        axis.errorbar(
            live_sigmas,
            means,
            yerr=spreads,
            marker=markers[(deadtime_index - 1) % len(markers)],
            color=colors[(deadtime_index - 1) % len(colors)],
            capsize=3,
            label=f"Fitted Dead time {deadtime_index}",
        )

    axis.axhline(
        true_deadtime_ms,
        color="black",
        linestyle="--",
        label=f"Generated dead time = {true_deadtime_ms:.3f} ms",
    )
    axis.set(
        xlabel="Generated live-time jitter sigma_L [ms]",
        ylabel="Fitted dead time [ms]",
        title="Toy MC: effect of live-time jitter on fitted dead times",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_deadtime_sigma_by_interval(rows, true_sigma_ms, output):
    """Plot the recovered incremental dead-time widths by interval."""
    figure, axis = plt.subplots(figsize=(9, 6))
    deadtime_count = max(
        sum(key.endswith("_edge_sigma_ms") for key in row)
        for row in rows
    )
    interval_numbers = np.arange(1, deadtime_count + 1)
    live_sigmas = sorted({float(row["live_sigma_ms"]) for row in rows})

    for live_sigma_index, live_sigma in enumerate(live_sigmas):
        means = []
        spreads = []
        for deadtime_index in interval_numbers:
            values = np.asarray(
                [
                    row[f"deadtime_{deadtime_index}_edge_sigma_ms"]
                    for row in rows
                    if float(row["live_sigma_ms"]) == live_sigma
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            means.append(float(values.mean()) if len(values) else np.nan)
            spreads.append(
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        offset = 0.08 * (
            live_sigma_index - 0.5 * (len(live_sigmas) - 1)
        )
        axis.errorbar(
            interval_numbers + offset,
            means,
            yerr=spreads,
            marker="o",
            capsize=3,
            label=f"Generated sigma_L = {live_sigma:g} ms",
        )

    axis.axhline(
        true_sigma_ms,
        color="black",
        linestyle="--",
        label=f"Generated sigma_D = {true_sigma_ms:g} ms",
    )
    axis.set(
        xlabel="Dead-time interval number",
        ylabel="Recovered incremental sigma_D [ms]",
        title="Toy MC: stability of variance-difference width estimator",
        xticks=interval_numbers,
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_example(
    result,
    trigger_rate_hz,
    true_deadtime_ms,
    live_ms,
    live_sigma_ms,
    output,
):
    figure, axis = plt.subplots(figsize=(11, 6.5))
    populated = result["counts"] > 0
    axis.scatter(
        result["centers"][populated] * 1000.0,
        result["counts"][populated],
        s=5,
        alpha=0.6,
        edgecolors="none",
    )
    for position_ms, position_error_ms, width_ms in zip(
        result["positions_ms"],
        result["position_errors_ms"],
        result["edge_widths_ms"],
    ):
        edge_half_range_ms = position_error_ms + width_ms
        axis.axvspan(
            position_ms - edge_half_range_ms,
            position_ms + edge_half_range_ms,
            color="tab:red",
            alpha=0.18,
        )
        axis.axvline(
            position_ms,
            color="tab:red",
            linestyle="--",
            linewidth=1,
        )
    nominal = nominal_edge_centers(
        true_deadtime_ms, live_ms, len(result["deadtimes_ms"])
    )
    for position_ms in nominal:
        axis.axvline(
            position_ms,
            color="black",
            linestyle=":",
            linewidth=1,
        )

    handles = []
    for index, (deadtime_ms, error_ms, sigma_ms) in enumerate(
        zip(
            result["deadtimes_ms"],
            result["deadtime_errors_ms"],
            result["deadtime_sigmas_ms"],
        ),
        start=1,
    ):
        if index == 1:
            label = (
                f"Initial deadtime: ({deadtime_ms:.4f} +/- "
                f"{error_ms:.4f} +/- {sigma_ms:.4f}) ms"
            )
        else:
            label = (
                f"Deadtime interval {index}: ({deadtime_ms:.4f} +/- "
                f"{error_ms:.4f} +/- {sigma_ms:.4f}) ms"
            )
        handles.append(
            Line2D(
                [], [], color="tab:red", linestyle="--", linewidth=1,
                label=label,
            )
        )
    handles.extend(
        [
            Patch(
                facecolor="tab:red",
                alpha=0.18,
                label="Edge range: mu +/- delta_mu +/- sigma",
            ),
            Line2D(
                [], [], color="black", linestyle=":", linewidth=1,
                label="Generated mean boundaries",
            ),
        ]
    )
    axis.set_yscale("log")
    axis.set(
        xlabel="Time difference between consecutive recorded events [ms]",
        ylabel="Number of intervals",
        title=(
            f"{trigger_rate_hz:g} Hz Poisson toy, "
            f"sigma_L={live_sigma_ms:g} ms"
        ),
    )
    axis.grid(True, alpha=0.3)
    axis.legend(handles=handles, loc="upper right", fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger-rate-hz", type=float, default=100.0)
    parser.add_argument("--recorded-events", type=int, default=100_000)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--deadtime-intervals", type=int, default=3)
    parser.add_argument("--deadtime-ms", type=float, default=1.3)
    parser.add_argument("--deadtime-sigma-ms", type=float, default=0.03)
    parser.add_argument("--live-ms", type=float, default=3.9)
    parser.add_argument(
        "--live-sigma-ms",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.10, 0.20),
    )
    parser.add_argument("--bins", type=int, default=4000)
    parser.add_argument("--max-difference-ms", type=float, default=20.0)
    parser.add_argument("--fit-half-range-ms", type=float, default=0.30)
    parser.add_argument(
        "--fit-statistic", choices=("poisson", "chi2"), default="chi2"
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trigger_rate_hz <= 0.0:
        raise ValueError("--trigger-rate-hz must be positive")
    if (
        args.recorded_events <= 0
        or args.replicates <= 0
        or args.deadtime_intervals <= 0
        or args.bins <= 0
    ):
        raise ValueError("event, replicate, and bin counts must be positive")
    if any(value < 0.0 for value in args.live_sigma_ms):
        raise ValueError("--live-sigma-ms values must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    example = None
    for sigma_index, live_sigma_ms in enumerate(args.live_sigma_ms):
        for replicate in range(args.replicates):
            seed = args.seed + 1000 * sigma_index + replicate
            intervals, proposed = simulate_recorded_intervals(
                recorded_events=args.recorded_events,
                trigger_rate_hz=args.trigger_rate_hz,
                deadtime_ms=args.deadtime_ms,
                deadtime_sigma_ms=args.deadtime_sigma_ms,
                live_ms=args.live_ms,
                live_sigma_ms=live_sigma_ms,
                seed=seed,
            )
            result = fit_histogram(
                intervals=intervals,
                bins=args.bins,
                max_difference_ms=args.max_difference_ms,
                deadtime_ms=args.deadtime_ms,
                deadtime_sigma_ms=args.deadtime_sigma_ms,
                live_ms=args.live_ms,
                live_sigma_ms=live_sigma_ms,
                fit_half_range_ms=args.fit_half_range_ms,
                statistic=args.fit_statistic,
                deadtime_intervals=args.deadtime_intervals,
            )
            rows.append(row_from_fit(live_sigma_ms, replicate, proposed, result))
            if sigma_index == len(args.live_sigma_ms) - 1 and replicate == 0:
                example = result
            print(
                f"sigma_L={live_sigma_ms:.3f} ms replicate={replicate}: "
                + ", ".join(
                    f"D{i + 1}={value:.6f} ms"
                    for i, value in enumerate(result["deadtimes_ms"])
                )
            )

    csv_path = args.output_dir / "toy_live_jitter_scan.csv"
    scan_plot = args.output_dir / "toy_live_jitter_scan.png"
    sigma_plot = args.output_dir / "toy_deadtime_sigma_by_interval.png"
    example_plot = args.output_dir / "toy_live_jitter_example.png"
    example_fit_plot = (
        args.output_dir / "toy_live_jitter_example_erf_fit.png"
    )
    write_csv(csv_path, rows)
    plot_scan(rows, args.deadtime_ms, scan_plot)
    plot_deadtime_sigma_by_interval(
        rows, args.deadtime_sigma_ms, sigma_plot
    )
    plot_example(
        example,
        args.trigger_rate_hz,
        args.deadtime_ms,
        args.live_ms,
        args.live_sigma_ms[-1],
        example_plot,
    )
    draw_erf_fits_root(
        example["fit_x"],
        example["fit_y"],
        example["prepared_edges"],
        example["fits"],
        example["fit_summary"],
        example_fit_plot,
        title_prefix=(
            f"Toy {args.trigger_rate_hz:g} Hz, "
            f"sigma_L={args.live_sigma_ms[-1]:g} ms: edge"
        ),
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {scan_plot}")
    print(f"Wrote {sigma_plot}")
    print(f"Wrote {example_plot}")
    print(f"Wrote {example_fit_plot}")


if __name__ == "__main__":
    main()
