"""Legacy single-deadtime/prescale rate CSV reconstruction and refresh."""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from .config import OUTPUT_DIR
from .any_veto_prescale import fetch_any_veto_prescale, prescale_pass_fraction
from .prepost import DEFAULT_PREPOST_CSV, read_pre_over_post_ratio
from .raw_trigger_rate import (
    VetoRateBin, calculate_recorded_rate_rows, write_rate_csv,
)
from .daq_conditions import solve_true_rate_from_reco

HERE = OUTPUT_DIR


def _add_correction_arguments(parser):
    parser.add_argument("--prepost-ratio-csv", type=Path, default=DEFAULT_PREPOST_CSV)
    parser.add_argument("--assumed-deadtime-ms", type=float, default=1.22)
    parser.add_argument("--assumed-auto-triggered-ms", type=float, default=3.9)


def _parse_rebuild_args():
    parser = argparse.ArgumentParser(description="Rebuild a run's legacy AnyVetoed/NonVetoed rate CSV from raw files.")
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--raw-pattern", action="append")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output CSV; default is the existing batch run directory for "
            "--analysis-bins"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write run_<run>_trigger_rate.csv in this directory",
    )
    parser.add_argument("--analysis-bins", type=int, default=5000)
    parser.add_argument("--bin-seconds", type=float, default=2.0)
    parser.add_argument("--correction-window-seconds", type=float, default=10.0)
    parser.add_argument(
        "--correction-smooth-method",
        choices=("gaussian", "moving-average"),
        default="gaussian",
    )
    parser.add_argument("--recent-id-limit", type=int, default=100_000)
    parser.add_argument("--reorder-tolerance", type=int, default=1024)
    parser.add_argument(
        "--trigger-order",
        choices=("merged", "external-sort"),
        default="merged",
    )
    _add_correction_arguments(parser)
    return parser.parse_args()


def rebuild_main():
    args = _parse_rebuild_args()
    from .timestamp_diagnostics import DEFAULT_RAW_PATTERN, discover_raw_files
    if args.output is not None and args.output_dir is not None:
        raise ValueError("--output and --output-dir cannot be combined")
    patterns = args.raw_pattern or [DEFAULT_RAW_PATTERN]
    paths = discover_raw_files(args.run, patterns)
    if not paths:
        raise RuntimeError(f"no raw files found for run {args.run}")
    prescale = fetch_any_veto_prescale(args.run)
    prepost_ratio = read_pre_over_post_ratio(args.run, args.prepost_ratio_csv)
    if args.output is not None:
        output = args.output
    elif args.output_dir is not None:
        output = args.output_dir / f"run_{args.run}_trigger_rate.csv"
    else:
        output = (
            HERE
            / "deadtime_trigger_rate"
            / "batch"
            / f"run_{args.run}_bins_{args.analysis_bins}"
            / f"run_{args.run}_trigger_rate.csv"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    calculation_args = SimpleNamespace(
        recent_id_limit=args.recent_id_limit,
        reorder_tolerance=args.reorder_tolerance,
        bin_seconds=args.bin_seconds,
        trigger_order=args.trigger_order,
        output_dir=output.parent,
        correction_window_seconds=args.correction_window_seconds,
        assumed_deadtime_ms=args.assumed_deadtime_ms,
        assumed_auto_triggered_ms=args.assumed_auto_triggered_ms,
        correction_smooth_method=args.correction_smooth_method,
        any_vetoed_prescale_factor=prescale.correction_factor,
        pre_over_post_ratio=prepost_ratio,
    )
    recorded, stats = calculate_recorded_rate_rows(paths, calculation_args)
    bins = estimate_true_rate_bins(
        recorded,
        correction_window_seconds=args.correction_window_seconds,
        deadtime_ms=args.assumed_deadtime_ms,
        auto_triggered_ms=args.assumed_auto_triggered_ms,
        smooth_method=args.correction_smooth_method,
        any_vetoed_prescale_factor=prescale.correction_factor,
        pre_over_post_ratio=prepost_ratio,
    )
    write_rate_csv(output, bins)
    metadata = {
        "run_number": args.run,
        "raw_files": [str(path) for path in paths],
        "trigger_source": "raw TLU events",
        "configuration": prescale.configuration,
        "faser_calib_rate": prescale.calib_rate,
        "veto_trigger_stretch": prescale.veto_stretch,
        "any_vetoed_pass_fraction": prescale.pass_fraction,
        "pre_over_post_ratio": prepost_ratio,
        "any_vetoed_prescale_factor": prescale.correction_factor,
        "bin_seconds": args.bin_seconds,
        "correction_window_seconds": args.correction_window_seconds,
        "correction_smooth_method": args.correction_smooth_method,
        "assumed_deadtime_ms": args.assumed_deadtime_ms,
        "assumed_auto_triggered_ms": args.assumed_auto_triggered_ms,
        "unique_triggers": stats.unique_triggers,
        "duplicate_triggers": stats.duplicate_triggers,
    }
    metadata_path = output.with_name(output.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"run {args.run}: wrote {output} with {stats.unique_triggers:,} "
        f"unique triggers; AnyVetoed correction x{prescale.correction_factor:g}",
        flush=True,
    )


def solve_true_rates(recorded, deadtime_s, live_window_s):
    """Vector form of daq_conditions.solve_true_rate_from_reco."""
    target = np.asarray(recorded, dtype=float)
    result = np.zeros_like(target)
    valid = np.isfinite(target) & (target > 0)
    if not np.any(valid):
        return result
    goal = target[valid]
    lo = np.zeros_like(goal)
    hi = np.maximum(1.0, 2 * goal + 1)

    def reco(rate):
        accepted = -np.expm1(-live_window_s * rate)
        return np.divide(
            rate * accepted, accepted + deadtime_s * rate,
            out=np.zeros_like(rate), where=rate > 0,
        )

    for _ in range(100):
        low_hi = reco(hi) < goal
        if not np.any(low_hi):
            break
        hi[low_hi] *= 2
    if np.any(reco(hi) < goal):
        raise ValueError("true-rate inversion has no upper bracket")
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        low_mid = reco(mid) < goal
        lo[low_mid] = mid[low_mid]
        hi[~low_mid] = mid[~low_mid]
    result[valid] = 0.5 * (lo + hi)
    result[~np.isfinite(target)] = math.nan
    return result


def update_chunk(rows, ratio, factor, deadtime_s, live_window_s):
    smoothed = np.asarray([float(row["smoothed_rate_hz"]) for row in rows])
    pre_veto = smoothed * ratio
    true = solve_true_rates(pre_veto, deadtime_s, live_window_s)
    fractions = np.asarray([float(row["any_vetoed_fraction"]) for row in rows])
    any_rate = true * fractions
    for index, row in enumerate(rows):
        row["pre_over_post_ratio"] = ratio
        row["pre_veto_reco_rate_hz"] = pre_veto[index]
        row["true_rate_hz"] = true[index]
        row["live_fraction"] = (
            smoothed[index] / true[index] if true[index] > 0 else ""
        )
        row["true_over_reco"] = (
            true[index] / smoothed[index] if smoothed[index] > 0 else ""
        )
        row["any_vetoed_prescale_factor"] = factor
        row["deadtime_corrected_any_vetoed_rate_hz"] = any_rate[index]
        row["prescale_corrected_any_vetoed_rate_hz"] = any_rate[index] * factor
        row["corrected_any_vetoed_rate_hz"] = any_rate[index] * factor
    return rows


def refresh_main():
    parser = argparse.ArgumentParser(description="Recalculate legacy raw-derived rate bins with TLU prescale corrections.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    _add_correction_arguments(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("run_*_trigger_rate.csv"))
    if not files:
        raise ValueError("no raw-derived trigger-rate CSV files found")
    for source in files:
        match = re.fullmatch(r"run_(\d+)_trigger_rate\.csv", source.name)
        if match is None:
            continue
        run = int(match.group(1))
        metadata_path = source.with_name(source.stem + "_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        calib_rate = float(metadata["faser_calib_rate"])
        fraction = prescale_pass_fraction(
            metadata["veto_trigger_stretch"], calib_rate
        )
        factor = 1 / fraction
        ratio = read_pre_over_post_ratio(run, args.prepost_ratio_csv)
        destination = args.output_dir / source.name
        count = 0
        with source.open(encoding="utf-8", newline="") as input_stream, \
                destination.open("w", encoding="utf-8", newline="") as output_stream:
            reader = csv.DictReader(input_stream)
            fieldnames = list(reader.fieldnames)
            for field in ("pre_veto_reco_rate_hz", "pre_over_post_ratio"):
                if field not in fieldnames:
                    fieldnames.append(field)
            writer = csv.DictWriter(output_stream, fieldnames=fieldnames)
            writer.writeheader()
            chunk = []
            for row in reader:
                chunk.append(row)
                if len(chunk) >= 5000:
                    writer.writerows(update_chunk(
                        chunk, ratio, factor,
                        args.assumed_deadtime_ms / 1000,
                        args.assumed_auto_triggered_ms / 1000,
                    ))
                    count += len(chunk)
                    chunk = []
            if chunk:
                writer.writerows(update_chunk(
                    chunk, ratio, factor,
                    args.assumed_deadtime_ms / 1000,
                    args.assumed_auto_triggered_ms / 1000,
                ))
                count += len(chunk)
        metadata["pre_over_post_ratio"] = ratio
        metadata["any_vetoed_pass_fraction"] = fraction
        metadata["any_vetoed_prescale_factor"] = factor
        metadata["correction_source"] = (
            "(2 * veto trigger stretch + 3) / (FASER_CalibRate + 1) "
            "and per-run TLU PreVeto/PostVeto counters"
        )
        (args.output_dir / metadata_path.name).write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"run {run}: {count} bins, Pre/Post {ratio:.6f}, prescale x{factor:.6g}", flush=True)
    print(f"Wrote {len(files)} runs to {args.output_dir}")



@dataclass(frozen=True)
class TrueRateBin:
    start_utime: float
    stop_utime: float
    count: int
    rate_hz: float
    smoothed_rate_hz: float
    true_rate_hz: float
    pre_veto_reco_rate_hz: float
    pre_over_post_ratio: float
    live_fraction: Optional[float]
    true_over_reco: Optional[float]
    any_vetoed_count: Optional[int] = None
    non_vetoed_count: Optional[int] = None
    any_vetoed_rate_hz: float = math.nan
    non_vetoed_rate_hz: float = math.nan
    smoothed_any_vetoed_rate_hz: float = math.nan
    smoothed_non_vetoed_rate_hz: float = math.nan
    any_vetoed_fraction: float = math.nan
    any_vetoed_prescale_factor: float = 1.0
    deadtime_corrected_any_vetoed_rate_hz: float = math.nan
    prescale_corrected_any_vetoed_rate_hz: float = math.nan
    corrected_any_vetoed_rate_hz: float = math.nan


def smooth_rate_values(values, window_bins, method="gaussian"):
    if method not in {"gaussian", "moving-average"}:
        raise ValueError(
            f"unsupported smoothing method {method!r}; "
            "choose gaussian or moving-average"
        )
    if window_bins <= 1:
        return list(values)
    radius = max(1, window_bins // 2)
    sigma = max(1.0, window_bins / 3.0)
    output = []
    for index in range(len(values)):
        weighted_sum = 0.0
        weight_sum = 0.0
        for offset in range(-radius, radius + 1):
            source_index = index + offset
            if source_index < 0 or source_index >= len(values):
                continue
            value = values[source_index]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            weight = (
                math.exp(-0.5 * (offset / sigma) ** 2)
                if method == "gaussian"
                else 1.0
            )
            weighted_sum += weight * value
            weight_sum += weight
        output.append(
            weighted_sum / weight_sum if weight_sum > 0.0 else math.nan
        )
    return output


def estimate_true_rate_bins(
    bins,
    correction_window_seconds,
    deadtime_ms,
    auto_triggered_ms,
    smooth_method="gaussian",
    any_vetoed_prescale_factor=1.0,
    pre_over_post_ratio=1.0,
):
    if correction_window_seconds <= 0:
        raise ValueError("correction window must be positive")
    if deadtime_ms < 0:
        raise ValueError("assumed deadtime must be non-negative")
    if auto_triggered_ms <= 0:
        raise ValueError("assumed auto-triggered window must be positive")
    if (
        not math.isfinite(any_vetoed_prescale_factor)
        or any_vetoed_prescale_factor < 1.0
    ):
        raise ValueError(
            "AnyVetoed prescale correction factor must be finite and >= 1"
        )
    if not math.isfinite(pre_over_post_ratio) or pre_over_post_ratio < 1.0:
        raise ValueError("PreVeto/PostVeto ratio must be finite and >= 1")
    if not bins:
        return []

    bin_seconds = bins[0].stop_utime - bins[0].start_utime
    if bin_seconds <= 0:
        raise ValueError("rate bins must have positive width")
    window_bins = max(1, int(round(correction_window_seconds / bin_seconds)))
    if window_bins > 1 and window_bins % 2 == 0:
        window_bins += 1
    smoothed = smooth_rate_values(
        [item.rate_hz for item in bins],
        window_bins,
        method=smooth_method,
    )
    has_veto_rates = all(isinstance(item, VetoRateBin) for item in bins)
    if has_veto_rates:
        smoothed_any_vetoed = smooth_rate_values(
            [item.any_vetoed_rate_hz for item in bins],
            window_bins,
            method=smooth_method,
        )
        smoothed_non_vetoed = smooth_rate_values(
            [item.non_vetoed_rate_hz for item in bins],
            window_bins,
            method=smooth_method,
        )
    else:
        smoothed_any_vetoed = [math.nan] * len(bins)
        smoothed_non_vetoed = [math.nan] * len(bins)

    output = []
    for item, reco_rate, any_vetoed, non_vetoed in zip(
        bins,
        smoothed,
        smoothed_any_vetoed,
        smoothed_non_vetoed,
    ):
        pre_veto_reco_rate = reco_rate * pre_over_post_ratio
        true_rate = solve_true_rate_from_reco(
            pre_veto_reco_rate,
            deadtime_ms / 1000.0,
            auto_triggered_ms / 1000.0,
        )
        if (
            not math.isfinite(reco_rate)
            or reco_rate <= 0.0
            or not math.isfinite(true_rate)
            or true_rate <= 0.0
        ):
            live_fraction = None
            true_over_reco = None
        else:
            live_fraction = reco_rate / true_rate
            true_over_reco = true_rate / reco_rate
        reco_veto_total = any_vetoed + non_vetoed
        if (
            math.isfinite(any_vetoed)
            and math.isfinite(non_vetoed)
            and reco_veto_total > 0.0
            and math.isfinite(true_rate)
        ):
            any_vetoed_fraction = any_vetoed / reco_veto_total
            deadtime_corrected_any_vetoed_rate = (
                true_rate * any_vetoed_fraction
            )
            prescale_corrected_any_vetoed_rate = (
                deadtime_corrected_any_vetoed_rate
                * any_vetoed_prescale_factor
            )
        else:
            any_vetoed_fraction = math.nan
            deadtime_corrected_any_vetoed_rate = math.nan
            prescale_corrected_any_vetoed_rate = math.nan
        output.append(
            TrueRateBin(
                start_utime=item.start_utime,
                stop_utime=item.stop_utime,
                count=item.count,
                rate_hz=item.rate_hz,
                smoothed_rate_hz=reco_rate,
                true_rate_hz=true_rate,
                pre_veto_reco_rate_hz=pre_veto_reco_rate,
                pre_over_post_ratio=pre_over_post_ratio,
                live_fraction=live_fraction,
                true_over_reco=true_over_reco,
                any_vetoed_count=(
                    item.any_vetoed_count if has_veto_rates else None
                ),
                non_vetoed_count=(
                    item.non_vetoed_count if has_veto_rates else None
                ),
                any_vetoed_rate_hz=(
                    item.any_vetoed_rate_hz if has_veto_rates else math.nan
                ),
                non_vetoed_rate_hz=(
                    item.non_vetoed_rate_hz if has_veto_rates else math.nan
                ),
                smoothed_any_vetoed_rate_hz=any_vetoed,
                smoothed_non_vetoed_rate_hz=non_vetoed,
                any_vetoed_fraction=any_vetoed_fraction,
                any_vetoed_prescale_factor=any_vetoed_prescale_factor,
                deadtime_corrected_any_vetoed_rate_hz=(
                    deadtime_corrected_any_vetoed_rate
                ),
                prescale_corrected_any_vetoed_rate_hz=(
                    prescale_corrected_any_vetoed_rate
                ),
                # Backward-compatible name for the final AnyVetoed rate.
                corrected_any_vetoed_rate_hz=(
                    prescale_corrected_any_vetoed_rate
                ),
            )
        )
    return output

