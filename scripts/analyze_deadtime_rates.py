#!/usr/bin/env python3
"""Fit deadtime and calculate trigger rates for the intervals in a CSV file."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import math
import shutil
from types import SimpleNamespace

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_runlist import config
from stable_runlist.add_stable_luminosity import write_json
from stable_runlist.interval_csv import prepare_output, read_intervals, selection_directory
from stable_runlist.prepost import selected_ratios

def analyze_run(run, options):
    """Read the selected events, fit deadtime, count triggers, and save plots."""
    number = run["run_number"]
    job = options.output_dir / "batch" / f"run_{number}_bins_{options.bins}"
    job.mkdir(parents=True, exist_ok=True)
    log = job / "analysis.log"
    try:
        with log.open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
            import matplotlib.pyplot as plt
            import numpy as np
            from stable_runlist import deadtime_and_trigger_rate as deadtime
            from stable_runlist.raw_trigger_rate import calculate_recorded_rate_rows, write_rate_csv

            # 1. Find raw files and load the luminosity time series.
            paths = deadtime.discover_raw_files(number, options.raw_pattern or [deadtime.DEFAULT_RAW_PATTERN])
            if not paths:
                raise RuntimeError(f"no raw files found for run {number}")
            time_intervals = [(i["start_utime"], i["stop_utime"]) for i in run["stable_list"]]
            lumi_csv = Path(options.luminosity_csv.format(
                run=number, run06=f"{number:06d}",
            )) if options.luminosity_csv else None
            lumi_options = SimpleNamespace(
                run=number, output_dir=job, luminosity_csv=lumi_csv,
                luminosity_script=options.luminosity_script or deadtime.DEFAULT_LUMINOSITY_SCRIPT,
                lumi_bin_seconds=0.2, lumi_smooth_bins=11, lumi_plot_stride=1,
                lumi_measurement=None, lumi_field="Luminosity",
            )
            luminosity, lumi_source = deadtime.obtain_luminosity(lumi_options)
            local_lumi = job / f"run_{number}_timeseries.csv"
            if lumi_source.resolve() != local_lumi.resolve():
                shutil.copy2(lumi_source, local_lumi)

            # 2. Build the TLU time-difference histogram within the CSV intervals.
            differences, event_count, skipped = deadtime.read_stable_tlu_time_differences(
                paths, time_intervals, expected_run=number, merge_files=True,
            )
            if not differences:
                raise RuntimeError("no positive TLU differences in the selected intervals")
            counts, edges = np.histogram(
                differences, bins=options.bins, range=(0, deadtime.PLOT_MAX_DIFFERENCE),
            )
            centers = (edges[:-1] + edges[1:]) / 2.0
            histogram_csv = job / f"run_{number}_stable_deadtime_histogram.csv"
            deadtime.write_histogram_csv(histogram_csv, centers, counts)
            print(f"Selected {event_count} events; skipped {skipped} non-positive differences")

            # 3. Locate the edges, fit them with PyROOT, and save D1-D3.
            manual_ranges = deadtime.parse_manual_erf_ranges(options.erf_fit_range)
            initial = deadtime.find_initial_deadtime_interval(centers, counts)
            candidates = deadtime.find_deadtime_intervals(centers, counts, edge_fraction=0.45)
            if initial is not None:
                candidates.insert(0, initial)
            candidates = deadtime.limit_deadtime_intervals(candidates, 3, manual_ranges)
            fits = deadtime.plot_erf_edge_fits(
                centers, counts, candidates, job / f"run_{number}_deadtime_erf_fit.png",
                dpi=options.dpi, manual_ranges=manual_ranges,
                statistic=options.erf_fit_statistic, run_number=number,
            )
            fitted = deadtime.limit_deadtime_intervals(
                deadtime.deadtime_intervals_from_erf_fits(fits, manual_ranges), 3, manual_ranges,
            )
            deadtime.write_deadtime_csv(job / f"run_{number}_deadtime.csv", number, fitted)
            print("Deadtime [ms]: " + ", ".join(f"{i['deadtime_s'] * 1000:.6f}" for i in fitted))

            # 4. Deduplicate raw events and count only the selected time intervals.
            rate_options = SimpleNamespace(
                run=number, output_dir=job, bin_seconds=options.bin_seconds,
                trigger_order=options.trigger_order, recent_id_limit=100_000,
                reorder_tolerance=1024,
            )
            rate_bins, rate_stats = calculate_recorded_rate_rows(paths, rate_options, time_intervals)
            if not rate_bins:
                raise RuntimeError("no raw trigger-rate coverage in the selected intervals")
            write_rate_csv(job / f"run_{number}_trigger_rate.csv", rate_bins)
            print(f"Processed {rate_stats.unique_triggers} unique and {rate_stats.duplicate_triggers} duplicate triggers")

            # 5. Save the deadtime/rate/luminosity plot and the veto-fraction plot.
            figure, axes = plt.subplots(2, 1, figsize=(12, 12))
            deadtime.plot_deadtime(axes[0], centers, counts, fitted)
            axes[0].set_xlim(0, 0.020)
            deadtime.plot_trigger_rate(axes[1], rate_bins, luminosity, 11)
            figure.suptitle(f"AHCAL run {number}: deadtime, trigger rate, and luminosity", fontsize=15)
            figure.tight_layout(rect=(0, 0, 1, 0.97))
            figure.savefig(job / f"run_{number}_deadtime_trigger_rate_luminosity.png", dpi=options.dpi)
            plt.close(figure)

            figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                        gridspec_kw={"height_ratios": (2, 1)})
            deadtime.plot_trigger_rate_and_any_vetoed_fraction(axes[0], axes[1], rate_bins)
            figure.suptitle(f"AHCAL run {number}: trigger rates and AnyVetoed ratio", fontsize=15)
            figure.tight_layout(rect=(0, 0, 1, 0.96))
            figure.savefig(job / f"run_{number}_trigger_rate_any_vetoed_ratio.png", dpi=options.dpi)
            plt.close(figure)
        return {"run_number": number, "status": "ok", "log": str(log),
                "trigger_rate_csv": str(job / f"run_{number}_trigger_rate.csv")}
    except Exception as error:
        import traceback
        with log.open("a") as stream:
            traceback.print_exc(file=stream)
        return {"run_number": number, "status": "error", "error": str(error), "log": str(log)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="interval CSV; its rows are the entire selection")
    parser.add_argument("--run", "-r", type=int, action="append", help="restrict to this run in the CSV; repeatable")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--raw-pattern", action="append")
    parser.add_argument("--bins", type=int, default=5000, help="deadtime histogram bins")
    parser.add_argument("--bin-seconds", type=float, default=2.0, help="recorded trigger-rate bin width")
    parser.add_argument("--workers", type=int, default=1, help="independent per-run worker processes")
    parser.add_argument("--trigger-order", choices=("merged", "external-sort"), default="merged")
    parser.add_argument("--erf-fit-statistic", choices=("poisson", "chi2"), default="poisson")
    parser.add_argument("--erf-fit-range", action="append", nargs=3, metavar=("LEFT_MS", "RIGHT_MS", "DIRECTION"))
    parser.add_argument("--luminosity-csv", help="instantaneous luminosity series; may contain {run} or {run06}")
    parser.add_argument("--luminosity-script", type=Path, help="alternative FAIR luminosity downloader")
    parser.add_argument("--prepost-ratio-csv", type=Path, action="append")
    parser.add_argument("--secret-file", default=config.DEFAULT_SECRET_FILE)
    parser.add_argument("--live-window-ms", type=float, default=3.9)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if any(n <= 0 for n in (args.bins, args.workers, args.dpi)) or any(
        not math.isfinite(n) or n <= 0 for n in (args.bin_seconds, args.live_window_ms)
    ):
        parser.error("bins, workers, dpi, bin-seconds and live-window-ms must be positive and finite")
    runs = read_intervals(args.input, args.run)
    if not runs:
        raise ValueError("input CSV contains no intervals")
    args.output_dir = (args.output_dir or selection_directory(config.SELECTION_ANALYSIS_DIR, runs)).resolve()
    prepare_output(args.output_dir, runs, args.input)
    write_json(args.output_dir / "analysis_status.json", {"status": "running"})
    write_json(args.output_dir / "analysis_report.json", [])
    ratios = selected_ratios(runs, args.prepost_ratio_csv,
                             args.output_dir / "prepost_veto_ratios.csv", args.secret_file,
                             allow_missing=True)
    reports = []

    def record(result):
        reports.append(result)
        write_json(args.output_dir / "analysis_report.json", sorted(reports, key=lambda x: x["run_number"]))
        print(f"[{len(reports)}/{len(runs)}] run {result['run_number']}: {result['status']}; {result['log']}", flush=True)
        if result.get("error"):
            print(result["error"], flush=True)

    if args.workers == 1:
        for run in runs:
            record(analyze_run(run, args))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(analyze_run, run, args) for run in runs]
            for future in as_completed(futures):
                record(future.result())
    complete = {row["run_number"] for row in reports if row["status"] == "ok"}
    if not complete:
        write_json(args.output_dir / "analysis_status.json", {"status": "failed"})
        return 1

    from stable_runlist import rate_summary
    rows = []
    for run in runs:
        if run["run_number"] not in complete:
            continue
        row = rate_summary.collect(run, args.output_dir / "batch", args.bins)
        row["pre_over_post_ratio"] = ratios.get(run["run_number"], math.nan)
        if run["run_number"] not in ratios:
            row["status"] = "missing_prepost_ratio"
        row["used_for_deadtime_model"] = True
        rows.append(row)
    model = rate_summary.add_true_rate_corrections(rows, args.live_window_ms)
    summary_dir = args.output_dir / "summary"
    summary, _, _ = rate_summary.write_summary_files(rows, model, summary_dir)
    rate_summary.make_plots(rows, summary_dir, args.dpi)
    from stable_runlist import fit_luminosity_true_rate_pyroot as fitting
    fitting.fit_configuration_groups(summary, summary_dir, skip_insufficient=True)
    incomplete = len(complete) != len(runs) or any(row["status"] != "ok" for row in rows)
    write_json(args.output_dir / "analysis_status.json", {"status": "partial" if incomplete else "complete"})
    print(f"Wrote analysis and shared D1-D3 model to {args.output_dir}")
    return int(incomplete)


if __name__ == "__main__":
    raise SystemExit(main())
