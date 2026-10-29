#!/usr/bin/env python3
"""Integrate delivered and effective luminosity over the intervals in a CSV file."""

import argparse
import json

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_runlist import config
from stable_runlist.add_stable_luminosity import (
    CachedLuminosityReader, compact_luminosity_output, find_rate_source,
    load_selection_cache, read_count_bins, sum_luminosity_results, write_json,
)
from stable_runlist.interval_csv import prepare_output, read_intervals, selection_directory, write_table
from stable_runlist.prepost import selected_ratios

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="interval CSV; its rows are the integration intervals")
    parser.add_argument("--run", "-r", type=int, action="append", help="restrict to this run in the CSV; repeatable")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path, help="stage 3 output; otherwise inferred from the CSV selection")
    parser.add_argument("--model", type=Path, help="shared D1-D3 model; otherwise use stage 3 or the existing calibrated model")
    parser.add_argument("--trigger-rate-dir", type=Path, action="append")
    parser.add_argument("--prepost-ratio-csv", type=Path, action="append")
    parser.add_argument("--secret-file", default=config.DEFAULT_SECRET_FILE)
    parser.add_argument("--lumi-cache", type=Path)
    parser.add_argument("--fetch-lumi-only", action="store_true", help="populate the LB cache without integrating")
    parser.add_argument("--lumi-tag", default="OflLumi-Run3-010")
    parser.add_argument("--lumi-acct-tag", default="OflLumiAcct-Run3-008")
    parser.add_argument("--cool-offline-connection", default="COOLOFL_TRIGGER/CONDBR2")
    args = parser.parse_args(argv)
    runs = read_intervals(args.input, args.run)
    if not runs:
        raise ValueError("input CSV contains no intervals")
    output = (args.output_dir or selection_directory(config.SELECTION_LUMINOSITY_DIR, runs)).resolve()
    prepare_output(output, runs, args.input)
    cache_path = args.lumi_cache or output / "luminosity_blocks.json"
    cache = load_selection_cache(runs, cache_path, args.lumi_tag, args.lumi_acct_tag, args.cool_offline_connection)
    if args.fetch_lumi_only:
        print(f"Wrote LB cache: {cache_path}")
        return 0
    analysis = (args.analysis_dir or selection_directory(config.SELECTION_ANALYSIS_DIR, runs)).resolve()
    analysis_status = analysis / "analysis_status.json"
    if args.model is None and analysis_status.exists() and json.loads(analysis_status.read_text())["status"] in ("running", "failed"):
        raise ValueError(f"stage 3 did not finish successfully in {analysis}; rerun it or supply --model explicitly")
    model_path = args.model or analysis / "summary/deadtime_true_rate_model.json"
    if args.model is None and not model_path.exists():
        model_path = config.MODEL_JSON
    model = json.loads(model_path.read_text())
    if model["model"] != "three_interval_D3_repeated":
        raise ValueError("a three_interval_D3_repeated model is required")
    ratio_paths = args.prepost_ratio_csv
    if ratio_paths is None and (analysis / "prepost_veto_ratios.csv").exists():
        ratio_paths = [analysis / "prepost_veto_ratios.csv"]
    ratios = selected_ratios(runs, ratio_paths, output / "prepost_veto_ratios.csv", args.secret_file,
                             allow_missing=True)
    directories = args.trigger_rate_dir or [
        config.ANALYSIS_DIR / "stable_luminosity/raw_trigger_counts",
        config.ANALYSIS_DIR / "raw_anyveto_measured_prescale_prepost",
        config.ANALYSIS_DIR / "raw_anyveto_trigger_rates",
    ]
    from stable_runlist.effective_luminosity import StableLuminosityCalculator
    report_path = analysis / "analysis_report.json"
    reports = {row["run_number"]: row for row in json.loads(report_path.read_text())} if report_path.exists() else None
    results = []
    for run in runs:
        number = run["run_number"]
        if reports is not None:
            report = reports.get(number, {})
            local_rates = [Path(report["trigger_rate_csv"])] if report.get("status") == "ok" and report.get("trigger_rate_csv") else []
        else:
            local_rates = sorted((analysis / "batch").glob(f"run_{number}_bins_*/run_{number}_trigger_rate.csv"))
        if args.trigger_rate_dir:
            rate_path = find_rate_source(number, directories, analysis / "batch")
        else:
            rate_path = local_rates[0] if local_rates else find_rate_source(number, directories, config.ANALYSIS_DIR / "batch")
        bins = read_count_bins(rate_path) if rate_path else []
        calculator = StableLuminosityCalculator(CachedLuminosityReader(cache["runs"][str(number)]["blocks"]), model)
        result = calculator.calculate(run["stable_list"], bins, ratios.get(number))
        result.update(run_number=number, configuration=run["configuration"],
                      selected_intervals=run["stable_list"],
                      trigger_rate_source=str(rate_path) if rate_path else None)
        results.append(result)
        print(f"run {number}: {result['luminosity_calculation_status']}, "
              f"missing {result['effective_luminosity_missing_time_sec']:.3f} s", flush=True)
    total = sum_luminosity_results(results)
    compact = [compact_luminosity_output(result) for result in results]
    # These products describe the supplied CSV; the legacy final runlist JSON
    # and previously generated analysis results are not modified.
    write_json(output / "luminosity.json", compact)
    rows = [{k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in compact]
    write_table(output / "luminosity.csv", rows, rows[0].keys())
    write_json(output / "luminosity_total.json", dict(total, model_source=str(model_path), input_csv=str(args.input.resolve())))
    print(f"Covered luminosity: {total['luminosity_covered_ub_inv'] / 1e9:.6f} fb^-1; "
          f"effective: {total['effective_luminosity_covered_ub_inv'] / 1e9:.6f} fb^-1")
    print(f"Wrote {output / 'luminosity.csv'} and {output / 'luminosity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
