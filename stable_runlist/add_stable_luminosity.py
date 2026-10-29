#!/usr/bin/env python3
"""Add LB-based luminosities to an existing run list, preserving its selections."""

from .config import OUTPUT_DIR, RAW_RATE_DIR, RUNLIST_JSON, BACKUP_DIR, MODEL_JSON

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import tempfile

HERE = OUTPUT_DIR
DEFAULT_MODEL = MODEL_JSON
DEFAULT_RATIOS = HERE / "deadtime_trigger_rate/summary_runs_csv_by_config/prepost_veto_ratios.csv"
ADDITIONAL_RATIOS = DEFAULT_RATIOS.with_name("prepost_veto_ratios_noveto.csv")
EXTRA_RATIOS = HERE / "deadtime_trigger_rate/stable_luminosity/prepost_veto_ratios_additional.csv"
REMAINING_RATIOS = HERE / "deadtime_trigger_rate/stable_luminosity/prepost_veto_ratios_remaining.csv"
DEFAULT_CACHE = HERE / "deadtime_trigger_rate/stable_luminosity/luminosity_blocks.json"


def write_json(path, data):
    """Replace only after a complete, finite JSON document has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                     encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink()
            raise
    temporary.replace(path)


def compact_luminosity_output(record):
    """Keep combined uncertainties; omit their components and calculation metadata."""
    prefixes = (
        "effective_luminosity_stat_", "effective_luminosity_covered_stat_",
        "effective_luminosity_deadtime_sys_", "effective_luminosity_covered_deadtime_sys_",
    )
    result = {key: value for key, value in record.items()
              if key != "luminosity_calculation" and not key.startswith(prefixes)}
    if "luminosity_blocks" in result:
        result["luminosity_blocks"] = [compact_luminosity_output(block)
                                       for block in result["luminosity_blocks"]]
    return result


def selection(run):
    return [[x["start_utime"], x["stop_utime"]] for x in run["stable_list"]]


def fetch_luminosity(runs, cache_path, tag, acct_tag,
                     offline_database="COOLOFL_TRIGGER/CONDBR2"):
    from .luminosity_gaps import LuminosityGapReader
    source = {"lumi_tag": tag, "lumi_acct_tag": acct_tag}
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {"source": source, "runs": {}}
    if cache["source"] != source:
        raise ValueError("luminosity cache tags differ; use a different --lumi-cache")
    reader = LuminosityGapReader(tag, acct_tag, offline_database=offline_database)
    try:
        for index, run in enumerate(runs, 1):
            key = str(run["run_number"])
            old = cache["runs"].get(key)
            if old is not None and old["stable_intervals"] == selection(run):
                continue
            blocks = reader.luminosity_blocks(run["stable_list"])
            # Empty queries may indicate a failed DB connection. Never cache a
            # failure as a successful empty result for a nonempty selection.
            if run["stable_list"] and not blocks:
                raise RuntimeError(f"run {key}: no luminosity records returned")
            cache["runs"][key] = {"stable_intervals": selection(run), "blocks": blocks}
            write_json(cache_path, cache)
            print(f"LB fetch {index}/{len(runs)}: run {key}, {len(blocks)} records", flush=True)
    finally:
        reader.close()
    return cache


class CachedLuminosityReader:
    def __init__(self, blocks):
        self.blocks = blocks

    def luminosity_blocks(self, stable_list):
        return self.blocks


def read_count_bins(path):
    """Use only raw counts and physical bin duration, ignoring corrected columns."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"start_utime", "stop_utime", "count"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: raw count/time columns are required")
        result = []
        for row in reader:
            start, stop, count = (float(row[k]) for k in ("start_utime", "stop_utime", "count"))
            if not all(math.isfinite(x) for x in (start, stop, count)) or stop <= start or count < 0:
                raise ValueError(f"{path}: invalid count or bin duration")
            result.append({"start": start, "stop": stop, "rate": count / (stop-start),
                           "stat_error": math.sqrt(count) / (stop-start)})
    return result


def find_rate_source(run_number, directories, batch_dir):
    candidates = [d / f"run_{run_number}_trigger_rate.csv" for d in directories]
    candidates.extend(sorted(batch_dir.glob(f"run_{run_number}_bins_*/run_{run_number}_trigger_rate.csv")))
    candidates.append(RAW_RATE_DIR / f"run{run_number}_trigger_rate.csv")
    for path in candidates:
        if path.is_file():
            return path
    return None


def read_ratios(paths):
    ratios = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("status", "ok") != "ok":
                    continue
                key, value = int(row["run_number"]), float(row["pre_over_post_ratio"])
                if not math.isfinite(value) or value < 1:
                    raise ValueError(f"{path}: invalid Pre/Post for run {key}")
                if key in ratios and not math.isclose(ratios[key], value, rel_tol=1e-12):
                    raise ValueError(f"conflicting Pre/Post values for run {key}")
                ratios[key] = value
    return ratios


def contained(requested, available):
    """Allow a previously fetched stable cache to serve a good sub-selection."""
    available = sorted(available)
    for start, stop in requested:
        cursor = start
        for left, right in available:
            if left <= cursor < right:
                cursor = max(cursor, right)
            if cursor >= stop:
                break
        if cursor < stop:
            return False
    return True


def load_selection_cache(runs, path, tag, acct_tag, connection):
    source = {"lumi_tag": tag, "lumi_acct_tag": acct_tag}
    cache = json.loads(path.read_text()) if path.exists() else {"source": source, "runs": {}}
    if cache["source"] != source:
        raise ValueError("luminosity cache tags differ; select another --lumi-cache")
    # Seed a separate selection cache from the existing shared cache without
    # changing that cache or querying COOL for intervals it already contains.
    shared = json.loads(DEFAULT_CACHE.read_text()) if DEFAULT_CACHE.exists() else {}
    candidates = [cache]
    if shared.get("source") == source:
        candidates.append(shared)
    for run in runs:
        key, intervals = str(run["run_number"]), selection(run)
        for candidate in candidates:
            old = candidate.get("runs", {}).get(key)
            if old is not None and contained(intervals, old["stable_intervals"]):
                cache["runs"][key] = dict(old, stable_intervals=intervals)
                break
    write_json(path, cache)
    if any(cache["runs"].get(str(r["run_number"]), {}).get("stable_intervals") != selection(r) for r in runs):
        cache = fetch_luminosity(runs, path, tag, acct_tag, connection)
    return cache


def sum_luminosity_results(results):
    """Common D1-D3 shifts add coherently; run count statistics add in quadrature."""
    center = sum(r["effective_luminosity_covered_ub_inv"] for r in results)
    low = sum(r["effective_luminosity_covered_at_deadtime_low_ub_inv"] for r in results)
    high = sum(r["effective_luminosity_covered_at_deadtime_high_ub_inv"] for r in results)
    complete = all(r["luminosity_calculation_status"] == "complete" for r in results)
    delivered = sum(r["luminosity_covered_ub_inv"] for r in results)
    output = {
        "runs": len(results), "complete_runs": sum(r["luminosity_calculation_status"] == "complete" for r in results),
        "luminosity_covered_ub_inv": delivered,
        "luminosity_ub_inv": delivered if all(r["luminosity_ub_inv"] is not None for r in results) else None,
        "effective_luminosity_covered_ub_inv": center,
        "effective_luminosity_ub_inv": center if complete else None,
        "luminosity_missing_time_sec": sum(r["luminosity_missing_time_sec"] for r in results),
        "effective_luminosity_missing_time_sec": sum(r["effective_luminosity_missing_time_sec"] for r in results),
    }
    for side, systematic in (("minus", max(0, center - min(low, high))),
                             ("plus", max(0, max(low, high) - center))):
        stat = math.sqrt(sum(r[f"effective_luminosity_covered_stat_{side}_ub_inv"] ** 2 for r in results))
        error = math.hypot(stat, systematic)
        output[f"effective_luminosity_covered_total_error_{side}_ub_inv"] = error
        output[f"effective_luminosity_total_error_{side}_ub_inv"] = error if complete else None
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, default=RUNLIST_JSON)
    parser.add_argument("--output", type=Path, help="defaults to updating --run-json, with a backup")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prepost-ratio-csv", type=Path, action="append")
    parser.add_argument("--trigger-rate-dir", type=Path, action="append")
    parser.add_argument("--batch-dir", type=Path, default=HERE / "deadtime_trigger_rate/batch")
    parser.add_argument("--lumi-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fetch-lumi", action="store_true", help="fetch missing cache entries using PyCool")
    parser.add_argument("--fetch-lumi-only", action="store_true", help="populate the cache without modifying the run JSON")
    parser.add_argument("--lumi-tag", default="OflLumi-Run3-010")
    parser.add_argument("--lumi-acct-tag", default="OflLumiAcct-Run3-008")
    parser.add_argument("--cool-offline-connection", default="COOLOFL_TRIGGER/CONDBR2",
                        help="optional explicit read-only COOL connection, e.g. frontier://...")
    args = parser.parse_args()
    runs = json.loads(args.run_json.read_text())
    if args.fetch_lumi or args.fetch_lumi_only:
        cache = fetch_luminosity(runs, args.lumi_cache, args.lumi_tag, args.lumi_acct_tag,
                                args.cool_offline_connection)
    else:
        cache = json.loads(args.lumi_cache.read_text())
    if args.fetch_lumi_only:
        return

    from .effective_luminosity import StableLuminosityCalculator
    model = json.loads(args.model.read_text())
    if model["model"] != "three_interval_D3_repeated":
        raise ValueError("a three_interval_D3_repeated model is required")
    ratio_paths = args.prepost_ratio_csv or [p for p in
                    (DEFAULT_RATIOS, ADDITIONAL_RATIOS, EXTRA_RATIOS, REMAINING_RATIOS) if p.exists()]
    ratios = read_ratios(ratio_paths)
    directories = args.trigger_rate_dir or [HERE / "deadtime_trigger_rate/stable_luminosity/raw_trigger_counts",
                                            HERE / "deadtime_trigger_rate/raw_anyveto_measured_prescale_prepost",
                                            HERE / "deadtime_trigger_rate/raw_anyveto_trigger_rates"]
    output_runs = []
    for run in runs:
        number = int(run["run_number"])
        cached = cache["runs"].get(str(number))
        if cached is None or cached["stable_intervals"] != selection(run):
            raise ValueError(f"run {number}: luminosity cache is missing or selection changed; use --fetch-lumi")
        source = find_rate_source(number, directories, args.batch_dir)
        bins = read_count_bins(source) if source else []
        ratio = ratios.get(number)
        calculator = StableLuminosityCalculator(CachedLuminosityReader(cached["blocks"]), model)
        result = calculator.calculate(run["stable_list"], bins, ratio)
        output_runs.append(compact_luminosity_output(dict(run, **result)))
        print(f"run {number}: {result['luminosity_calculation_status']}, "
              f"missing {result['effective_luminosity_missing_time_sec']:.3f} s", flush=True)
    output = args.output or args.run_json
    if output.resolve() == args.run_json.resolve():
        backup = (BACKUP_DIR / output.with_suffix(".before_stable_luminosity.json").name
                  if output.resolve() == RUNLIST_JSON.resolve()
                  else output.with_suffix(".before_stable_luminosity.json"))
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(output, backup)
    write_json(output, output_runs)
    print(f"Wrote {output}: {sum(r['luminosity_calculation_status'] == 'complete' for r in output_runs)}/{len(runs)} complete")


if __name__ == "__main__":
    main()
