#!/usr/bin/env python3
"""CLI for building AHCAL stable/good intervals."""

import argparse
import re
import sys
from pathlib import Path

from .config import (
    DEFAULT_DAQ_BIN_SECONDS,
    DEFAULT_DAQ_MAX_GAP_SECONDS,
    DEFAULT_DB,
    DEFAULT_EFFECTIVE_LUMI_ANY_VETOED_MEASUREMENT,
    DEFAULT_EFFECTIVE_LUMI_AUTO_TRIGGERED_MS,
    DEFAULT_EFFECTIVE_LUMI_DEADTIME_MS,
    DEFAULT_EFFECTIVE_LUMI_NON_VETOED_MEASUREMENT,
    DEFAULT_INFLUX_DATABASE,
    DEFAULT_INFLUX_HOST,
    DEFAULT_INFLUX_PASSWORD,
    DEFAULT_INFLUX_PORT,
    DEFAULT_INFLUX_USER,
    DEFAULT_INFLUX_VERIFY_SSL,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_LUMI_ACCT_TAG,
    DEFAULT_LUMI_TAG,
    DEFAULT_POST_VETO_MEASUREMENT,
    DEFAULT_PRE_VETO_MEASUREMENT,
    DEFAULT_REQUIRED_MEASUREMENTS,
    DEFAULT_SECRET_FILE,
    DEFAULT_VETO_COUNTER_POST_RUN_SECONDS,
)
from .daq_conditions import DAQConditionReader, PrePostVetoRatioReader
from .effective_luminosity import EffectiveLuminosityCalculator
from .time_intervals import interval_entry, merge_intervals, subtract_intervals
from .json_io import load_runs, write_runs
from .lhc_stable import LHCStableReader
from .luminosity_gaps import LuminosityGapReader
from .raw_daq_conditions import RawDAQConditionReader
from .time_intervals import format_time, parse_time


class StableRunListReader:
    def __init__(self, args):
        self.args = args
        self.run_filter_re = (
            re.compile(args.run_config_filter_regex)
            if args.run_config_filter_regex
            else None
        )
        self.allowed_config_re = (
            re.compile(args.allowed_config_regex)
            if args.allowed_config_regex
            else None
        )
        self.lhc_reader = None
        self.daq_reader = None
        self.lumi_reader = None
        self.veto_ratio_reader = None

    def run(self, runs=None, use_input_intervals=False):
        runs = load_runs(Path(self.args.input)) if runs is None else runs
        self.use_input_intervals = use_input_intervals
        enriched = []

        if not use_input_intervals:
            self.lhc_reader = LHCStableReader(self.args.db, verbose=self.args.verbose)
        required_measurements = (
            self.args.required_measurement
            if self.args.required_measurement
            else list(DEFAULT_REQUIRED_MEASUREMENTS)
        )

        if not self.args.no_daq:
            if self.args.daq_source == "raw":
                self.daq_reader = RawDAQConditionReader(
                    raw_patterns=self.args.raw_pattern,
                    bin_seconds=self.args.daq_bin_seconds,
                    max_gap_seconds=self.args.daq_max_gap_seconds,
                    recent_id_limit=self.args.raw_recent_id_limit,
                    reorder_tolerance=self.args.raw_reorder_tolerance,
                    verbose=self.args.verbose,
                )
            else:
                self.daq_reader = DAQConditionReader(
                    host=self.args.influx_host,
                    port=self.args.influx_port,
                    username=self.args.influx_user,
                    password=self.args.influx_password,
                    database=self.args.influx_database,
                    secret_file=self.args.secret_file,
                    bin_seconds=self.args.daq_bin_seconds,
                    max_gap_seconds=self.args.daq_max_gap_seconds,
                    required_measurements=required_measurements,
                    verify_ssl=self.args.influx_verify_ssl,
                    verbose=self.args.verbose,
                )
        if not self.args.no_lumi:
            self.lumi_reader = LuminosityGapReader(
                tag=self.args.lumi_tag,
                acct_tag=self.args.lumi_acct_tag,
                verbose=self.args.verbose,
            )
        if self.args.prepost_veto_ratio:
            self.veto_ratio_reader = PrePostVetoRatioReader(
                host=self.args.influx_host,
                port=self.args.influx_port,
                username=self.args.influx_user,
                password=self.args.influx_password,
                database=self.args.influx_database,
                secret_file=self.args.secret_file,
                pre_veto_measurement=self.args.pre_veto_measurement,
                post_veto_measurement=self.args.post_veto_measurement,
                post_run_seconds=self.args.veto_counter_post_run_seconds,
                verify_ssl=self.args.influx_verify_ssl,
                verbose=self.args.verbose,
            )

        try:
            for run in runs:
                output_run = self.process_run(run)
                if output_run is not None:
                    enriched.append(output_run)
        finally:
            if self.lhc_reader is not None:
                self.lhc_reader.close()
            if self.lumi_reader is not None:
                self.lumi_reader.close()

        if self.lumi_reader is not None and enriched:
            stable_time = sum(run.get("stable_time_sec", 0.0) for run in enriched)
            good_time = sum(run.get("good_time_sec", 0.0) for run in enriched)
            if stable_time > 0 and good_time == 0 and all(
                self.has_only_luminosity_exclusions(run) for run in enriched
            ):
                raise RuntimeError(
                    "ATLAS luminosity lookup excluded every stable interval. "
                    "This usually means COOL/Frontier access failed. Rerun with "
                    "--no-lumi, or fix the ATLAS/Athena COOL environment."
                )

        write_runs(Path(self.args.output or self.args.input), enriched)
        return enriched

    def process_run(self, run):
        run_number = int(run["run_number"])
        configuration = str(run.get("configuration", ""))

        if self.run_filter_re and not self.run_filter_re.search(configuration):
            if self.args.verbose:
                print(f"Skipping run {run_number}: configuration {configuration}")
            return None

        since = parse_time(str(run["start_time"]))
        until = parse_time(str(run["stop_time"]))
        if since >= until:
            if self.args.verbose:
                print(f"Skipping run {run_number}: invalid time interval")
            return None

        if self.args.verbose:
            print(
                f"Processing run {run_number}: "
                f"{format_time(since)} - {format_time(until)}"
            )

        if getattr(self, "use_input_intervals", False):
            stable_list = run["stable_list"]
        else:
            assert self.lhc_reader is not None
            stable_list = self.lhc_reader.stable_intervals(since, until)
        if not stable_list:
            if self.args.verbose:
                print(f"Skipping run {run_number}: no stable beams")
            return None

        excluded_list = list(run.get("excluded_list", []))
        if self.allowed_config_re and not self.allowed_config_re.search(configuration):
            excluded_list.extend(
                interval_entry(
                    stable["start_utime"],
                    stable["stop_utime"],
                    f"Run config {configuration}",
                )
                for stable in stable_list
            )
        if self.lumi_reader is not None:
            excluded_list.extend(self.lumi_reader.excluded_intervals(stable_list))
        if self.daq_reader is not None:
            prepare_run = getattr(self.daq_reader, "prepare_run", None)
            if prepare_run is not None:
                prepare_run(run_number, since, until)
            excluded_list.extend(self.daq_reader.excluded_intervals(stable_list))
        excluded_list = merge_intervals(excluded_list)

        good_list = subtract_intervals(stable_list, excluded_list)
        stable_time_sec = sum(
            item["stop_utime"] - item["start_utime"] for item in stable_list
        )
        excluded_time_sec = sum(
            item["stop_utime"] - item["start_utime"] for item in excluded_list
        )
        good_time_sec = sum(
            item["stop_utime"] - item["start_utime"] for item in good_list
        )
        effective_lumi = {}
        if self.args.effective_lumi:
            if self.lumi_reader is None:
                raise RuntimeError("--effective-lumi requires luminosity DB access")
            if self.daq_reader is None:
                raise RuntimeError("--effective-lumi requires a DAQ data source")
            calculator = EffectiveLuminosityCalculator(
                lumi_reader=self.lumi_reader,
                daq_reader=self.daq_reader,
                any_vetoed_measurement=self.args.effective_lumi_any_vetoed_measurement,
                non_vetoed_measurement=self.args.effective_lumi_non_vetoed_measurement,
                deadtime_ms=self.args.effective_lumi_deadtime_ms,
                auto_triggered_ms=self.args.effective_lumi_auto_triggered_ms,
                verbose=self.args.verbose,
            )
            effective_lumi = calculator.calculate(good_list)
        prepost_veto_summary = None
        if self.veto_ratio_reader is not None:
            prepost_veto_summary = self.veto_ratio_reader.run_summary(since, until)

        output_run = {"run_number": run_number}
        for key, value in run.items():
            if key not in (
                "RunNumber",
                "run_number",
                "stable_list",
                "excluded_list",
                "good_list",
                "stable_time_sec",
                "excluded_time_sec",
                "good_time_sec",
                "luminosity_blocks",
                "luminosity_ub_inv",
                "luminosity_pb_inv",
                "effective_luminosity_ub_inv",
                "effective_luminosity_pb_inv",
                "daq_lumi_live_fraction",
                "effective_luminosity_missing_time_sec",
                "raw_trigger_summary",
                "prepost_veto_summary",
            ):
                output_run[key] = value
        output_run["start_utime"] = since
        output_run["stop_utime"] = until
        output_run["stable_list"] = stable_list
        output_run["excluded_list"] = excluded_list
        output_run["good_list"] = good_list
        output_run["stable_time_sec"] = stable_time_sec
        output_run["excluded_time_sec"] = excluded_time_sec
        output_run["good_time_sec"] = good_time_sec
        if isinstance(self.daq_reader, RawDAQConditionReader):
            output_run["raw_trigger_summary"] = self.daq_reader.run_summary()
        if prepost_veto_summary is not None:
            output_run["prepost_veto_summary"] = prepost_veto_summary
        output_run.update(effective_lumi)
        return output_run

    @staticmethod
    def has_only_luminosity_exclusions(run):
        excluded = run.get("excluded_list", [])
        if not excluded:
            return False
        for item in excluded:
            reason = item.get("reason", "")
            if (
                "Missing ATLAS luminosity" not in reason
                and "Missing ATLAS LBTIME" not in reason
            ):
                return False
        return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Read ahcal_run_times.json, attach LHC STABLE BEAMS intervals, "
            "DAQ/luminosity/config excluded intervals, and AHCAL good intervals."
        )
    )
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="input JSON path")
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help="output JSON path (default: %(default)s)",
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="LHC BeamData COOL sqlite DB")
    parser.add_argument(
        "--run-config-filter-regex",
        default="",
        help="only process runs whose configuration matches this regex; empty disables",
    )
    parser.add_argument(
        "--allowed-config-regex",
        "--physics-config-regex",
        default="AHCALPhysics",
        dest="allowed_config_regex",
        help="configs not matching this regex are excluded over the stable-beam interval",
    )
    parser.add_argument("--no-lumi", action="store_true", help="do not query ATLAS luminosity DB")
    parser.add_argument(
        "--lumi-tag",
        default=DEFAULT_LUMI_TAG,
        help="ATLAS offline luminosity tag used to check missing luminosity",
    )
    parser.add_argument(
        "--lumi-acct-tag",
        default=DEFAULT_LUMI_ACCT_TAG,
        help=(
            "use /TRIGGER/OFLLUMI/LumiAccounting with this tag instead of "
            "OflPrefLumi/LBTIME, matching faser_luminosity.py --acct"
        ),
    )
    parser.add_argument(
        "--no-daq",
        action="store_true",
        help="disable the configured InfluxDB/raw DAQ check",
    )
    parser.add_argument(
        "--daq-source",
        choices=("influx", "raw"),
        default="influx",
        help="DAQ information source (default: influx)",
    )
    parser.add_argument("--influx-host", default=DEFAULT_INFLUX_HOST)
    parser.add_argument("--influx-port", default=DEFAULT_INFLUX_PORT, type=int)
    parser.add_argument("--influx-user", default=DEFAULT_INFLUX_USER)
    parser.add_argument("--influx-password", default=DEFAULT_INFLUX_PASSWORD)
    parser.add_argument("--influx-database", default=DEFAULT_INFLUX_DATABASE)
    parser.add_argument(
        "--influx-no-verify",
        action="store_false",
        dest="influx_verify_ssl",
        default=DEFAULT_INFLUX_VERIFY_SSL,
        help="disable HTTPS certificate verification for InfluxDB",
    )
    parser.add_argument(
        "--influx-verify",
        action="store_true",
        dest="influx_verify_ssl",
        help="enable HTTPS certificate verification for InfluxDB",
    )
    parser.add_argument("--secret-file", default=DEFAULT_SECRET_FILE)
    parser.add_argument(
        "--prepost-veto-ratio",
        action="store_true",
        help=(
            "read run-final TLU PreVeto/PostVeto counters from InfluxDB and "
            "write both ratios and the veto fraction"
        ),
    )
    parser.add_argument(
        "--pre-veto-measurement",
        default=DEFAULT_PRE_VETO_MEASUREMENT,
        help="InfluxDB measurement for the TLU PreVeto counter",
    )
    parser.add_argument(
        "--post-veto-measurement",
        default=DEFAULT_POST_VETO_MEASUREMENT,
        help="InfluxDB measurement for the TLU PostVeto counter",
    )
    parser.add_argument(
        "--veto-counter-post-run-seconds",
        default=DEFAULT_VETO_COUNTER_POST_RUN_SECONDS,
        type=float,
        help="seconds after run stop searched for final TLU counter samples",
    )
    parser.add_argument("--daq-bin-seconds", default=DEFAULT_DAQ_BIN_SECONDS, type=float)
    parser.add_argument(
        "--daq-max-gap-seconds",
        default=DEFAULT_DAQ_MAX_GAP_SECONDS,
        type=float,
        help="DAQ counter gaps larger than this are excluded",
    )
    parser.add_argument(
        "--required-measurement",
        action="append",
        default=None,
        help="InfluxDB counter measurement required during stable beams; repeatable",
    )
    parser.add_argument(
        "--raw-pattern",
        action="append",
        default=[],
        help=(
            "raw-file glob for --daq-source raw; supports {run} and {run06}; "
            "repeatable"
        ),
    )
    parser.add_argument(
        "--raw-recent-id-limit",
        default=100_000,
        type=int,
        help="number of recent TriggerIDs retained for duplicate detection",
    )
    parser.add_argument(
        "--raw-reorder-tolerance",
        default=1024,
        type=int,
        help="maximum backward TriggerID step treated as event reordering",
    )
    parser.add_argument(
        "--effective-lumi",
        action="store_true",
        help=(
            "calculate effective luminosity by multiplying ATLAS lumiblock "
            "luminosity by the DAQ live fraction Rreco/Rtrue"
        ),
    )
    parser.add_argument(
        "--effective-lumi-any-vetoed-measurement",
        default=DEFAULT_EFFECTIVE_LUMI_ANY_VETOED_MEASUREMENT,
        help="InfluxDB counter used as the AnyVetoed term in Rreco",
    )
    parser.add_argument(
        "--effective-lumi-non-vetoed-measurement",
        default=DEFAULT_EFFECTIVE_LUMI_NON_VETOED_MEASUREMENT,
        help="InfluxDB counter used as the NonVetoed term in Rreco",
    )
    parser.add_argument(
        "--effective-lumi-deadtime-ms",
        default=DEFAULT_EFFECTIVE_LUMI_DEADTIME_MS,
        type=float,
        help="DAQ deadtime D in ms for the Rreco/Rtrue bisection formula",
    )
    parser.add_argument(
        "--effective-lumi-auto-triggered-ms",
        default=DEFAULT_EFFECTIVE_LUMI_AUTO_TRIGGERED_MS,
        type=float,
        help="DAQ auto-triggered window L in ms for the Rreco/Rtrue bisection formula",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.no_daq and args.daq_source == "raw" and not args.raw_pattern:
        parser.error("--daq-source raw requires at least one --raw-pattern")
    if args.raw_recent_id_limit <= 0:
        parser.error("--raw-recent-id-limit must be > 0")
    if args.raw_reorder_tolerance < 0:
        parser.error("--raw-reorder-tolerance must be >= 0")
    if args.veto_counter_post_run_seconds < 0:
        parser.error("--veto-counter-post-run-seconds must be >= 0")
    if args.effective_lumi:
        if args.no_lumi:
            parser.error("--effective-lumi cannot be used with --no-lumi")
        if args.no_daq:
            parser.error("--effective-lumi cannot be used with --no-daq")
        if args.effective_lumi_deadtime_ms < 0:
            parser.error("--effective-lumi-deadtime-ms must be >= 0")
        if args.effective_lumi_auto_triggered_ms <= 0:
            parser.error("--effective-lumi-auto-triggered-ms must be > 0")
    return args


def main():
    args = parse_args()
    try:
        output = StableRunListReader(args).run()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    total_good = sum(run.get("good_time_sec", 0.0) for run in output)
    message = (
        f"wrote {len(output)} runs to {args.output or args.input} "
        f"(total AHCAL good time: {total_good:.1f} s)"
    )
    if args.effective_lumi:
        total_lumi = sum(run.get("luminosity_pb_inv", 0.0) for run in output)
        total_eff_lumi = sum(
            run.get("effective_luminosity_pb_inv", 0.0) for run in output
        )
        message += (
            f", lumi: {total_lumi:.6f} pb^-1, "
            f"effective lumi: {total_eff_lumi:.6f} pb^-1"
        )
    if args.prepost_veto_ratio:
        total_pre = sum(
            run["prepost_veto_summary"]["pre_veto_triggers"] for run in output
        )
        total_post = sum(
            run["prepost_veto_summary"]["post_veto_triggers"] for run in output
        )
        ratio = total_post / total_pre if total_pre else None
        ratio_text = f"{ratio:.8f}" if ratio is not None else "undefined"
        message += (
            f", PreVeto: {total_pre:,}, PostVeto: {total_post:,}, "
            f"Post/Pre: {ratio_text}"
        )
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
