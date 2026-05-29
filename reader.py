import argparse
import re
from pathlib import Path

from config import DEFAULT_REQUIRED_MEASUREMENTS
from daq_conditions import DAQConditionReader
from intervals import interval_entry, merge_intervals, subtract_intervals
from json_io import load_runs, write_runs
from lhc_stable import LHCStableReader
from luminosity_gaps import LuminosityGapReader
from time_utils import format_time, parse_time


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

    def run(self):
        runs = load_runs(Path(self.args.input))
        enriched = []

        self.lhc_reader = LHCStableReader(self.args.db, verbose=self.args.verbose)
        required_measurements = (
            self.args.required_measurement
            if self.args.required_measurement
            else list(DEFAULT_REQUIRED_MEASUREMENTS)
        )

        if not self.args.no_daq:
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

        try:
            for run in runs:
                output_run = self.process_run(run)
                if output_run is not None:
                    enriched.append(output_run)
        finally:
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
            print(f"Processing run {run_number}: {format_time(since)} - {format_time(until)}")

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
        return output_run

    @staticmethod
    def has_only_luminosity_exclusions(run):
        excluded = run.get("excluded_list", [])
        if not excluded:
            return False
        for item in excluded:
            reason = item.get("reason", "")
            if "Missing ATLAS luminosity" not in reason and "Missing ATLAS LBTIME" not in reason:
                return False
        return True
