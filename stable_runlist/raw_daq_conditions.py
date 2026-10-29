"""DAQ-condition and live-fraction source backed by AHCAL raw files."""

from dataclasses import asdict
import math

from .daq_conditions import solve_true_rate_from_reco
from .time_intervals import interval_entry, merge_intervals
from .raw_trigger_rate import (
    RateAccumulator,
    MergedRawEventReader,
    TriggerDeduplicator,
    discover_raw_files,
)


class RawDAQConditionReader:
    def __init__(
        self,
        raw_patterns,
        bin_seconds,
        max_gap_seconds,
        recent_id_limit=100_000,
        reorder_tolerance=1024,
        verbose=False,
    ):
        if not raw_patterns:
            raise ValueError("raw DAQ source requires at least one raw file pattern")
        if bin_seconds <= 0:
            raise ValueError("raw DAQ bin size must be positive")
        if max_gap_seconds <= 0:
            raise ValueError("raw DAQ maximum gap must be positive")
        self.raw_patterns = list(raw_patterns)
        self.bin_seconds = float(bin_seconds)
        self.max_gap_seconds = float(max_gap_seconds)
        self.recent_id_limit = recent_id_limit
        self.reorder_tolerance = reorder_tolerance
        self.verbose = verbose
        self._active_run = None
        self._cache = {}

    def prepare_run(self, run_number, since, until):
        del since, until  # raw files contain their own absolute CPU timestamps
        run_number = int(run_number)
        if run_number not in self._cache:
            self._cache = {run_number: self._read_run(run_number)}
        self._active_run = run_number

    def _read_run(self, run_number):
        paths = discover_raw_files(run_number, self.raw_patterns)
        if self.verbose:
            print(f"[raw DAQ] run {run_number}: {len(paths)} file(s)")
            for path in paths:
                print(f"[raw DAQ]   {path}")

        accumulator = RateAccumulator(self.bin_seconds)
        deduplicator = TriggerDeduplicator(
            recent_id_limit=self.recent_id_limit,
            reorder_tolerance=self.reorder_tolerance,
        )
        wrong_run_events = 0
        merged_reader = MergedRawEventReader(paths)
        for event in merged_reader:
            if event.run_number != run_number:
                wrong_run_events += 1
                continue
            decision = deduplicator.process(event)
            if not decision.duplicate:
                accumulator.add(event)

        return {
            "paths": paths,
            "accumulator": accumulator,
            "deduplicator": deduplicator,
            "read_stats": merged_reader.stats(),
            "wrong_run_events": wrong_run_events,
        }

    def _active_data(self):
        if self._active_run is None:
            raise RuntimeError("raw DAQ run was not prepared")
        return self._cache[self._active_run]

    def excluded_intervals(self, stable_list):
        data = self._active_data()
        if not data["paths"]:
            return [
                interval_entry(
                    stable["start_utime"],
                    stable["stop_utime"],
                    "Missing AHCAL raw files",
                )
                for stable in stable_list
            ]

        excluded = []
        for stable in stable_list:
            excluded.extend(
                self._gap_intervals(
                    data["accumulator"],
                    stable["start_utime"],
                    stable["stop_utime"],
                )
            )
        return merge_intervals(excluded)

    def _gap_intervals(self, accumulator, since, until):
        bins = accumulator.bins(since=since, until=until)
        nonempty = [item for item in bins if item.count > 0]
        if not nonempty:
            return [
                interval_entry(since, until, "Missing AHCAL raw trigger data")
            ]

        excluded = []
        gap_start = since
        for item in nonempty:
            event_bin_start = max(since, item.start_utime)
            if event_bin_start - gap_start > self.max_gap_seconds:
                excluded.append(
                    interval_entry(
                        gap_start,
                        event_bin_start,
                        "AHCAL raw trigger gap",
                    )
                )
            gap_start = min(until, item.stop_utime)
        if until - gap_start > self.max_gap_seconds:
            excluded.append(
                interval_entry(gap_start, until, "AHCAL raw trigger gap")
            )
        return excluded

    def live_fraction_series(
        self,
        since,
        until,
        any_vetoed_measurement,
        non_vetoed_measurement,
        deadtime_s,
        auto_triggered_s,
    ):
        # Measurement names apply only to InfluxDB.  The raw source counts all
        # unique recorded TLU TriggerIDs present in the DAQ stream.
        del any_vetoed_measurement, non_vetoed_measurement
        data = self._active_data()
        output = []
        for item in data["accumulator"].bins(since=since, until=until):
            r_reco = item.rate_hz
            r_true = solve_true_rate_from_reco(
                r_reco,
                deadtime_s,
                auto_triggered_s,
            )
            if not math.isfinite(r_true) or r_true <= 0:
                output.append((item.start_utime, math.nan))
            else:
                output.append((item.start_utime, r_reco / r_true))
        return output

    def run_summary(self):
        data = self._active_data()
        accumulator = data["accumulator"]
        first_utime = accumulator.first_time
        last_utime = accumulator.last_time
        return {
            "source": "raw",
            "raw_files": [str(path) for path in data["paths"]],
            "bin_seconds": self.bin_seconds,
            "recent_id_limit": self.recent_id_limit,
            "reorder_tolerance": self.reorder_tolerance,
            "first_trigger_utime": first_utime,
            "last_trigger_utime": last_utime,
            "wrong_run_events": data["wrong_run_events"],
            **data["read_stats"],
            **asdict(data["deduplicator"].stats),
            "trigger_id_gaps": data[
                "deduplicator"
            ].trigger_id_gap_details(),
            "eid_cycle_matching_loss_details": data[
                "deduplicator"
            ].eid_cycle_matching_loss_details(),
        }
