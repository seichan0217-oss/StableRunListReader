#!/usr/bin/env python3
#  ./raw_trigger_rate.py --run 22163 \
#    --raw-pattern '/eos/experiment/faser/raw/2026/{run06}/FaserAHCAL-
#    Physics-{run06}-*.raw' \
#    --bin-seconds 2 \
#    --output-prefix run22163_trigger_rate


"""Calculate AHCAL recorded-trigger rates directly from DAQ raw files."""

from .config import RAW_RATE_DIR, OUTPUT_DIR, RUNLIST_JSON

import argparse
from bisect import bisect_right
import csv
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import glob
import heapq
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Optional

from .ahcal_raw_reader import RawFileReader, RawFormatError, RawTriggerEvent


EID_CYCLE_JUMP_MIN = 65_530
DEFAULT_OUTPUT_PREFIX = str(RAW_RATE_DIR / "raw_trigger_rate")


@dataclass(frozen=True)
class TriggerDecision:
    event: RawTriggerEvent
    segment: int
    duplicate: bool
    duplicate_reason: str = ""


@dataclass
class TriggerProcessingStats:
    input_triggers: int = 0
    unique_triggers: int = 0
    duplicate_triggers: int = 0
    duplicate_timestamp_conflicts: int = 0
    counter_segments: int = 0
    counter_resets: int = 0
    counter_rollovers: int = 0
    small_out_of_order_ids: int = 0
    first_trigger_id: Optional[int] = None
    last_trigger_id: Optional[int] = None
    first_cpu_timestamp_us: Optional[int] = None
    last_cpu_timestamp_us: Optional[int] = None
    missing_trigger_ids_in_physics_raw: int = 0
    trigger_id_gap_count: int = 0
    largest_trigger_id_gap: int = 0
    eid_cycle_matching_losses: int = 0


@dataclass(frozen=True)
class TriggerIDGap:
    segment: int
    first_missing_trigger_id: int
    last_missing_trigger_id: int
    previous_trigger_id: int
    next_trigger_id: int
    previous_cpu_timestamp_us: int
    next_cpu_timestamp_us: int

    @property
    def missing_count(self):
        return self.last_missing_trigger_id - self.first_missing_trigger_id + 1

    def as_dict(self):
        return {
            "segment": self.segment,
            "previous_trigger_id": self.previous_trigger_id,
            "next_trigger_id": self.next_trigger_id,
            "first_missing_trigger_id": self.first_missing_trigger_id,
            "last_missing_trigger_id": self.last_missing_trigger_id,
            "missing_count": self.missing_count,
            "previous_utime": self.previous_cpu_timestamp_us / 1_000_000.0,
            "next_utime": self.next_cpu_timestamp_us / 1_000_000.0,
        }


@dataclass(frozen=True)
class EIDCycleMatchingLoss:
    previous_trigger_id: int
    excursion_trigger_id: int
    recovery_trigger_id: int
    previous_event_counter: int
    excursion_event_counter: int
    recovery_event_counter: int
    excursion_cpu_timestamp_us: int

    def as_dict(self):
        return {
            "previous_trigger_id": self.previous_trigger_id,
            "excursion_trigger_id": self.excursion_trigger_id,
            "recovery_trigger_id": self.recovery_trigger_id,
            "previous_event_counter": self.previous_event_counter,
            "excursion_event_counter": self.excursion_event_counter,
            "recovery_event_counter": self.recovery_event_counter,
            "excursion_utime": self.excursion_cpu_timestamp_us / 1_000_000.0,
        }


@dataclass(frozen=True)
class RateBin:
    start_utime: float
    stop_utime: float
    count: int
    rate_hz: float


@dataclass(frozen=True)
class VetoRateBin(RateBin):
    any_vetoed_count: int
    non_vetoed_count: int
    any_vetoed_rate_hz: float
    non_vetoed_rate_hz: float

    @property
    def any_vetoed_fraction(self):
        return self.any_vetoed_count / self.count if self.count else math.nan


class RateAccumulator:
    def __init__(self, bin_seconds):
        if bin_seconds <= 0:
            raise ValueError("bin_seconds must be positive")
        self.bin_seconds = float(bin_seconds)
        self.bin_counts = {}
        self.first_time = None
        self.last_time = None

    def add(self, event):
        timestamp = event.unix_time
        if self.first_time is None or timestamp < self.first_time:
            self.first_time = timestamp
        if self.last_time is None or timestamp > self.last_time:
            self.last_time = timestamp
        bin_index = math.floor(timestamp / self.bin_seconds)
        self.bin_counts[bin_index] = self.bin_counts.get(bin_index, 0) + 1

    def bins(self, since=None, until=None):
        if since is not None:
            first_bin = math.floor(since / self.bin_seconds)
        elif self.first_time is not None:
            first_bin = math.floor(self.first_time / self.bin_seconds)
        else:
            return []
        if until is not None:
            last_bin = math.ceil(until / self.bin_seconds) - 1
        elif self.last_time is not None:
            last_bin = math.floor(self.last_time / self.bin_seconds)
        else:
            return []

        output = []
        for bin_index in range(first_bin, last_bin + 1):
            start = bin_index * self.bin_seconds
            stop = start + self.bin_seconds
            count = self.bin_counts.get(bin_index, 0)
            output.append(
                RateBin(
                    start_utime=start,
                    stop_utime=stop,
                    count=count,
                    rate_hz=count / self.bin_seconds,
                )
            )
        return output


class VetoRateAccumulator(RateAccumulator):
    """Bin recorded triggers by the TLU input-5/6 veto decision."""

    def __init__(self, bin_seconds):
        super().__init__(bin_seconds)
        self.any_vetoed_counts = {}
        self.non_vetoed_counts = {}

    def add(self, event):
        super().add(event)
        bin_index = math.floor(event.unix_time / self.bin_seconds)
        is_any_vetoed = bool(event.inputs[4] or event.inputs[5])
        counts = (
            self.any_vetoed_counts
            if is_any_vetoed
            else self.non_vetoed_counts
        )
        counts[bin_index] = counts.get(bin_index, 0) + 1

    def bins(self, since=None, until=None):
        output = []
        for item in super().bins(since=since, until=until):
            bin_index = math.floor(item.start_utime / self.bin_seconds)
            any_count = self.any_vetoed_counts.get(bin_index, 0)
            non_count = self.non_vetoed_counts.get(bin_index, 0)
            output.append(
                VetoRateBin(
                    start_utime=item.start_utime,
                    stop_utime=item.stop_utime,
                    count=item.count,
                    rate_hz=item.rate_hz,
                    any_vetoed_count=any_count,
                    non_vetoed_count=non_count,
                    any_vetoed_rate_hz=any_count / self.bin_seconds,
                    non_vetoed_rate_hz=non_count / self.bin_seconds,
                )
            )
        return output


def format_raw_pattern(pattern, run_number):
    try:
        return pattern.format(run=run_number, run06=f"{run_number:06d}")
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(
            f"invalid raw pattern {pattern!r}; only {{run}} and {{run06}} "
            "placeholders are supported"
        ) from error


def discover_raw_files(run_number, patterns):
    matches = []
    for pattern in patterns:
        expanded = format_raw_pattern(pattern, run_number)
        matches.extend(glob.glob(expanded))
    # Resolve duplicate spellings/symlinks where possible, while retaining only
    # regular files. Sorting also gives deterministic tie-breaking.
    unique = {}
    for match in matches:
        path = Path(match)
        if path.is_file():
            try:
                key = str(path.resolve())
            except OSError:
                key = str(path.absolute())
            unique[key] = path
    return sorted(unique.values(), key=lambda item: str(item))


class MergedRawEventReader:
    def __init__(self, paths):
        self.readers = [RawFileReader(path) for path in paths]

    def __iter__(self):
        streams = [iter(reader) for reader in self.readers]
        key = lambda event: (
            event.cpu_timestamp_us,
            event.event_counter,
            event.trigger_id,
            event.path,
        )
        yield from heapq.merge(*streams, key=key)

    def stats(self):
        return {
            "raw_events_seen": sum(
                reader.stats.events_seen for reader in self.readers
            ),
            "raw_events_with_tlu": sum(
                reader.stats.events_with_tlu for reader in self.readers
            ),
            "raw_events_without_tlu": sum(
                reader.stats.events_without_tlu for reader in self.readers
            ),
            "malformed_raw_events": sum(
                reader.stats.malformed_events for reader in self.readers
            ),
        }


def iter_merged_raw_events(paths):
    yield from MergedRawEventReader(paths)


RAW_EVENT_FIELDNAMES = (
    "path",
    "file_event_index",
    "run_number",
    "event_id",
    "event_counter",
    "cpu_timestamp_us",
    "tlu_timestamp",
    "trigger_id",
    "inputs",
)


def _raw_event_row(event):
    return {
        "path": event.path,
        "file_event_index": event.file_event_index,
        "run_number": event.run_number,
        "event_id": event.event_id,
        "event_counter": event.event_counter,
        "cpu_timestamp_us": event.cpu_timestamp_us,
        "tlu_timestamp": event.tlu_timestamp,
        "trigger_id": event.trigger_id,
        "inputs": "".join(str(value) for value in event.inputs),
    }


def _raw_event_from_row(row):
    return RawTriggerEvent(
        path=row["path"],
        file_event_index=int(row["file_event_index"]),
        run_number=int(row["run_number"]),
        event_id=int(row["event_id"]),
        event_counter=int(row["event_counter"]),
        cpu_timestamp_us=int(row["cpu_timestamp_us"]),
        tlu_timestamp=int(row["tlu_timestamp"]),
        trigger_id=int(row["trigger_id"]),
        inputs=tuple(int(value) for value in row["inputs"]),
        fine_timestamps=(),
    )


class TimeSortedRawEventReader:
    """Externally sort raw events before applying order-dependent decisions."""

    def __init__(self, paths, temporary_parent=None):
        self.readers = [RawFileReader(path) for path in paths]
        self.temporary_parent = temporary_parent

    def __iter__(self):
        parent = None
        if self.temporary_parent is not None:
            parent = Path(self.temporary_parent)
            parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".raw_trigger_rate_sort-",
            dir=parent,
        ) as directory:
            directory = Path(directory)
            unsorted_path = directory / "events-unsorted.csv"
            sorted_path = directory / "events-sorted.csv"
            with unsorted_path.open(
                "w", encoding="utf-8", newline=""
            ) as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=RAW_EVENT_FIELDNAMES,
                )
                for reader in self.readers:
                    for event in reader:
                        writer.writerow(_raw_event_row(event))

            environment = os.environ.copy()
            environment["LC_ALL"] = "C"
            with sorted_path.open("wb") as output:
                subprocess.run(
                    [
                        "sort",
                        "-t,",
                        "-k6,6n",
                        "-k5,5n",
                        "-k8,8n",
                        "-k1,1",
                        "-k2,2n",
                        "-T",
                        str(directory),
                        str(unsorted_path),
                    ],
                    stdout=output,
                    check=True,
                    env=environment,
                )

            with sorted_path.open(
                "r", encoding="utf-8", newline=""
            ) as input_file:
                reader = csv.DictReader(
                    input_file,
                    fieldnames=RAW_EVENT_FIELDNAMES,
                )
                for row in reader:
                    yield _raw_event_from_row(row)

    def stats(self):
        return {
            "raw_events_seen": sum(
                reader.stats.events_seen for reader in self.readers
            ),
            "raw_events_with_tlu": sum(
                reader.stats.events_with_tlu for reader in self.readers
            ),
            "raw_events_without_tlu": sum(
                reader.stats.events_without_tlu for reader in self.readers
            ),
            "malformed_raw_events": sum(
                reader.stats.malformed_events for reader in self.readers
            ),
        }


class TriggerDeduplicator:
    """Deduplicate TriggerIDs while preserving reset/rollover segments."""

    def __init__(self, recent_id_limit=100_000, reorder_tolerance=1024):
        if recent_id_limit <= 0:
            raise ValueError("recent_id_limit must be positive")
        if reorder_tolerance < 0:
            raise ValueError("reorder_tolerance must be non-negative")
        self.recent_id_limit = recent_id_limit
        self.reorder_tolerance = reorder_tolerance
        self.stats = TriggerProcessingStats()
        self.segment = 0
        self.last_unique_trigger_id = None
        self.last_cpu_timestamp_us = None
        self.recent_ids = OrderedDict()
        self.segment_max_trigger_id = None
        self.segment_max_event = None
        self.completed_gaps = []
        self.current_gaps = []
        self.last_unique_event = None
        self.eid_cycle_candidate = None
        self.eid_cycle_matching_losses = []

    def process(self, event):
        self.stats.input_triggers += 1
        if self.stats.first_trigger_id is None:
            self.stats.first_trigger_id = event.trigger_id
            self.stats.first_cpu_timestamp_us = event.cpu_timestamp_us
            self.stats.counter_segments = 1
        self.stats.last_trigger_id = event.trigger_id
        self.stats.last_cpu_timestamp_us = event.cpu_timestamp_us

        if self.eid_cycle_candidate is not None:
            candidate = self.eid_cycle_candidate
            excursion = candidate["excursion"]
            previous = candidate["previous"]
            recovered = (
                excursion.trigger_id - event.trigger_id >= EID_CYCLE_JUMP_MIN
                and event.event_counter == excursion.event_counter + 1
                and event.cpu_timestamp_us >= excursion.cpu_timestamp_us
                and abs(event.trigger_id - (previous.trigger_id + 1))
                <= self.reorder_tolerance
            )
            if recovered:
                self.current_gaps = candidate["current_gaps"]
                self.segment_max_trigger_id = candidate["segment_max_trigger_id"]
                self.segment_max_event = candidate["segment_max_event"]
                self.last_unique_trigger_id = previous.trigger_id
                self.last_unique_event = previous
                loss = EIDCycleMatchingLoss(
                    previous_trigger_id=previous.trigger_id,
                    excursion_trigger_id=excursion.trigger_id,
                    recovery_trigger_id=event.trigger_id,
                    previous_event_counter=previous.event_counter,
                    excursion_event_counter=excursion.event_counter,
                    recovery_event_counter=event.event_counter,
                    excursion_cpu_timestamp_us=excursion.cpu_timestamp_us,
                )
                self.eid_cycle_matching_losses.append(loss)
                self.stats.eid_cycle_matching_losses = len(
                    self.eid_cycle_matching_losses
                )
                self._refresh_gap_stats()
            self.eid_cycle_candidate = None

        large_backward_step = (
            self.last_unique_trigger_id is not None
            and event.trigger_id + self.reorder_tolerance
            < self.last_unique_trigger_id
        )
        if large_backward_step:
            rollover = (
                self.last_unique_trigger_id >= 0xFFFF0000
                and event.trigger_id <= 0x0000FFFF
            )
            self.segment += 1
            self.stats.counter_segments += 1
            if rollover:
                self.stats.counter_rollovers += 1
            else:
                self.stats.counter_resets += 1
            self.recent_ids.clear()
            self.completed_gaps.extend(self.current_gaps)
            self.current_gaps = []
            self.segment_max_trigger_id = None
            self.segment_max_event = None

        previous = self.recent_ids.get(event.trigger_id)
        if previous is not None:
            first_cpu_timestamp_us, first_tlu_timestamp = previous
            conflict = (
                first_tlu_timestamp != event.tlu_timestamp
                and first_cpu_timestamp_us != event.cpu_timestamp_us
            )
            self.stats.duplicate_triggers += 1
            if conflict:
                self.stats.duplicate_timestamp_conflicts += 1
            self.recent_ids.move_to_end(event.trigger_id)
            return TriggerDecision(
                event=event,
                segment=self.segment,
                duplicate=True,
                duplicate_reason=(
                    "same TriggerID in segment with different timestamps"
                    if conflict
                    else "same TriggerID in segment"
                ),
            )

        if (
            not large_backward_step
            and self.last_unique_trigger_id is not None
            and event.trigger_id < self.last_unique_trigger_id
        ):
            self.stats.small_out_of_order_ids += 1

        if (
            not large_backward_step
            and self.last_unique_event is not None
            and event.trigger_id - self.last_unique_trigger_id
            >= EID_CYCLE_JUMP_MIN
            and event.event_counter == self.last_unique_event.event_counter + 1
            and event.cpu_timestamp_us >= self.last_unique_event.cpu_timestamp_us
        ):
            self.eid_cycle_candidate = {
                "previous": self.last_unique_event,
                "excursion": event,
                "current_gaps": list(self.current_gaps),
                "segment_max_trigger_id": self.segment_max_trigger_id,
                "segment_max_event": self.segment_max_event,
            }

        self._update_trigger_id_gaps(event)
        self.recent_ids[event.trigger_id] = (
            event.cpu_timestamp_us,
            event.tlu_timestamp,
        )
        if len(self.recent_ids) > self.recent_id_limit:
            self.recent_ids.popitem(last=False)
        self.last_unique_trigger_id = event.trigger_id
        self.last_cpu_timestamp_us = event.cpu_timestamp_us
        self.last_unique_event = event
        self.stats.unique_triggers += 1
        return TriggerDecision(event=event, segment=self.segment, duplicate=False)

    def _update_trigger_id_gaps(self, event):
        if self.segment_max_trigger_id is None:
            self.segment_max_trigger_id = event.trigger_id
            self.segment_max_event = event
            return

        if event.trigger_id > self.segment_max_trigger_id:
            if event.trigger_id > self.segment_max_trigger_id + 1:
                previous = self.segment_max_event
                self.current_gaps.append(
                    TriggerIDGap(
                        segment=self.segment,
                        first_missing_trigger_id=self.segment_max_trigger_id + 1,
                        last_missing_trigger_id=event.trigger_id - 1,
                        previous_trigger_id=self.segment_max_trigger_id,
                        next_trigger_id=event.trigger_id,
                        previous_cpu_timestamp_us=previous.cpu_timestamp_us,
                        next_cpu_timestamp_us=event.cpu_timestamp_us,
                    )
                )
                self._refresh_gap_stats()
            self.segment_max_trigger_id = event.trigger_id
            self.segment_max_event = event
        else:
            if self._remove_late_trigger_from_gaps(event.trigger_id):
                self._refresh_gap_stats()

    def _remove_late_trigger_from_gaps(self, trigger_id):
        updated = []
        removed = False
        for gap in self.current_gaps:
            if not (
                gap.first_missing_trigger_id
                <= trigger_id
                <= gap.last_missing_trigger_id
            ):
                updated.append(gap)
                continue
            removed = True
            if gap.first_missing_trigger_id < trigger_id:
                updated.append(
                    TriggerIDGap(
                        segment=gap.segment,
                        first_missing_trigger_id=gap.first_missing_trigger_id,
                        last_missing_trigger_id=trigger_id - 1,
                        previous_trigger_id=gap.previous_trigger_id,
                        next_trigger_id=gap.next_trigger_id,
                        previous_cpu_timestamp_us=gap.previous_cpu_timestamp_us,
                        next_cpu_timestamp_us=gap.next_cpu_timestamp_us,
                    )
                )
            if trigger_id < gap.last_missing_trigger_id:
                updated.append(
                    TriggerIDGap(
                        segment=gap.segment,
                        first_missing_trigger_id=trigger_id + 1,
                        last_missing_trigger_id=gap.last_missing_trigger_id,
                        previous_trigger_id=gap.previous_trigger_id,
                        next_trigger_id=gap.next_trigger_id,
                        previous_cpu_timestamp_us=gap.previous_cpu_timestamp_us,
                        next_cpu_timestamp_us=gap.next_cpu_timestamp_us,
                    )
                )
        self.current_gaps = updated
        return removed

    def _refresh_gap_stats(self):
        gaps = self.completed_gaps + self.current_gaps
        counts = [gap.missing_count for gap in gaps]
        self.stats.missing_trigger_ids_in_physics_raw = sum(counts)
        self.stats.trigger_id_gap_count = len(gaps)
        self.stats.largest_trigger_id_gap = max(counts, default=0)

    def trigger_id_gap_details(self):
        return [
            gap.as_dict()
            for gap in self.completed_gaps + self.current_gaps
        ]

    def eid_cycle_matching_loss_details(self):
        return [
            loss.as_dict()
            for loss in self.eid_cycle_matching_losses
        ]


def calculate_rate_bins(events, bin_seconds, since=None, until=None):
    accumulator = RateAccumulator(bin_seconds)
    for event in events:
        timestamp = event.unix_time
        if since is not None and timestamp < since:
            continue
        if until is not None and timestamp >= until:
            continue
        accumulator.add(event)
    return accumulator.bins(since=since, until=until)


def write_rate_csv(path, bins):
    """Write the supplied bin fields; recorded inputs have no correction columns."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(asdict(bins[0])) if bins else ("start_utime", "stop_utime", "count", "rate_hz")
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(item) for item in bins)


def write_rate_svg(
    path,
    bins,
    title="AHCAL recorded-trigger rate",
    plot_style="points",
):
    if plot_style not in {"bars", "points", "line"}:
        raise ValueError(
            f"unsupported plot style {plot_style!r}; "
            "choose bars, points, or line"
        )
    width = 1000
    height = 560
    left, right, top, bottom = 85, 30, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    plotted_rates = [item.rate_hz for item in bins]
    max_rate = max(plotted_rates, default=0.0)
    y_max = max(1.0, max_rate * 1.08)
    if bins:
        origin = bins[0].start_utime
        total_time = max(bins[-1].stop_utime - origin, 1.0)
    else:
        origin = 0.0
        total_time = 1.0

    def x_coord(timestamp):
        return left + (timestamp - origin) / total_time * plot_width

    def y_coord(rate):
        return top + plot_height - rate / y_max * plot_height

    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="30" text-anchor="middle" '
            f'font-family="sans-serif" font-size="20">{title}</text>'
        ),
    ]
    for tick in range(6):
        fraction = tick / 5
        y = top + plot_height * (1 - fraction)
        value = y_max * fraction
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" stroke="#dddddd"/>'
        )
        elements.append(
            f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="13">{value:.3g}</text>'
        )
    for tick in range(6):
        fraction = tick / 5
        x = left + plot_width * fraction
        value = total_time * fraction
        elements.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{top+plot_height}" stroke="#eeeeee"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{top+plot_height+24}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13">{value:.3g}</text>'
        )
    points = [
        (
            x_coord(0.5 * (item.start_utime + item.stop_utime)),
            y_coord(item.rate_hz),
        )
        for item in bins
    ]
    if plot_style == "bars":
        for item in bins:
            x1 = x_coord(item.start_utime)
            x2 = x_coord(item.stop_utime)
            y = y_coord(item.rate_hz)
            elements.append(
                f'<rect class="rate-bar" x="{x1:.3f}" y="{y:.3f}" '
                f'width="{max(x2-x1, 0.2):.3f}" '
                f'height="{top+plot_height-y:.3f}" fill="#2878b5"/>'
            )
    else:
        if plot_style == "line" and points:
            coordinates = " ".join(
                f"{x:.3f},{y:.3f}" for x, y in points
            )
            elements.append(
                f'<polyline class="rate-line" points="{coordinates}" '
                'fill="none" stroke="#2878b5" stroke-width="1.5" '
                'stroke-linejoin="round"/>'
            )
        for x, y in points:
            elements.append(
                f'<circle class="rate-point" cx="{x:.3f}" cy="{y:.3f}" '
                'r="2.5" fill="#2878b5"/>'
            )
    elements.extend(
        [
            (
                f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" '
                f'y2="{top+plot_height}" stroke="black"/>'
            ),
            (
                f'<line x1="{left}" y1="{top}" x2="{left}" '
                f'y2="{top+plot_height}" stroke="black"/>'
            ),
            (
                f'<text x="{left+plot_width/2}" y="{height-18}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="15">'
                "Time from first bin (s)</text>"
            ),
            (
                f'<text x="22" y="{top+plot_height/2}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="15" '
                f'transform="rotate(-90 22 {top+plot_height/2})">'
                "Rate (Hz)</text>"
            ),
            "</svg>",
        ]
    )
    Path(path).write_text("\n".join(elements) + "\n", encoding="utf-8")


def _event_csv_writer(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    output = Path(path).open("w", encoding="utf-8", newline="")
    fieldnames = (
        *RAW_EVENT_FIELDNAMES,
        "segment",
        "duplicate",
        "duplicate_reason",
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    return output, writer


def _prefixed_path(prefix, suffix):
    return Path(f"{prefix}{suffix}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read AHCAL DAQ raw files without FAIR/ROOT, remove duplicate "
            "TriggerIDs, and write a recorded-trigger rate histogram."
        )
    )
    parser.add_argument("raw_files", nargs="*", help="input .raw files")
    parser.add_argument("--run", type=int, help="run number used with --raw-pattern")
    parser.add_argument(
        "--raw-pattern",
        action="append",
        default=[],
        help="glob containing {run} or {run06}; repeatable",
    )
    parser.add_argument("--bin-seconds", type=float, default=2.0)
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=(
            "output path prefix for CSV, SVG, and JSON files "
            f"(default: {DEFAULT_OUTPUT_PREFIX})"
        ),
    )
    parser.add_argument(
        "--plot-style",
        choices=("bars", "points", "line"),
        default="points",
        help="SVG plot style (default: bars)",
    )
    parser.add_argument(
        "--events-csv",
        help="optional per-trigger diagnostic CSV, including duplicates",
    )
    parser.add_argument("--recent-id-limit", type=int, default=100_000)
    parser.add_argument("--reorder-tolerance", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run is not None and args.output_prefix == DEFAULT_OUTPUT_PREFIX:
        args.output_prefix = str(RAW_RATE_DIR / f"run{args.run}_trigger_rate")
    if args.bin_seconds <= 0:
        print("error: --bin-seconds must be > 0", file=sys.stderr)
        return 2
    paths = [Path(item) for item in args.raw_files]
    if args.raw_pattern:
        if args.run is None:
            print("error: --raw-pattern requires --run", file=sys.stderr)
            return 2
        paths.extend(discover_raw_files(args.run, args.raw_pattern))
    paths = sorted({path.resolve() for path in paths if path.is_file()})
    if not paths:
        print("error: no raw files found", file=sys.stderr)
        return 2

    deduplicator = TriggerDeduplicator(
        recent_id_limit=args.recent_id_limit,
        reorder_tolerance=args.reorder_tolerance,
    )
    rate_accumulator = VetoRateAccumulator(args.bin_seconds)
    event_output = None
    event_writer = None
    if args.events_csv:
        event_output, event_writer = _event_csv_writer(args.events_csv)
    try:
        sorted_reader = TimeSortedRawEventReader(
            paths,
            temporary_parent=Path(args.output_prefix).parent,
        )
        for event in sorted_reader:
            decision = deduplicator.process(event)
            if event_writer is not None:
                row = _raw_event_row(event)
                row.update(
                    {
                        "segment": decision.segment,
                        "duplicate": int(decision.duplicate),
                        "duplicate_reason": decision.duplicate_reason,
                    }
                )
                event_writer.writerow(row)
            if not decision.duplicate:
                rate_accumulator.add(event)
    except (
        OSError,
        RawFormatError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if event_output is not None:
            event_output.close()

    bins = rate_accumulator.bins()
    prefix = Path(args.output_prefix)
    csv_path = _prefixed_path(prefix, ".csv")
    svg_path = _prefixed_path(prefix, ".svg")
    json_path = _prefixed_path(prefix, ".json")
    write_rate_csv(csv_path, bins)
    write_rate_svg(
        svg_path,
        bins,
        title="Recorded trigger rates",
        plot_style=args.plot_style,
    )
    summary = {
        "raw_files": [str(path) for path in paths],
        "bin_seconds": args.bin_seconds,
        "corrections_applied": False,
        "plot_style": args.plot_style,
        "recent_id_limit": args.recent_id_limit,
        "reorder_tolerance": args.reorder_tolerance,
        **sorted_reader.stats(),
        **asdict(deduplicator.stats),
        "trigger_id_gaps": deduplicator.trigger_id_gap_details(),
        "eid_cycle_matching_loss_details": (
            deduplicator.eid_cycle_matching_loss_details()
        ),
    }
    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {csv_path}, {svg_path}, and {json_path} "
        f"({deduplicator.stats.unique_triggers} unique, "
        f"{deduplicator.stats.duplicate_triggers} duplicate triggers)"
    )
    print(f"unique triggers: {deduplicator.stats.unique_triggers:,}")
    print(
        "Physics raw missing TriggerIDs: "
        f"{deduplicator.stats.missing_trigger_ids_in_physics_raw:,}"
    )
    print(
        f"TriggerID gap ranges: "
        f"{deduplicator.stats.trigger_id_gap_count:,}"
    )
    print(
        f"largest TriggerID gap: "
        f"{deduplicator.stats.largest_trigger_id_gap:,}"
    )
    print(
        "16-bit EID cycle matching losses: "
        f"{deduplicator.stats.eid_cycle_matching_losses:,}"
    )
    return 0


DEFAULT_COUNTS_OUTPUT = OUTPUT_DIR / "deadtime_trigger_rate/stable_luminosity/raw_trigger_counts"
DEFAULT_COUNTS_PATTERN = "/eos/experiment/faser/raw/*/{run06}/FaserAHCAL-Physics-{run06}-*.raw"


def rebuild_counts(run, patterns, output_dir, bin_seconds):
    from .add_stable_luminosity import write_json

    started = time.monotonic()
    paths = discover_raw_files(run, patterns)
    if not paths:
        raise RuntimeError(f"no raw files found for run {run}")
    reader = MergedRawEventReader(paths)
    deduplicator = TriggerDeduplicator()
    accumulator = VetoRateAccumulator(bin_seconds)
    for event in reader:
        if event.run_number != run:
            raise ValueError(f"{event.path}: expected run {run}, got {event.run_number}")
        if not deduplicator.process(event).duplicate:
            accumulator.add(event)
    bins = accumulator.bins()
    if not bins:
        raise RuntimeError(f"run {run}: no recorded triggers")
    output = output_dir / f"run_{run}_trigger_rate.csv"
    temporary = output.with_suffix(".csv.tmp")
    try:
        write_rate_csv(temporary, bins)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "run_number": run,
        "raw_files": [str(path) for path in paths],
        "raw_size_bytes": sum(path.stat().st_size for path in paths),
        "bin_seconds": bin_seconds,
        "corrections_applied": False,
        "ordering": "CPU timestamp heap merge, as in calculate_rate_rows",
        "reader_stats": reader.stats(),
        "trigger_stats": asdict(deduplicator.stats),
        "first_trigger_utime": accumulator.first_time,
        "last_trigger_utime": accumulator.last_time,
        "start_utime": bins[0].start_utime,
        "stop_utime": bins[-1].stop_utime,
        "bins": len(bins),
        "elapsed_seconds": time.monotonic() - started,
        "output": str(output),
        "status": "ok",
    }
    write_json(output.with_name(output.stem + "_metadata.json"), metadata)
    return metadata


def counts_main():
    from .add_stable_luminosity import find_rate_source, write_json

    parser = argparse.ArgumentParser(description="Build uncorrected trigger counts for runs without rate CSVs.")
    parser.add_argument("--run-json", type=Path, default=RUNLIST_JSON)
    parser.add_argument("--run", type=int, action="append", help="explicit run; otherwise process missing CSVs")
    parser.add_argument("--raw-pattern", action="append")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COUNTS_OUTPUT)
    parser.add_argument("--bin-seconds", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.bin_seconds <= 0 or args.workers < 1:
        parser.error("bin-seconds and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directories = [args.output_dir,
                   OUTPUT_DIR / "deadtime_trigger_rate/raw_anyveto_measured_prescale_prepost",
                   OUTPUT_DIR / "deadtime_trigger_rate/raw_anyveto_trigger_rates"]
    runs = args.run
    if runs is None:
        runs = [int(r["run_number"]) for r in json.loads(args.run_json.read_text())
                if find_rate_source(int(r["run_number"]), directories,
                                    OUTPUT_DIR / "deadtime_trigger_rate/batch") is None]
    runs = list(dict.fromkeys(runs))
    results = []
    report = args.output_dir.parent / "raw_count_generation.json"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(rebuild_counts, run, args.raw_pattern or [DEFAULT_COUNTS_PATTERN],
                               args.output_dir, args.bin_seconds): run for run in runs}
        for future in as_completed(pending):
            run = pending[future]
            try:
                result = future.result()
                detail = f"{result['trigger_stats']['unique_triggers']} triggers, {result['bins']} bins"
            except Exception as error:
                result = {"run_number": run, "status": "error", "error": str(error)}
                detail = str(error)
            results.append(result)
            write_json(report, sorted(results, key=lambda r: r["run_number"]))
            print(f"[{len(results)}/{len(runs)}] run {run}: {result['status']}, {detail}", flush=True)
    return int(any(r["status"] != "ok" for r in results))


def calculate_recorded_rate_rows(paths, args, intervals=None):
    """Deduplicate and bin recorded triggers without applying corrections."""
    deduplicator = TriggerDeduplicator(
        recent_id_limit=args.recent_id_limit,
        reorder_tolerance=args.reorder_tolerance,
    )
    accumulator = VetoRateAccumulator(args.bin_seconds)
    selected = [VetoRateAccumulator(args.bin_seconds) for _ in (intervals or [])]
    starts = [start for start, stop in (intervals or [])]
    first_time = last_time = None
    if args.trigger_order == "external-sort":
        reader = TimeSortedRawEventReader(
            paths, temporary_parent=args.output_dir,
        )
    else:
        # Raw events are ordered within each file. Heap-merging the streams
        # avoids writing and externally sorting a run-sized temporary CSV.
        reader = MergedRawEventReader(paths)
    for event in reader:
        if intervals is not None and event.run_number != args.run:
            raise RuntimeError(f"raw file contains run {event.run_number}, expected {args.run}")
        first_time = event.unix_time if first_time is None else min(first_time, event.unix_time)
        last_time = event.unix_time if last_time is None else max(last_time, event.unix_time)
        decision = deduplicator.process(event)
        if not decision.duplicate:
            if intervals is None:
                accumulator.add(event)
            else:
                index = bisect_right(starts, event.unix_time) - 1
                if index >= 0 and event.unix_time < intervals[index][1]:
                    selected[index].add(event)
    if intervals is None:
        return accumulator.bins(), deduplicator.stats
    result = []
    if first_time is not None:
        # Preserve the old raw-bin coverage convention, while clipping exactly
        # to each requested interval. Never create bins beyond the raw coverage.
        coverage_start = math.floor(first_time / args.bin_seconds) * args.bin_seconds
        coverage_stop = (math.floor(last_time / args.bin_seconds) + 1) * args.bin_seconds
        for (start, stop), counter in zip(intervals, selected):
            start, stop = max(start, coverage_start), min(stop, coverage_stop)
            if start >= stop:
                continue
            for item in counter.bins(since=start, until=stop):
                left, right = max(start, item.start_utime), min(stop, item.stop_utime)
                width = right - left
                result.append(VetoRateBin(
                    left, right, item.count, item.count / width,
                    item.any_vetoed_count, item.non_vetoed_count,
                    item.any_vetoed_count / width, item.non_vetoed_count / width,
                ))
    return result, deduplicator.stats
