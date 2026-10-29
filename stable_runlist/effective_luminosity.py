import math

import numpy as np

from .trigger_rate_correction import correction_scenarios
from .time_intervals import format_time


class StableLuminosityCalculator:
    """Integrate LB luminosity and raw-rate live fractions over stable_list.

    Counts in disjoint raw bins are approximated as independent Poisson
    variables. A raw bin crossing LB/selection boundaries remains one input.
    D1-D3 variations are common to every bin; Pre/Post is held fixed.
    """

    def __init__(self, lumi_reader, model):
        self.lumi_reader = lumi_reader
        self.model = model

    def calculate(self, stable_list, rate_bins, pre_over_post):
        intervals = sorted((float(x["start_utime"]), float(x["stop_utime"]))
                           for x in stable_list)
        if any(a >= b for a, b in intervals) or any(
            b > c for (_, b), (c, _) in zip(intervals, intervals[1:])
        ):
            raise ValueError("stable intervals must be positive and disjoint")
        stable_time = sum(b - a for a, b in intervals)
        bins = sorted(rate_bins, key=lambda x: x["start"])
        if any(x["stop"] <= x["start"] for x in bins) or any(
            a["stop"] > b["start"] for a, b in zip(bins, bins[1:])
        ):
            raise ValueError("raw rate bins must be positive and disjoint")
        r = np.asarray([x["rate"] for x in bins], dtype=float)
        errors = np.asarray([x["stat_error"] for x in bins], dtype=float)
        usable = np.isfinite(r) & (r >= 0) & np.isfinite(errors) & (errors >= 0)
        if bins and pre_over_post is not None:
            scenarios = correction_scenarios(r, errors, pre_over_post, self.model)
            live = {key: item["live_fraction"] for key, item in scenarios.items()}
            usable &= np.logical_and.reduce([np.isfinite(x) for x in live.values()])
        else:
            usable[:] = False
            live = {key: np.zeros(len(bins)) for key in
                    ("central", "stat_low", "stat_high", "deadtime_low", "deadtime_high")}
        starts = np.asarray([x["start"] for x in bins])
        stops = np.asarray([x["stop"] for x in bins])
        central = live["central"]
        stat_minus = np.maximum(0, central - np.minimum(live["stat_low"], live["stat_high"]))
        stat_plus = np.maximum(0, np.maximum(live["stat_low"], live["stat_high"]) - central)

        # Group clipped pieces of the same ATLAS LB before taking its mean.
        grouped = {}
        lumi_intervals = []
        skipped = []
        for block in self.lumi_reader.luminosity_blocks(stable_list):
            if int(block["atlas_run"]) <= 0 or int(block["lumi_block"]) <= 0:
                skipped.append(dict(block, reason="no valid ATLAS run/LB identifier"))
                continue
            ell = float(block["luminosity_ub_inv_s"])
            if not math.isfinite(ell) or ell < 0:
                continue
            pieces = [(max(a, float(block["start_utime"])),
                       min(b, float(block["stop_utime"]))) for a, b in intervals]
            pieces = [(a, b) for a, b in pieces if a < b]
            if not pieces:
                continue
            key = (int(block["atlas_run"]), int(block["lumi_block"]))
            entry = grouped.setdefault(key, {"source": block, "pieces": [], "ell": ell})
            if not math.isclose(entry["ell"], ell, rel_tol=1e-9):
                raise ValueError("inconsistent luminosity for the same LB")
            entry["pieces"].extend(pieces)
            lumi_intervals.extend(pieces)
        ordered = sorted(lumi_intervals)
        if any(b > c + 1e-6 for (_, b), (c, _) in zip(ordered, ordered[1:])):
            raise ValueError("overlapping luminosity records would double-count time")

        blocks = []
        total_coefficients = np.zeros(len(bins))
        lumi_time = covered_time = luminosity = 0.0
        for key, entry in sorted(grouped.items(), key=lambda item: min(item[1]["pieces"])):
            pieces, ell = sorted(entry["pieces"]), entry["ell"]
            duration = sum(b - a for a, b in pieces)
            overlaps = np.zeros(len(bins))
            for a, b in pieces:
                overlaps += np.maximum(0, np.minimum(stops, b) - np.maximum(starts, a))
            overlaps = np.where(usable, overlaps, 0)
            covered = float(overlaps.sum())
            coefficients = overlaps * ell
            total_coefficients += coefficients
            lumi_time += duration
            covered_time += covered
            luminosity += ell * duration
            complete = math.isclose(covered, duration, abs_tol=1e-5, rel_tol=0)
            effective = float(np.sum(coefficients * np.where(usable, central, 0)))
            block = dict(entry["source"])
            block.update(
                start_utime=pieces[0][0], stop_utime=pieces[-1][1],
                start_time=format_time(pieces[0][0]), stop_time=format_time(pieces[-1][1]),
                stable_intervals=[{"start_utime": a, "stop_utime": b} for a, b in pieces],
                stable_time_sec=duration, luminosity_ub_inv=ell * duration,
                daq_covered_time_sec=covered, daq_missing_time_sec=max(0, duration-covered),
                daq_live_fraction=(effective / (ell * covered) if ell > 0 and covered > 0 else
                                   float(np.sum(overlaps * np.where(usable, central, 0))) / covered
                                   if covered > 0 else None),
                effective_luminosity_ub_inv=effective if complete else None,
                effective_luminosity_covered_ub_inv=effective,
            )
            block.update(self._uncertainties(coefficients, live, usable, stat_minus, stat_plus,
                                             complete))
            blocks.append(block)

        lumi_missing = max(0.0, stable_time - lumi_time)
        missing = max(0.0, stable_time - covered_time)
        complete = math.isclose(missing, 0, abs_tol=1e-5)
        lumi_complete = math.isclose(lumi_missing, 0, abs_tol=1e-5)
        effective = float(np.sum(total_coefficients * np.where(usable, central, 0)))
        result = {
            "luminosity_blocks": blocks,
            "luminosity_skipped_records": skipped,
            "luminosity_ub_inv": luminosity if lumi_complete else None,
            "luminosity_pb_inv": luminosity / 1e6 if lumi_complete else None,
            "luminosity_covered_ub_inv": luminosity,
            "luminosity_covered_time_sec": lumi_time,
            "luminosity_missing_time_sec": lumi_missing,
            "effective_luminosity_ub_inv": effective if complete else None,
            "effective_luminosity_pb_inv": effective / 1e6 if complete else None,
            "effective_luminosity_covered_ub_inv": effective,
            "effective_luminosity_covered_pb_inv": effective / 1e6,
            "effective_luminosity_covered_time_sec": covered_time,
            "effective_luminosity_missing_time_sec": missing,
            "daq_lumi_live_fraction": effective / luminosity if complete and luminosity > 0 else None,
            "luminosity_calculation_status": "complete" if complete else "incomplete",
        }
        result.update(self._uncertainties(total_coefficients, live, usable,
                                         stat_minus, stat_plus, complete))
        return result

    @staticmethod
    def _uncertainties(coefficients, live, usable, stat_minus, stat_plus, complete):
        def integral(values):
            return float(np.sum(coefficients * np.where(usable, values, 0)))
        central = integral(live["central"])
        low, high = integral(live["deadtime_low"]), integral(live["deadtime_high"])
        values = {
            "stat_minus": float(np.sqrt(np.sum(np.square(coefficients * np.where(usable, stat_minus, 0))))),
            "stat_plus": float(np.sqrt(np.sum(np.square(coefficients * np.where(usable, stat_plus, 0))))),
            "deadtime_sys_minus": max(0, central - min(low, high)),
            "deadtime_sys_plus": max(0, max(low, high) - central),
        }
        for side in ("minus", "plus"):
            values[f"total_error_{side}"] = math.hypot(
                values[f"stat_{side}"], values[f"deadtime_sys_{side}"])
        output = {}
        for name, value in values.items():
            output[f"effective_luminosity_{name}_ub_inv"] = value if complete else None
            output[f"effective_luminosity_covered_{name}_ub_inv"] = value
        for name, value in (("deadtime_low", low), ("deadtime_high", high)):
            output[f"effective_luminosity_at_{name}_ub_inv"] = value if complete else None
            output[f"effective_luminosity_covered_at_{name}_ub_inv"] = value
        return output


class EffectiveLuminosityCalculator:
    def __init__(
        self,
        lumi_reader,
        daq_reader,
        any_vetoed_measurement,
        non_vetoed_measurement,
        deadtime_ms,
        auto_triggered_ms,
        verbose=False,
    ):
        self.lumi_reader = lumi_reader
        self.daq_reader = daq_reader
        self.any_vetoed_measurement = any_vetoed_measurement
        self.non_vetoed_measurement = non_vetoed_measurement
        self.deadtime_s = deadtime_ms / 1000.0
        self.auto_triggered_s = auto_triggered_ms / 1000.0
        self.verbose = verbose

    def calculate(self, good_list):
        blocks = []
        for good in good_list:
            since = good["start_utime"]
            until = good["stop_utime"]
            lumi_blocks = self.lumi_reader.luminosity_blocks([good])
            live_series = self.daq_reader.live_fraction_series(
                since,
                until,
                self.any_vetoed_measurement,
                self.non_vetoed_measurement,
                self.deadtime_s,
                self.auto_triggered_s,
            )

            if self.verbose:
                print(
                    f"Effective lumi {since:.3f}-{until:.3f}: "
                    f"{len(lumi_blocks)} lumi blocks, {len(live_series)} DAQ bins"
                )

            for block in lumi_blocks:
                enriched = dict(block)
                live_fraction = self._average_live_fraction(
                    live_series,
                    block["start_utime"],
                    block["stop_utime"],
                )
                enriched["daq_live_fraction"] = live_fraction
                if live_fraction is None:
                    enriched["effective_luminosity_ub_inv"] = None
                else:
                    enriched["effective_luminosity_ub_inv"] = (
                        block["luminosity_ub_inv"] * live_fraction
                    )
                blocks.append(enriched)

        luminosity = sum(
            block["luminosity_ub_inv"]
            for block in blocks
            if self._is_number(block.get("luminosity_ub_inv"))
        )
        effective_luminosity = sum(
            block["effective_luminosity_ub_inv"]
            for block in blocks
            if self._is_number(block.get("effective_luminosity_ub_inv"))
        )
        missing_time = sum(
            block["stop_utime"] - block["start_utime"]
            for block in blocks
            if block.get("daq_live_fraction") is None
        )
        live_fraction = None
        if luminosity > 0.0:
            live_fraction = effective_luminosity / luminosity

        return {
            "luminosity_blocks": blocks,
            "luminosity_ub_inv": luminosity,
            "luminosity_pb_inv": luminosity / 1.0e6,
            "effective_luminosity_ub_inv": effective_luminosity,
            "effective_luminosity_pb_inv": effective_luminosity / 1.0e6,
            "daq_lumi_live_fraction": live_fraction,
            "effective_luminosity_missing_time_sec": missing_time,
        }

    @staticmethod
    def _average_live_fraction(live_series, since, until):
        values = [
            value
            for timestamp, value in live_series
            if since <= timestamp < until
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0.0
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _is_number(value):
        return isinstance(value, (int, float)) and math.isfinite(value)
