import math
import json
import ssl as ssl_module
import urllib.parse
import urllib.request
from urllib.parse import urlunparse

from .time_intervals import interval_entry, merge_intervals
from .secret_config import replace_environment_variables
from .time_intervals import parse_influx_time, to_iso_utc


def influx_interval(seconds):
    if seconds <= 0:
        raise ValueError("InfluxDB bin size must be positive")
    if seconds >= 1 and abs(seconds - round(seconds)) < 1e-9:
        return f"{int(round(seconds))}s"
    milliseconds = int(round(seconds * 1000))
    if milliseconds <= 0:
        raise ValueError("InfluxDB bin size is too small")
    return f"{milliseconds}ms"


class RequestsInfluxResult:
    def __init__(self, payload):
        self.payload = payload

    def get_points(self):
        points = []
        for result in self.payload.get("results", []):
            for series in result.get("series", []):
                columns = series.get("columns", [])
                for values in series.get("values", []):
                    points.append(dict(zip(columns, values)))
        return points


class RequestsInfluxClient:
    def __init__(
        self,
        host,
        port,
        username,
        password,
        database,
        ssl=True,
        verify_ssl=True,
    ):
        self.database = database
        self.auth = (username, password) if username or password else None
        self.verify_ssl = verify_ssl
        scheme = "https" if ssl else "http"
        netloc = f"{host}:{port}"
        self.query_url = urlunparse((scheme, netloc, "/query", "", "", ""))

    def query(self, query):
        params = urllib.parse.urlencode({"db": self.database, "q": query})
        request = urllib.request.Request(f"{self.query_url}?{params}")
        if self.auth is not None:
            import base64

            raw = f"{self.auth[0]}:{self.auth[1]}".encode("utf-8")
            token = base64.b64encode(raw).decode("ascii")
            request.add_header("Authorization", f"Basic {token}")

        context = None
        if not self.verify_ssl:
            context = ssl_module._create_unverified_context()
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            payload = json.loads(response.read().decode("utf-8"))
        for result in payload.get("results", []):
            if "error" in result:
                raise RuntimeError(result["error"])
        return RequestsInfluxResult(payload)


def make_influx_client(host, port, username, password, database, verify_ssl=True):
    try:
        from influxdb import InfluxDBClient
    except ImportError:
        return RequestsInfluxClient(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            ssl=True,
            verify_ssl=verify_ssl,
        )

    return InfluxDBClient(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        ssl=True,
        verify_ssl=verify_ssl,
    )


class PrePostVetoRatioReader:
    """Read the run-final TLU PreVeto/PostVeto hardware counters."""

    def __init__(
        self,
        host,
        port,
        username,
        password,
        database,
        secret_file,
        pre_veto_measurement,
        post_veto_measurement,
        post_run_seconds=10.0,
        verify_ssl=True,
        verbose=False,
        client=None,
    ):
        if post_run_seconds < 0:
            raise ValueError("post_run_seconds must be non-negative")
        self.client = client or RequestsInfluxClient(
            host=host,
            port=port,
            username=replace_environment_variables(username, secret_file),
            password=replace_environment_variables(password, secret_file),
            database=replace_environment_variables(database, secret_file),
            ssl=True,
            verify_ssl=verify_ssl,
        )
        self.pre_veto_measurement = pre_veto_measurement
        self.post_veto_measurement = post_veto_measurement
        self.post_run_seconds = float(post_run_seconds)
        self.verbose = verbose

    def _query_aggregate(self, aggregate, measurement, since, until):
        query = (
            f'SELECT {aggregate}("value") AS v FROM "{measurement}" '
            f"WHERE time >= '{to_iso_utc(since)}' and "
            f"time <= '{to_iso_utc(until)}'"
        )
        if self.verbose:
            print(f"[PrePostVeto] {query}")
        points = list(self.client.query(query).get_points())
        for point in points:
            value = point.get("v")
            if isinstance(value, (int, float)) and math.isfinite(value):
                return value, point.get("time")
        return None, None

    def _query_counter_series(self, measurement, since, until, bin_seconds):
        interval = influx_interval(bin_seconds)
        query = (
            f'SELECT max("value") AS v FROM "{measurement}" '
            f"WHERE time >= '{to_iso_utc(since)}' and "
            f"time <= '{to_iso_utc(until)}' "
            f"GROUP BY time({interval}) fill(previous)"
        )
        if self.verbose:
            print(f"[PrePostVeto] {query}")
        points = list(self.client.query(query).get_points())
        return [
            (parse_influx_time(point["time"]), point.get("v"))
            for point in points
        ]

    @staticmethod
    def _counter_as_int(value, measurement):
        if value is None:
            raise RuntimeError(
                f"no valid InfluxDB counter value found for {measurement}"
            )
        rounded = int(round(value))
        if rounded < 0 or abs(value - rounded) > 1e-6:
            raise RuntimeError(
                f"invalid non-integer counter value {value!r} for {measurement}"
            )
        return rounded

    def _run_final_counter(self, measurement, since, until):
        # The TLU counters are reset at run start. The DAQ stop method reads and
        # publishes their final values, so prefer the first sample just after
        # the recorded stop time, as done by FAIR's PrePostTriggersArchiver.
        value, timestamp = self._query_aggregate(
            "first",
            measurement,
            until,
            until + self.post_run_seconds,
        )
        source = "first sample after run stop"
        if value is None:
            value, timestamp = self._query_aggregate(
                "last",
                measurement,
                since,
                until,
            )
            source = "last sample within run"
        return self._counter_as_int(value, measurement), timestamp, source

    def run_summary(self, since, until):
        if until <= since:
            raise ValueError("run stop time must be after run start time")
        pre, pre_timestamp, pre_source = self._run_final_counter(
            self.pre_veto_measurement,
            since,
            until,
        )
        post, post_timestamp, post_source = self._run_final_counter(
            self.post_veto_measurement,
            since,
            until,
        )
        if post > pre:
            raise RuntimeError(
                "PostVeto counter exceeds PreVeto counter "
                f"({post:,} > {pre:,})"
            )

        post_over_pre = post / pre if pre else None
        pre_over_post = pre / post if post else None
        vetoed = pre - post
        summary = {
            "source": "InfluxDB TLU hardware counters",
            "pre_veto_measurement": self.pre_veto_measurement,
            "post_veto_measurement": self.post_veto_measurement,
            "pre_veto_triggers": pre,
            "post_veto_triggers": post,
            "pre_minus_post_triggers": vetoed,
            "post_over_pre_ratio": post_over_pre,
            "pre_over_post_ratio": pre_over_post,
            "veto_fraction": (vetoed / pre if pre else None),
            "pre_veto_sample_time": pre_timestamp,
            "post_veto_sample_time": post_timestamp,
            "pre_veto_sample_source": pre_source,
            "post_veto_sample_source": post_source,
        }
        if self.verbose:
            ratio_text = (
                f"{post_over_pre:.8f}" if post_over_pre is not None else "undefined"
            )
            print(
                f"[PrePostVeto] Pre={pre:,}, Post={post:,}, "
                f"Post/Pre={ratio_text}"
            )
        return summary

    def ratio_series(self, since, until, bin_seconds):
        """Return binned counter increments and instantaneous veto ratios."""
        if until <= since:
            raise ValueError("run stop time must be after run start time")
        if bin_seconds <= 0:
            raise ValueError("bin_seconds must be positive")

        # Include one bin before the run so the first in-run delta has a
        # baseline, matching FAIR/scripts/Luminosity.py.
        query_since = max(0.0, since - bin_seconds)
        pre_series = self._query_counter_series(
            self.pre_veto_measurement, query_since, until, bin_seconds
        )
        post_series = self._query_counter_series(
            self.post_veto_measurement, query_since, until, bin_seconds
        )
        pre_by_time = dict(pre_series)
        post_by_time = dict(post_series)
        timestamps = sorted(set(pre_by_time) | set(post_by_time))

        rows = []
        previous_pre = None
        previous_post = None
        for timestamp in timestamps:
            pre = pre_by_time.get(timestamp)
            post = post_by_time.get(timestamp)
            valid = (
                isinstance(pre, (int, float))
                and math.isfinite(pre)
                and isinstance(post, (int, float))
                and math.isfinite(post)
            )
            delta_pre = None
            delta_post = None
            if valid and previous_pre is not None and previous_post is not None:
                candidate_pre = pre - previous_pre
                candidate_post = post - previous_post
                if candidate_pre >= 0 and candidate_post >= 0:
                    delta_pre = candidate_pre
                    delta_post = candidate_post
            if valid:
                previous_pre = pre
                previous_post = post

            if timestamp < since or timestamp >= until:
                continue
            acceptance = (
                delta_post / delta_pre
                if delta_pre is not None and delta_pre > 0
                else None
            )
            inverse_correction = (
                delta_pre / delta_post
                if delta_post is not None and delta_post > 0
                else None
            )
            rows.append(
                {
                    "start_utime": timestamp,
                    "stop_utime": min(timestamp + bin_seconds, until),
                    "pre_veto_delta": delta_pre,
                    "post_veto_delta": delta_post,
                    "post_over_pre_ratio": acceptance,
                    "pre_over_post_ratio": inverse_correction,
                    "veto_fraction": (
                        1.0 - acceptance if acceptance is not None else None
                    ),
                }
            )
        return rows


def counter_rate_series(counter_series, bin_seconds):
    rate = []
    previous = None
    for timestamp, value in counter_series:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            rate.append((timestamp, math.nan))
            continue
        if previous is None:
            rate.append((timestamp, math.nan))
            previous = value
            continue
        delta = value - previous
        if delta < 0:
            rate.append((timestamp, math.nan))
        else:
            rate.append((timestamp, delta / float(bin_seconds)))
        previous = value
    return rate


def solve_true_rate_from_reco(r_reco, deadtime_s, auto_triggered_s):
    # Match FAIR/scripts/Luminosity.py:
    # R_reco = R_true(1-exp(-L R_true)) / (1-exp(-L R_true)+D R_true)
    if not math.isfinite(r_reco) or r_reco <= 0.0:
        return 0.0
    if (
        not math.isfinite(deadtime_s)
        or deadtime_s < 0.0
        or not math.isfinite(auto_triggered_s)
        or auto_triggered_s <= 0.0
    ):
        return r_reco

    def reco_from_true(r_true):
        one_minus_exp = 1.0 - math.exp(-auto_triggered_s * r_true)
        denom = one_minus_exp + deadtime_s * r_true
        if denom <= 0.0:
            return 0.0
        return (r_true * one_minus_exp) / denom

    lo = 0.0
    hi = max(1.0, r_reco * 2.0 + 1.0)
    for _ in range(100):
        if reco_from_true(hi) >= r_reco:
            break
        hi *= 2.0

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if reco_from_true(mid) < r_reco:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


class DAQConditionReader:
    def __init__(
        self,
        host,
        port,
        username,
        password,
        database,
        secret_file,
        bin_seconds,
        max_gap_seconds,
        required_measurements,
        verify_ssl=True,
        verbose=False,
    ):
        self.client = make_influx_client(
            host=host,
            port=port,
            username=replace_environment_variables(username, secret_file),
            password=replace_environment_variables(password, secret_file),
            database=replace_environment_variables(database, secret_file),
            verify_ssl=verify_ssl,
        )
        self.bin_seconds = bin_seconds
        self.max_gap_seconds = max_gap_seconds
        self.required_measurements = required_measurements
        self.verbose = verbose

    def query_counter(
        self,
        measurement,
        since,
        until,
    ):
        interval = influx_interval(self.bin_seconds)
        query = (
            f'SELECT max("value") AS v FROM "{measurement}" '
            f"WHERE time >= '{to_iso_utc(since)}' and time <= '{to_iso_utc(until)}' "
            f"GROUP BY time({interval}) fill(previous)"
        )
        if self.verbose:
            print(f"[DAQ] {query}")
        result = self.client.query(query)
        points = list(result.get_points())
        return [(parse_influx_time(point["time"]), point.get("v")) for point in points]

    def live_fraction_series(
        self,
        since,
        until,
        any_vetoed_measurement,
        non_vetoed_measurement,
        deadtime_s,
        auto_triggered_s,
    ):
        query_since = max(0.0, since - self.bin_seconds)
        any_rates = counter_rate_series(
            self.query_counter(any_vetoed_measurement, query_since, until),
            self.bin_seconds,
        )
        non_rates = counter_rate_series(
            self.query_counter(non_vetoed_measurement, query_since, until),
            self.bin_seconds,
        )

        by_timestamp = {}
        for timestamp, rate in any_rates:
            by_timestamp.setdefault(timestamp, {})["any"] = rate
        for timestamp, rate in non_rates:
            by_timestamp.setdefault(timestamp, {})["non"] = rate

        live_fraction = []
        for timestamp in sorted(by_timestamp):
            if timestamp < since or timestamp >= until:
                continue
            rates = by_timestamp[timestamp]
            any_rate = rates.get("any", math.nan)
            non_rate = rates.get("non", math.nan)
            if (
                not isinstance(any_rate, (int, float))
                or not math.isfinite(any_rate)
                or not isinstance(non_rate, (int, float))
                or not math.isfinite(non_rate)
            ):
                live_fraction.append((timestamp, math.nan))
                continue

            r_reco = any_rate + non_rate
            r_true = solve_true_rate_from_reco(
                r_reco,
                deadtime_s,
                auto_triggered_s,
            )
            if r_true <= 0.0:
                live_fraction.append((timestamp, math.nan))
            else:
                live_fraction.append((timestamp, r_reco / r_true))
        return live_fraction

    def excluded_intervals(
        self,
        stable_list,
    ):
        excluded = []
        for stable in stable_list:
            since = stable["start_utime"]
            until = stable["stop_utime"]
            for measurement in self.required_measurements:
                series = self.query_counter(measurement, since, until)
                excluded.extend(
                    self._missing_or_bad_counter_intervals(
                        measurement,
                        series,
                        since,
                        until,
                    )
                )
        return merge_intervals(excluded)

    def _missing_or_bad_counter_intervals(
        self,
        measurement,
        series,
        since,
        until,
    ):
        valid = [
            (timestamp, value)
            for timestamp, value in series
            if isinstance(value, (int, float)) and math.isfinite(value)
        ]
        if not valid:
            return [interval_entry(since, until, f"Missing DAQ counter {measurement}")]

        excluded = []
        first_time = valid[0][0]
        last_time = valid[-1][0]

        if first_time - since > self.max_gap_seconds:
            excluded.append(
                interval_entry(since, first_time, f"Missing DAQ counter {measurement}")
            )

        previous_time = valid[0][0]
        for timestamp, value in valid[1:]:
            gap = timestamp - previous_time
            if gap > self.max_gap_seconds:
                excluded.append(
                    interval_entry(
                        previous_time,
                        timestamp,
                        f"DAQ counter gap {measurement}",
                    )
                )
            previous_time = timestamp

        if until - last_time > self.max_gap_seconds:
            excluded.append(
                interval_entry(last_time, until, f"Missing DAQ counter {measurement}")
            )

        return excluded
