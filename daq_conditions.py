import math
import json
import ssl as ssl_module
import urllib.parse
import urllib.request
from urllib.parse import urlunparse

from intervals import interval_entry, merge_intervals
from secrets import replace_environment_variables
from time_utils import parse_influx_time, to_iso_utc


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
