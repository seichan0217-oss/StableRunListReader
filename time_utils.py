import datetime as dt


TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def parse_time(value):
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    else:
        parsed = parsed.astimezone(dt.timezone.utc)
    return parsed.timestamp()


def format_time(timestamp):
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).strftime(TIME_FORMAT)


def to_iso_utc(timestamp):
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def parse_influx_time(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
