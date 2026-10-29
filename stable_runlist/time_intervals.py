"""UTC conversions and operations on stable/good time intervals."""

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
    return (
        dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def parse_influx_time(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()

def interval_entry(since, until, reason=None):
    entry = {
        "start_time": format_time(since),
        "stop_time": format_time(until),
        "start_utime": since,
        "stop_utime": until,
    }
    if reason is not None:
        entry["reason"] = reason
    return entry


def merge_intervals(intervals):
    items = sorted(
        (item for item in intervals if item["start_utime"] < item["stop_utime"]),
        key=lambda item: (item["start_utime"], item["stop_utime"]),
    )
    if not items:
        return []

    merged = [dict(items[0])]
    for item in items[1:]:
        last = merged[-1]
        if item["start_utime"] <= last["stop_utime"]:
            if item["stop_utime"] > last["stop_utime"]:
                last["stop_utime"] = item["stop_utime"]
                last["stop_time"] = item["stop_time"]
            if item.get("reason") and item["reason"] not in str(last.get("reason", "")):
                last["reason"] = f"{last.get('reason', 'Excluded')}; {item['reason']}"
        else:
            merged.append(dict(item))
    return merged


def subtract_intervals(
    stable_list,
    excluded_list,
):
    excluded = merge_intervals(excluded_list)
    good = []

    for stable in stable_list:
        pieces = [(stable["start_utime"], stable["stop_utime"])]
        for excluded_interval in excluded:
            esince = excluded_interval["start_utime"]
            euntil = excluded_interval["stop_utime"]
            next_pieces = []
            for since, until in pieces:
                if euntil <= since or esince >= until:
                    next_pieces.append((since, until))
                    continue
                if since < esince:
                    next_pieces.append((since, esince))
                if euntil < until:
                    next_pieces.append((euntil, until))
            pieces = next_pieces

        for since, until in pieces:
            entry = interval_entry(since, until)
            if "lhc_fill" in stable:
                entry["lhc_fill"] = stable["lhc_fill"]
            good.append(entry)

    return good
