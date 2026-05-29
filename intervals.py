from time_utils import format_time


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
