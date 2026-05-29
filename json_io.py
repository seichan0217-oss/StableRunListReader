import json
from pathlib import Path


def load_runs(path):
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")

    runs = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {index} in {path} is not a JSON object")
        for key in ("run_number", "start_time", "stop_time"):
            if key not in item:
                raise ValueError(f"Entry {index} in {path} is missing '{key}'")
        runs.append(dict(item))
    return runs


def write_runs(path, runs):
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(runs, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
