"""Calculate the AnyVetoed prescale correction from a RunInfo setting."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import urllib.request


RUNINFO_URL = "https://faser-runinfo.app.cern.ch/cgibin/getRunInfo.py"

@dataclass(frozen=True)
class AnyVetoPrescale:
    run_number: int
    configuration: str
    calib_rate: float
    veto_stretch: float
    pass_fraction: float
    correction_factor: float


def prescale_pass_fraction(veto_stretch, calib_rate):
    """Return (2 * stretch + 3) / (CalibRate + 1), or one when disabled."""
    veto_stretch = float(veto_stretch)
    calib_rate = float(calib_rate)
    if not math.isfinite(veto_stretch) or veto_stretch <= 0.0:
        raise ValueError("veto trigger stretch must be finite and positive")
    if not math.isfinite(calib_rate) or calib_rate < 0.0:
        raise ValueError("FASER_CalibRate must be finite and non-negative")
    if calib_rate == 0.0:
        return 1.0
    pass_fraction = (2.0 * veto_stretch + 3.0) / (calib_rate + 1.0)
    if not 0.0 < pass_fraction <= 1.0:
        raise ValueError(
            "calculated AnyVetoed pass fraction is outside (0, 1]: "
            f"{pass_fraction:g}"
        )
    return pass_fraction


def _tlu_settings(run_info):
    configuration = run_info.get("configuration") or {}
    for component in configuration.get("components", []):
        for module in component.get("modules", []):
            if module.get("type") == "TLUReceiver":
                return module.get("settings") or {}
    raise ValueError("RunInfo configuration has no TLUReceiver settings")


def prescale_from_run_info(run_info, run_number=None):
    """Calculate the input-5/6 pass fraction from the RunInfo TLU settings."""
    settings = _tlu_settings(run_info)
    stretches = settings.get("trigger_stretch")
    if not isinstance(stretches, list) or len(stretches) < 6:
        raise ValueError("TLUReceiver trigger_stretch has fewer than six inputs")
    veto_stretches = tuple(float(stretches[index]) for index in (4, 5))
    if any(not math.isfinite(value) or value <= 0.0 for value in veto_stretches):
        raise ValueError("veto trigger stretches must be finite and positive")
    if not math.isclose(veto_stretches[0], veto_stretches[1]):
        raise ValueError(
            "input 5 and 6 have different trigger stretches; one AnyVetoed "
            "prescale factor is not defined"
        )

    calib_rate = float(settings.get("FASER_CalibRate", 0.0))
    veto_stretch = veto_stretches[0]
    pass_fraction = prescale_pass_fraction(veto_stretch, calib_rate)
    correction_factor = 1.0 / pass_fraction

    resolved_run = int(
        run_number
        if run_number is not None
        else run_info.get("run_number", run_info.get("runno", -1))
    )
    return AnyVetoPrescale(
        run_number=resolved_run,
        configuration=str(run_info.get("configName") or ""),
        calib_rate=calib_rate,
        veto_stretch=veto_stretch,
        pass_fraction=pass_fraction,
        correction_factor=correction_factor,
    )


@lru_cache(maxsize=None)
def fetch_any_veto_prescale(run_number):
    """Fetch one run's setting and calculate its prescale fraction."""
    run_number = int(run_number)
    url = f"{RUNINFO_URL}?runno={run_number}"
    with urllib.request.urlopen(url, timeout=20) as response:
        run_info = json.loads(response.read().decode("utf-8"))
    if not run_info:
        raise RuntimeError(f"run {run_number} was not found in RunInfo")
    if run_info.get("type") != "AHCAL":
        raise RuntimeError(
            f"run {run_number} is type {run_info.get('type', 'unknown')}, "
            "not AHCAL"
        )
    return prescale_from_run_info(run_info, run_number=run_number)


def fetch_any_veto_prescales(run_numbers, max_workers=8):
    """Fetch several run settings concurrently, preserving input order."""
    ordered = list(dict.fromkeys(int(run) for run in run_numbers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        values = list(executor.map(fetch_any_veto_prescale, ordered))
    return {value.run_number: value for value in values}
