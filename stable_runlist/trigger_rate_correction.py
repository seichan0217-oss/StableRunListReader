"""D1-D3 correction and count statistics, without a prescale correction."""

import numpy as np


def solve_true_rates(recorded, deadtimes_s, live_window_s):
    """Vector inversion of the D3-repeated model; invalid inputs give NaN."""
    deadtimes = np.asarray(deadtimes_s, dtype=float)
    if deadtimes.shape != (3,) or not np.all(np.isfinite(deadtimes)) or np.any(deadtimes < 0):
        raise ValueError("three finite, non-negative deadtimes are required")
    if not np.isfinite(live_window_s) or live_window_s <= 0:
        raise ValueError("live window must be finite and positive")
    target = np.asarray(recorded, dtype=float)
    shape = target.shape
    target = target.reshape(-1)
    result = np.full_like(target, np.nan)
    result[target == 0] = 0.0
    valid = np.isfinite(target) & (target > 0)
    goal = target[valid]
    lo = np.zeros_like(goal)
    hi = np.maximum(1.0, goal * 2 + 1)

    def reco(rate):
        q = np.exp(-live_window_s * rate)
        p = -np.expm1(-live_window_s * rate)
        dead = deadtimes[0] + q * deadtimes[1] + q * q / p * deadtimes[2]
        return rate / (1 + rate * dead)

    for _ in range(100):
        expand = reco(hi) < goal
        if not np.any(expand):
            break
        hi[expand] *= 2
    bracketed = reco(hi) >= goal
    for _ in range(70):
        mid = (lo + hi) / 2
        lower = reco(mid) < goal
        lo = np.where(lower, mid, lo)
        hi = np.where(lower, hi, mid)
    result[valid] = np.where(bracketed, (lo + hi) / 2, np.nan)
    return result.reshape(shape)


def correction_scenarios(recorded, stat_error, pre_over_post, model):
    """Vary r +/- sqrt(N)/T or all D_i +/- run SD, keeping other inputs fixed.

    Pre/Post uncertainty is neglected. Live fractions use r/g(c*r), retaining
    the shared numerator/denominator fluctuation. At r=0 use the model limit.
    """
    r = np.asarray(recorded, dtype=float)
    error = np.broadcast_to(np.asarray(stat_error, dtype=float), r.shape)
    if not np.isfinite(pre_over_post) or pre_over_post < 1:
        raise ValueError("PreVeto/PostVeto must be finite and >= 1")
    dead = np.asarray(model["deadtime_mean_ms"], dtype=float) / 1000
    sigma = np.asarray(model["deadtime_std_ms"], dtype=float) / 1000
    if sigma.shape != (3,) or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError("three finite, non-negative deadtime standard deviations are required")
    window = float(model["live_window_ms"]) / 1000
    inputs = {
        "central": (r, dead),
        "stat_low": (np.maximum(0, r - error), dead),
        "stat_high": (r + error, dead),
        "deadtime_low": (r, np.maximum(0, dead - sigma)),
        "deadtime_high": (r, dead + sigma),
    }
    result = {}
    for key, (rate, times) in inputs.items():
        true = solve_true_rates(rate * pre_over_post, times, window)
        live = np.divide(rate, true, out=np.full_like(rate, np.nan), where=true > 0)
        live = np.where(rate == 0, 1 / (pre_over_post * (1 + times[2] / window)), live)
        result[key] = {"true_total": true, "live_fraction": live}
    return result


def poisson_rate_summary(bins, intervals, field="total_rate"):
    """Time-weighted raw rate and Poisson error from independent input bins.

    A partially selected bin estimates its full-bin rate. Its uncertainty is
    weighted by the selected duration; a bin shared by selections is counted
    as one fluctuating input, not as independent fragments.
    """
    exposure = weighted_rate = variance = 0.0
    for item in bins:
        start, stop = item["start"], item["stop"]
        duration = stop - start
        rate = item.get(field, np.nan)
        if duration <= 0 or not np.isfinite(rate) or rate < 0:
            continue
        overlap = sum(max(0.0, min(stop, b) - max(start, a)) for a, b in intervals)
        exposure += overlap
        weighted_rate += overlap * rate
        variance += overlap * overlap * rate / duration
    if exposure <= 0:
        return np.nan, np.nan, 0.0
    return weighted_rate / exposure, np.sqrt(variance) / exposure, exposure
