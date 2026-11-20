"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target time.

Model (v2, tuned on synthetic launch-week backtests in dev/backtest.py):

For every trophy, work with the logit of its earn rate (percent) against
x = ln(1 + hours since release). Only the last 18 hours of reads are fitted:
older reads describe a launch regime (a smaller, keener player base, a page
that had barely begun to refresh) that no longer drives the rates. A
weighted least-squares line is fitted per trophy -- weights grow with the
players tracked (less counting noise) and with recency -- and its slope is
shrunk toward a pooled slope (its tier's, blended with the global pool) by
an information weight, so a trophy whose own reads are few, flat or noisy
borrows the field's trajectory instead of extrapolating its own jitter.

The forecast median is anchored at the trophy's recent level (a weighted
average of the logit rates of its latest reads) and carried to the target
along the shrunk slope, with the extrapolated move damped by 0.85: a line
fitted on 18 hours is an optimistic guide to the next 108, and pinball loss
punishes an overshot median at both ends.

The spread combines a floor (a refresh of the page can move every rate
between two reads), a random-walk term growing with the square root of the
horizon in days, a share of the trophy's residual scatter around its fit,
and the slope's standard error projected over the extrapolation distance.
Quantiles are the median shifted by z_tau * sigma on the logit scale.

Run by the grader as `python3 -P -s /bot/main.py` (Python 3.13, standard
library only, no network); locally set MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR.
Deterministic: no randomness anywhere.
"""

import json
import math
import os
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RATE_LO = 0.01
RATE_HI = 99.99

WINDOW_HOURS = 18.0       # fit only the recent regime
DAMP = 0.85               # shrink the extrapolated move toward the anchor
SHRINK_PRIOR_N = 2.0      # slope shrinkage strength toward the pooled slope

# Spread ingredients (logit units).
SIGMA_FLOOR = 0.30        # a page refresh can jump every rate at once
SIGMA_RW = 0.40           # random-walk scale per sqrt(horizon day)
RESID_WEIGHT = 0.4        # share of a trophy's residual scatter to carry
SLOPE_ERR_WEIGHT = 1.0    # slope standard error projected to the target


def clamp_rate(pct):
    return min(RATE_HI, max(RATE_LO, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    v = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(v) if float(v).is_integer() else v


def parse_time(text):
    from datetime import datetime
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


class TrophyFit:
    __slots__ = ("slope", "se_slope", "resid_sd", "anchor_logit", "anchor_x",
                 "n_points", "info")

    def __init__(self):
        self.slope = 0.0
        self.se_slope = 0.0
        self.resid_sd = 0.0
        self.anchor_logit = 0.0
        self.anchor_x = 0.0
        self.n_points = 0
        self.info = 0.0


def fit_trophy(xs, ys, ws):
    """Weighted least squares of y on x, plus an anchored recent level."""
    fit = TrophyFit()
    n = len(xs)
    fit.n_points = n
    if n == 0:
        return fit
    sw = sum(ws)
    if sw <= 0:
        ws = [1.0] * n
        sw = float(n)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    if sxx > 1e-12:
        fit.slope = sxy / sxx
        intercept = my - fit.slope * mx
        resid = [y - (intercept + fit.slope * x) for x, y in zip(xs, ys)]
        dof = max(1.0, float(n) - 2.0)
        var = sum(w * r * r for r, w in zip(resid, ws)) / dof * (sw / n)
        fit.resid_sd = math.sqrt(max(var, 0.0))
        fit.se_slope = math.sqrt(max(var, 0.0) / sxx)
    else:
        resid = [y - my for y in ys]
        fit.resid_sd = math.sqrt(sum(w * r * r for r, w in zip(resid, ws)) / max(1.0, float(n)))
    # Information behind the slope: x-span covered, discounted by scatter.
    span = (max(xs) - min(xs)) if n > 1 else 0.0
    fit.info = span / (0.35 + fit.resid_sd)
    # Anchor: weighted mean of the logit rates over the last stretch of
    # fitted reads (the final quarter, at least the last three).
    k = max(3, (n + 3) // 4)
    tail_x = xs[-k:]
    tail_y = ys[-k:]
    tail_w = ws[-k:]
    tw = sum(tail_w)
    fit.anchor_logit = sum(w * y for y, w in zip(tail_y, tail_w)) / tw
    fit.anchor_x = sum(w * x for x, w in zip(tail_x, tail_w)) / tw
    return fit


def pooled_slope(fits, idxs):
    """Information-weighted mean slope over a group of trophy fits."""
    num = 0.0
    den = 0.0
    for i in idxs:
        f = fits[i]
        if f.n_points >= 2:
            num += f.slope * f.info
            den += f.info
    return num / den if den > 0 else 0.0


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels")
    history = case.get("history") or []
    if not history:
        raise ValueError("no history")
    trophies = case["trophies"]

    game = case.get("game") or {}
    if game.get("release_date"):
        release = parse_time(game["release_date"] + "T00:00:00Z")
    else:
        release = parse_time(history[0]["at"])
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))
    horizon_days = case["horizon_minutes"] / 1440.0

    times = [parse_time(r["at"]) for r in history]
    xs_all = [math.log1p(max(0.0, (t - release).total_seconds() / 3600.0)) for t in times]
    players = [max(1, int(r.get("players_tracked") or 1)) for r in history]
    p_last = players[-1]

    window_start = checkpoint - timedelta(hours=WINDOW_HOURS)
    keep = [j for j in range(len(history)) if times[j] >= window_start]
    if len(keep) < 3:
        keep = list(range(max(0, len(history) - 8), len(history)))

    fits = []
    for i in range(len(trophies)):
        xs, ys, ws = [], [], []
        n_reads = len(history)
        for j in keep:
            rate = history[j]["rates"][i]
            if rate is None:
                continue
            # Weight: more players tracked -> less counting noise; later
            # reads describe the regime we extrapolate from.
            precision = math.sqrt(min(players[j], 20 * p_last) / p_last)
            recency = 0.35 + 0.65 * (j / max(1, n_reads - 1))
            xs.append(xs_all[j])
            ys.append(logit(rate))
            ws.append(precision * recency)
        fits.append(fit_trophy(xs, ys, ws))

    tiers = {}
    for i, t in enumerate(trophies):
        tiers.setdefault(t.get("tier", "?"), []).append(i)
    global_slope = pooled_slope(fits, range(len(trophies)))
    tier_slope = {name: pooled_slope(fits, idxs) for name, idxs in tiers.items()}

    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        last_rate = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                last_rate = read["rates"][i]
                break
        if f.n_points == 0:
            base = logit(last_rate if last_rate is not None else 0.5)
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(base + z * 1.2)) for z in Z]})
            continue
        # Shrink the slope: own fit -> tier pool -> global pool.
        lam = f.info / (f.info + SHRINK_PRIOR_N)
        pool = 0.65 * tier_slope.get(trophy.get("tier", "?"), global_slope) \
            + 0.35 * global_slope
        slope = lam * f.slope + (1.0 - lam) * pool
        dx = x_target - f.anchor_x
        median = f.anchor_logit + DAMP * slope * dx
        median = min(logit(RATE_HI), max(logit(RATE_LO), median))
        sigma = math.sqrt(
            SIGMA_FLOOR ** 2
            + (SIGMA_RW * math.sqrt(max(horizon_days, 0.0))) ** 2
            + (RESID_WEIGHT * f.resid_sd) ** 2
            + (SLOPE_ERR_WEIGHT * f.se_slope * abs(dx)) ** 2
        )
        qs = [round4(inv_logit(median + z * sigma)) for z in Z]
        qs = [min(100.0, max(0.0, q)) for q in qs]
        qs.sort()
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
