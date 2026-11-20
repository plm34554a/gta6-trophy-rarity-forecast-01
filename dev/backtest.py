"""Synthetic backtests for the trophy forecast model.

No real future data exists locally (the practice histories are one constant
page), so this builds plausible launch-week worlds -- player growth, per-
trophy adoption curves, Exophase-style page refreshes that freeze the rates
between refreshes, count-quantised printing -- cuts a case at the final's
checkpoint (2026-11-21T11:30Z) with target 2026-11-26T00:00Z, and scores
each forecaster with the contest's own rule: mean pinball loss on the logit
scale. Variants: the two house baselines, main.py's model, and ablations of
it (module constants monkeypatched).
"""

import importlib.util
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)
CHECKPOINT = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
TARGET = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
TIERS = ["platinum"] + ["gold"] * 2 + ["silver"] * 6 + ["bronze"] * 46

spec = importlib.util.spec_from_file_location("entry", HERE.parent / "main.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def clamp(p):
    return min(99.99, max(0.01, p))


def logit(p):
    p = clamp(p) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg)) if lg >= 0 else 100.0 * math.exp(lg) / (1 + math.exp(lg))


def pinball(case_rows, truth):
    total, n = 0.0, 0
    for row in case_rows:
        y = truth[row["id"]]
        ly = logit(y)
        for tau, q in zip(LEVELS, row["q"]):
            d = ly - logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return total / n


def house_persistence(case):
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i] or 0.0
        lg = logit(r)
        rows.append({"id": t["id"], "q": [inv_logit(lg + z * spread) for z in Z]})
    return {"trophies": rows}


def house_curve(case):
    """Logit-linear in ln(1+hours since release), last 24h of reads, OLS."""
    rows = []
    hist = case["history"]
    target = entry.parse_time(case["target"])
    x_t = math.log1p((target - RELEASE).total_seconds() / 3600.0)
    cutoff = entry.parse_time(case["at"]) - timedelta(hours=24)
    recent = [r for r in hist if entry.parse_time(r["at"]) >= cutoff] or hist[-3:]
    spread_rw = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    for i, t in enumerate(case["trophies"]):
        pts = []
        for r in recent:
            if r["rates"][i] is not None:
                x = math.log1p((entry.parse_time(r["at"]) - RELEASE).total_seconds() / 3600.0)
                pts.append((x, logit(r["rates"][i])))
        if len(pts) < 3:
            lg = logit(hist[-1]["rates"][i] or 0.0)
            rows.append({"id": t["id"], "q": [inv_logit(lg + z * spread_rw) for z in Z]})
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        b = sxy / sxx if sxx > 0 else 0.0
        a = my - b * mx
        resid = math.sqrt(sum((p[1] - (a + b * p[0])) ** 2 for p in pts) / max(1, n - 2))
        lg = a + b * x_t
        spread = math.sqrt(spread_rw ** 2 + resid ** 2)
        rows.append({"id": t["id"], "q": [inv_logit(lg + z * spread) for z in Z]})
    return {"trophies": rows}


def make_world(rng, family, n_max, t50, tau_n, dense_tail):
    """Return (case, truth) with truth = {trophy_id: printed rate at target}."""
    # Per-trophy saturation and adoption-curve placement.
    trophies = []
    sat, cen, wid = [], [], []
    for i, tier in enumerate(TIERS):
        trophies.append({"id": i + 1, "name": f"T{i+1}", "description": "",
                         "tier": tier, "points": 15, "secret": False})
        if i == 1:
            sat.append(100.0)  # the always-earned trophy
        elif tier == "platinum":
            sat.append(rng.uniform(0.15, 0.8))
        elif tier == "gold":
            sat.append(rng.uniform(2.0, 9.0))
        elif tier == "silver":
            sat.append(rng.uniform(4.0, 18.0))
        else:
            sat.append(math.exp(rng.uniform(math.log(0.8), math.log(48.0))))
        cen.append(rng.uniform(3.1, 5.3))
        wid.append(rng.uniform(0.35, 0.95))

    def players(t_h):
        base = n_max / (1.0 + math.exp(-(t_h - t50) / tau_n))
        return max(60.0, base * (638.0 / (n_max / (1.0 + math.exp(t50 / tau_n)))))

    def engaged(t_h):
        return n_max / (1.0 + math.exp(-(t_h - (t50 + rng_shift)) / (tau_n * 1.25)))

    rng_shift = rng.uniform(15.0, 45.0)

    def true_rate(i, t_h):
        if sat[i] >= 100.0:
            return 100.0
        x = math.log1p(max(0.0, t_h))
        adopt = 1.0 / (1.0 + math.exp(-(x - cen[i]) / wid[i]))
        if family == "curve":
            return sat[i] * adopt
        # dilution family: earners scale with the slower engaged population
        dil = engaged(t_h) / max(players(t_h), 1.0)
        dil0 = engaged(168.0) / max(players(168.0), 1.0)
        return min(99.0, sat[i] * adopt * dil / dil0)

    # Page refreshes: every 3-9 hours the printed page catches up.
    refresh_times = []
    t = -1.0
    while t < 175.0:
        t += rng.uniform(3.0, 9.0)
        refresh_times.append(t)

    def page_state(t_h):
        rt = max([r for r in refresh_times if r <= t_h], default=refresh_times[0])
        n = int(round(players(rt)))
        rates = []
        for i in range(55):
            earners = int(round(true_rate(i, rt) / 100.0 * n))
            rates.append(math.floor(10000.0 * earners / n + 0.5) / 100.0)
        return n, rates

    # Reads: 30 min from -10 min; optionally 10 min after 48h.
    reads = []
    t_h = -1.0 / 6.0
    while True:
        when = RELEASE + timedelta(hours=t_h)
        if when >= CHECKPOINT:
            break
        n, rates = page_state(t_h)
        reads.append({"slot": when.isoformat().replace("+00:00", "Z"),
                      "at": when.isoformat().replace("+00:00", "Z"),
                      "players_tracked": n,
                      "trophies_earned": int(sum(r / 100.0 * n for r in rates)),
                      "rates": rates})
        t_h += (1.0 / 6.0) if (dense_tail and t_h >= 48.0) else 0.5

    horizon = int((TARGET - CHECKPOINT).total_seconds() // 60)
    case = {"schema": "makefun/gta6-trophies-case/v1", "id": "f00",
            "at": CHECKPOINT.isoformat().replace("+00:00", "Z"),
            "horizon_minutes": horizon,
            "target": TARGET.isoformat().replace("+00:00", "Z"),
            "levels": LEVELS,
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19", "source": "synthetic"},
            "trophies": trophies, "history": reads, "notes": "synthetic"}
    _, ans_rates = page_state(168.0)
    truth = {i + 1: ans_rates[i] for i in range(55)}
    return case, truth


def run_variant(name, fn, worlds):
    losses = [pinball(fn(case)["trophies"], truth) for case, truth in worlds]
    return sum(losses) / len(losses), losses


def main():
    worlds = []
    seed = 1000
    for family in ("curve", "dilution"):
        for n_max, t50, tau_n in ((40000, 30, 18), (100000, 60, 30), (250000, 100, 26), (80000, 45, 22)):
            for dense in (False, True):
                rng = random.Random(seed)
                seed += 1
                worlds.append(make_world(rng, family, n_max, t50, tau_n, dense))
    print(f"{len(worlds)} synthetic worlds")
    variants = [
        ("house persistence", house_persistence),
        ("house curve", house_curve),
        ("mine v1 (defaults)", lambda c: entry.forecast_case(c)),
    ]
    results = {}
    for name, fn in variants:
        mean, losses = run_variant(name, fn, worlds)
        results[name] = mean
        print(f"{name:28s} mean pinball {mean:.4f}  worst {max(losses):.4f}")
    return results


if __name__ == "__main__":
    main()


# ---- Parameterised clone of main.py's model, for structure search --------

def my_model(case, window_h=None, anchor=True, shrink_prior=6.0,
             s_floor=0.45, s_rw=0.45, resid_w=0.8, slope_err_w=1.0,
             tail_frac=0.25):
    from datetime import timedelta as _td
    history = case["history"]
    trophies = case["trophies"]
    target = entry.parse_time(case["target"])
    at = entry.parse_time(case["at"])
    x_target = math.log1p(max(0.0, (target - RELEASE).total_seconds() / 3600.0))
    horizon_days = case["horizon_minutes"] / 1440.0
    times = [entry.parse_time(r["at"]) for r in history]
    xs_all = [math.log1p(max(0.0, (t - RELEASE).total_seconds() / 3600.0)) for t in times]
    players = [max(1, int(r.get("players_tracked") or 1)) for r in history]
    p_last = players[-1]
    keep = [j for j in range(len(history))
            if window_h is None or times[j] >= at - _td(hours=window_h)]
    fits = []
    for i in range(len(trophies)):
        xs, ys, ws = [], [], []
        for j in keep:
            rate = history[j]["rates"][i]
            if rate is None:
                continue
            precision = math.sqrt(min(players[j], 20 * p_last) / p_last)
            recency = 0.35 + 0.65 * (keep.index(j) / max(1, len(keep) - 1)) if False else 0.35 + 0.65 * (j / max(1, len(history) - 1))
            xs.append(xs_all[j]); ys.append(logit(rate)); ws.append(precision * recency)
        fits.append(entry.fit_trophy(xs, ys, ws))
    tiers = {}
    for i, t in enumerate(trophies):
        tiers.setdefault(t.get("tier", "?"), []).append(i)
    gs = entry.pooled_slope(fits, range(len(trophies)))
    ts = {k: entry.pooled_slope(fits, v) for k, v in tiers.items()}
    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        lam = f.info / (f.info + shrink_prior)
        pool = 0.65 * ts.get(trophy.get("tier", "?"), gs) + 0.35 * gs
        slope = lam * f.slope + (1 - lam) * pool
        if anchor:
            dx = x_target - f.anchor_x
            median = f.anchor_logit + slope * dx
        else:
            # pure line: reconstruct intercept from anchor point on the
            # trophy's own fitted line through (anchor_x, anchor_logit)
            dx = x_target - f.anchor_x
            median = f.anchor_logit + slope * dx
        median = min(logit(99.99), max(logit(0.01), median))
        sigma = math.sqrt(s_floor ** 2 + (s_rw * math.sqrt(max(horizon_days, 0.0))) ** 2
                          + (resid_w * f.resid_sd) ** 2 + (slope_err_w * f.se_slope * abs(dx)) ** 2)
        rows.append({"id": trophy["id"], "q": sorted(inv_logit(median + z * sigma) for z in Z)})
    return {"trophies": rows}


def grid():
    worlds = []
    seed = 1000
    for family in ("curve", "dilution"):
        for n_max, t50, tau_n in ((40000, 30, 18), (100000, 60, 30), (250000, 100, 26), (80000, 45, 22)):
            for dense in (False, True):
                rng = random.Random(seed); seed += 1
                worlds.append(make_world(rng, family, n_max, t50, tau_n, dense))
    configs = [
        ("clone defaults", dict()),
        ("shrink 2", dict(shrink_prior=2.0)),
        ("shrink 15", dict(shrink_prior=15.0)),
        ("shrink 0 (own slope)", dict(shrink_prior=0.0)),
        ("window 24h", dict(window_h=24)),
        ("window 48h", dict(window_h=48)),
        ("window 24h shrink 2", dict(window_h=24, shrink_prior=2.0)),
        ("sigma floor .25 rw .35", dict(s_floor=0.25, s_rw=0.35)),
        ("sigma floor .6 rw .55", dict(s_floor=0.6, s_rw=0.55)),
        ("resid_w 0", dict(resid_w=0.0)),
        ("resid_w 1.5", dict(resid_w=1.5)),
        ("slope_err 0", dict(slope_err_w=0.0)),
        ("slope_err 2", dict(slope_err_w=2.0)),
    ]
    for name, kw in configs:
        mean, losses = run_variant(name, lambda c, kw=kw: my_model(c, **kw), worlds)
        print(f"{name:26s} mean {mean:.4f}  worst {max(losses):.4f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "grid":
    grid()


def grid2():
    def build(seed0, combos):
        worlds = []
        seed = seed0
        for family in ("curve", "dilution"):
            for n_max, t50, tau_n in combos:
                for dense in (False, True):
                    rng = random.Random(seed); seed += 1
                    worlds.append(make_world(rng, family, n_max, t50, tau_n, dense))
        return worlds
    train = build(1000, ((40000, 30, 18), (100000, 60, 30), (250000, 100, 26), (80000, 45, 22)))
    fresh = build(5000, ((60000, 40, 20), (150000, 80, 34), (30000, 25, 15), (180000, 55, 24)))
    configs = [
        ("w24 s2 base", dict(window_h=24, shrink_prior=2.0)),
        ("w24 s2 sig .25/.35", dict(window_h=24, shrink_prior=2.0, s_floor=0.25, s_rw=0.35)),
        ("w24 s2 sig .35/.45", dict(window_h=24, shrink_prior=2.0, s_floor=0.35, s_rw=0.45)),
        ("w24 s2 sig .45/.45 resid0", dict(window_h=24, shrink_prior=2.0, resid_w=0.0)),
        ("w36 s2", dict(window_h=36, shrink_prior=2.0)),
        ("w18 s2", dict(window_h=18, shrink_prior=2.0)),
        ("w24 s4", dict(window_h=24, shrink_prior=4.0)),
        ("w24 s1", dict(window_h=24, shrink_prior=1.0)),
        ("w24 s2 sig .3/.4 resid.4", dict(window_h=24, shrink_prior=2.0, s_floor=0.3, s_rw=0.4, resid_w=0.4)),
    ]
    for name, kw in configs:
        m_tr, _ = run_variant(name, lambda c, kw=kw: my_model(c, **kw), train)
        m_fr, l_fr = run_variant(name, lambda c, kw=kw: my_model(c, **kw), fresh)
        print(f"{name:28s} train {m_tr:.4f}  fresh {m_fr:.4f}  fresh-worst {max(l_fr):.4f}")
    for name, fn in (("house curve", house_curve), ("house persistence", house_persistence)):
        m_tr, _ = run_variant(name, fn, train)
        m_fr, _ = run_variant(name, fn, fresh)
        print(f"{name:28s} train {m_tr:.4f}  fresh {m_fr:.4f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "grid2":
    grid2()


def my_model2(case, window_h=18, shrink_prior=2.0, s_floor=0.35, s_rw=0.45,
              resid_w=0.4, slope_err_w=1.0, damp=1.0, x_mode="time"):
    """Variant with extrapolation damping and optional ln(players) regressor."""
    from datetime import timedelta as _td
    history = case["history"]
    trophies = case["trophies"]
    target = entry.parse_time(case["target"])
    at = entry.parse_time(case["at"])
    horizon_days = case["horizon_minutes"] / 1440.0
    times = [entry.parse_time(r["at"]) for r in history]
    players = [max(1, int(r.get("players_tracked") or 1)) for r in history]
    p_last = players[-1]

    def x_of(j, when=None, n=None):
        if x_mode == "players":
            return math.log(n if n is not None else players[j])
        t = when if when is not None else times[j]
        return math.log1p(max(0.0, (t - RELEASE).total_seconds() / 3600.0))

    if x_mode == "players":
        # extrapolate log-players linearly in ln(1+hours) from the window
        xs_t = [math.log1p(max(0.0, (t - RELEASE).total_seconds() / 3600.0)) for t in times]
        ys_n = [math.log(p) for p in players]
        keepn = [j for j in range(len(history)) if times[j] >= at - _td(hours=window_h)]
        fn = entry.fit_trophy([xs_t[j] for j in keepn], [ys_n[j] for j in keepn],
                              [1.0] * len(keepn))
        x_t_h = math.log1p(max(0.0, (target - RELEASE).total_seconds() / 3600.0))
        log_n_target = fn.anchor_logit + fn.slope * (x_t_h - fn.anchor_x)
        x_target = log_n_target
    else:
        x_target = x_of(0, when=target)
    keep = [j for j in range(len(history)) if times[j] >= at - _td(hours=window_h)]
    fits = []
    for i in range(len(trophies)):
        xs, ys, ws = [], [], []
        for j in keep:
            rate = history[j]["rates"][i]
            if rate is None:
                continue
            precision = math.sqrt(min(players[j], 20 * p_last) / p_last)
            recency = 0.35 + 0.65 * (j / max(1, len(history) - 1))
            xs.append(x_of(j)); ys.append(logit(rate)); ws.append(precision * recency)
        fits.append(entry.fit_trophy(xs, ys, ws))
    tiers = {}
    for i, t in enumerate(trophies):
        tiers.setdefault(t.get("tier", "?"), []).append(i)
    gs = entry.pooled_slope(fits, range(len(trophies)))
    ts = {k: entry.pooled_slope(fits, v) for k, v in tiers.items()}
    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        lam = f.info / (f.info + shrink_prior)
        pool = 0.65 * ts.get(trophy.get("tier", "?"), gs) + 0.35 * gs
        slope = lam * f.slope + (1 - lam) * pool
        dx = x_target - f.anchor_x
        median = f.anchor_logit + damp * slope * dx
        median = min(logit(99.99), max(logit(0.01), median))
        sigma = math.sqrt(s_floor ** 2 + (s_rw * math.sqrt(max(horizon_days, 0.0))) ** 2
                          + (resid_w * f.resid_sd) ** 2 + (slope_err_w * f.se_slope * abs(dx)) ** 2)
        rows.append({"id": trophy["id"], "q": sorted(inv_logit(median + z * sigma) for z in Z)})
    return {"trophies": rows}


def grid3():
    def build(seed0, combos):
        worlds, seed = [], seed0
        for family in ("curve", "dilution"):
            for n_max, t50, tau_n in combos:
                for dense in (False, True):
                    rng = random.Random(seed); seed += 1
                    worlds.append(make_world(rng, family, n_max, t50, tau_n, dense))
        return worlds
    train = build(1000, ((40000, 30, 18), (100000, 60, 30), (250000, 100, 26), (80000, 45, 22)))
    fresh = build(5000, ((60000, 40, 20), (150000, 80, 34), (30000, 25, 15), (180000, 55, 24)))
    configs = [
        ("w18 base", dict()),
        ("w18 sig .3/.4", dict(s_floor=0.3, s_rw=0.4)),
        ("w18 sig .4/.5", dict(s_floor=0.4, s_rw=0.5)),
        ("w18 damp .85", dict(damp=0.85)),
        ("w18 damp .7", dict(damp=0.7)),
        ("w18 players-x", dict(x_mode="players")),
        ("w24 players-x", dict(x_mode="players", window_h=24)),
        ("w18 damp.85 sig.3/.4", dict(damp=0.85, s_floor=0.3, s_rw=0.4)),
    ]
    for name, kw in configs:
        m_tr, l_tr = run_variant(name, lambda c, kw=kw: my_model2(c, **kw), train)
        m_fr, l_fr = run_variant(name, lambda c, kw=kw: my_model2(c, **kw), fresh)
        print(f"{name:24s} train {m_tr:.4f} (worst {max(l_tr):.3f})  fresh {m_fr:.4f} (worst {max(l_fr):.3f})")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "grid3":
    grid3()


def grid4():
    """Shock worlds: curve family, but a random subset of trophies gets a
    late adoption surge after the checkpoint and player growth stalls."""
    import copy
    worlds = []
    seed = 9000
    for n_max, t50, tau_n in ((100000, 60, 30), (60000, 40, 20), (180000, 55, 24), (40000, 30, 18)):
        for dense in (False, True):
            rng = random.Random(seed); seed += 1
            case, truth = make_world(rng, "curve", n_max, t50, tau_n, dense)
            # Surge: after the checkpoint, 12 random trophies' true rates
            # rise toward 1.6-2.6x their no-shock path (in odds terms).
            shocked = rng.sample(range(55), 12)
            for idx in shocked:
                y = truth[idx + 1]
                if 0.05 < y < 60:
                    lg = logit(y) + rng.uniform(0.45, 0.95)
                    truth[idx + 1] = inv_logit(lg)
            worlds.append((case, truth))
    for name, fn in (("house curve", house_curve),
                     ("house persistence", house_persistence),
                     ("chosen w18 d.85 s.3/.4", lambda c: my_model2(c, damp=0.85, s_floor=0.3, s_rw=0.4)),
                     ("chosen but wider", lambda c: my_model2(c, damp=0.85, s_floor=0.45, s_rw=0.5))):
        m, losses = run_variant(name, fn, worlds)
        print(f"{name:26s} shock mean {m:.4f}  worst {max(losses):.4f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "grid4":
    grid4()
