"""Audit step 3 for research/outside_lines.py: stress the men's opening-spread residual cell (63 of 63, +4.4%).

Reads only data/processed/outside-lines/matched.json and the raw ESPN scoreboard cache (for scheduled start times).
Re-implements the original's split_test / keep-above rule from its source so the SAME search can be re-run on
shuffled and simulated data. No network. Output: data/processed/audit/outside_lines/stress.txt (+ stress.json).

    .venv/bin/python research/audit/outside_lines_stress.py [robust|perm|plant|mech|all]
"""
import calendar
import glob
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATCHED = os.path.join(ROOT, "data/processed/outside-lines/matched.json")
RAW = os.path.join(ROOT, "data/raw/outside-lines")
OUT = os.path.join(ROOT, "data/processed/audit/outside_lines")
MEAS = ("spread", "ml", "open", "bpi")
LINES = []
RES = {}


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fee(a):
    return 7 * (a / 100) * (1 - a / 100)


def halves(rows):
    rows = sorted(rows, key=lambda r: (r["start"], r["event"]))
    return rows[:len(rows) // 2], rows[len(rows) // 2:]


def load(sigma=None, deg=2, fix_open_sign=False):
    """Rows as the original builds them: x_ = outside - ask, r_ = residual on a polynomial in mid fitted on the first
    half of the league's rows that have the measure. sigma=None keeps the original p_ values."""
    rows = json.load(open(MATCHED))
    sched = {}
    for lg in ("men", "women"):
        for f in glob.glob(os.path.join(RAW, lg, "scoreboard", "*.json")):
            for g in json.load(open(f)):
                sched[(lg, g["id"])] = calendar.timegm(time.strptime(g["date"], "%Y-%m-%dT%H:%MZ"))
    for i, r in enumerate(rows):
        r["i"] = i
        r["espn_start"] = sched.get((r["league"], r["espn"]))
        r["archived"] = r["day"] < "2026-07"
        # quote window ends 25 min before the estimated start (last season) or 2 min before the scheduled start (Nov 2026)
        r["quote_end_vs_sched"] = (r["start"] - (1500 if r["archived"] else 120) - r["espn_start"]) / 60
        if fix_open_sign and r["open"] is not None and r["open"] < 0:
            r["open"] = -r["open"]
            r["p_open"] = None
        if sigma is not None or r.get("p_open") is None:
            s = sigma[r["league"]] if isinstance(sigma, dict) else sigma
            if s is None:  # recover the original sigma from p_spread
                s = {"men": 11.239, "women": 11.129}[r["league"]]
            r["p_spread"] = None if r["close"] is None else 100 * phi(r["close"] / s)
            r["p_open"] = None if r["open"] is None else 100 * phi(r["open"] / s)
        r["move"] = None if r["open"] is None or r["close"] is None else r["close"] - r["open"]
        for k in MEAS:
            r["x_" + k] = None if r["p_" + k] is None else r["p_" + k] - r["ask"]
    for lg in ("men", "women"):
        for k in MEAS:
            part = [r for r in rows if r["league"] == lg and r["p_" + k] is not None]
            first, _ = halves(part)
            fit = np.polyfit([r["mid"] for r in first], [r["p_" + k] for r in first], deg) if len(first) >= 30 else None
            for r in part:
                r["r_" + k] = None if fit is None else float(r["p_" + k] - np.polyval(fit, r["mid"]))
            for r in rows:
                r.setdefault("r_" + k, None)
    return rows


TESTS = (("3a", lambda r: 90 <= r["mid"] < 99, "ask"), ("3b", lambda r: r["order"] and r["fill"], "dip"),
         ("3c", lambda r: 80 <= r["mid"] < 90, "ask"))


def net_paid(r, cost):
    if cost == "ask":
        return 100 * r["payout"] - r["ask"] - fee(r["ask"]), r["ask"] + fee(r["ask"])
    return 100 * r["payout"] - 80, 80.0


def stat(rows, cost="ask", rng=None, draws=4000):
    """n, wins, pooled return %, bootstrap 95% interval, and the return at the exact (Clopper-Pearson) 97.5% lower win rate."""
    n = len(rows)
    if n == 0:
        return dict(n=0, wins=0, ret=float("nan"), lo=float("nan"), hi=float("nan"), cp=float("nan"), ask=float("nan"))
    np_ = np.array([net_paid(r, cost) for r in rows])
    nets, paid = np_[:, 0], float(np_[:, 1].mean())
    wins = int(sum(r["payout"] for r in rows))
    rng = rng or np.random.default_rng(1)
    if n >= 5:
        b = nets[rng.integers(0, n, (draws, n))].mean(1)
        lo, hi = np.percentile(b, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return dict(n=n, wins=wins, ret=100 * nets.mean() / paid, lo=100 * lo / paid, hi=100 * hi / paid,
                cp=100 * (100 * cp_low(wins, n) - paid) / paid, ask=float(np.mean([r["ask"] for r in rows])))


def cp_low(k, n, alpha=0.025):
    """Exact lower confidence bound for a binomial proportion."""
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        # P(X >= k | mid)
        p = 1 - binom_cdf(k - 1, n, mid)
        if p < alpha:
            lo = mid
        else:
            hi = mid
    return lo


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    q = 0.0
    for i in range(0, k + 1):
        q += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(max(p, 1e-300)) + (n - i) * math.log(max(1 - p, 1e-300)))
    return min(1.0, q)


def fmt(c):
    if not c["n"]:
        return "n   0"
    return (f"n {c['n']:4d} wins {c['wins']:4d} ask {c['ask']:5.1f} ret {c['ret']:+6.2f}% boot[{c['lo']:+6.2f},{c['hi']:+6.2f}] "
            f"ret at exact 97.5% lower win rate {c['cp']:+6.2f}%")


def pick_rule(sig1, net1, min_keep=25):
    """The original's keep-above search: thresholds at the 10th..80th percentile of the first half, best mean net."""
    best = None
    for q in range(10, 81, 5):
        t = float(np.percentile(sig1, q))
        k = sig1 >= t
        if k.sum() >= min_keep:
            m = float(net1[k].mean())
            if best is None or m > best[0]:
                best = (m, t, q)
    return best


def members(rows):
    """Every split_test the original ran: 3 tests x (men, women, both) x 4 measures x (raw, residual)."""
    out = []
    for code, keep, cost in TESTS:
        base = [r for r in rows if keep(r)]
        for lg in ("men", "women", "both"):
            part = [r for r in base if lg == "both" or r["league"] == lg]
            for k in MEAS:
                for field in (("p_" if cost == "dip" else "x_") + k, "r_" + k):
                    rs = [r for r in part if r.get(field) is not None]
                    if len(rs) < 40:
                        continue
                    first, second = halves(rs)
                    d = dict(name=f"{code} {lg} {field}", code=code, lg=lg, field=field, cost=cost)
                    for h, hr in (("1", first), ("2", second)):
                        d["rows" + h] = hr
                        d["sig" + h] = np.array([r[field] for r in hr])
                        npd = np.array([net_paid(r, cost) for r in hr])
                        d["net" + h], d["paid" + h] = npd[:, 0], npd[:, 1]
                        d["idx" + h] = np.array([r["i"] for r in hr])
                        d["isw" + h] = np.array([r["league"] == "women" for r in hr])
                    out.append(d)
    return out


def score(sig1, net1, sig2, net2, paid2):
    """Apply the original rule; return dict of second-half numbers (or None)."""
    b = pick_rule(sig1, net1)
    if b is None:
        return None
    _, t, q = b
    k1, k2 = sig1 >= t, sig2 >= t
    n2 = int(k2.sum())
    if n2 == 0 or n2 == len(sig2):
        return dict(t=t, q=q, n1=int(k1.sum()), w1=int((net1[k1] > 0).sum()), n2=n2, w2=int((net2[k2] > 0).sum()) if n2 else 0,
                    ret2=100 * net2[k2].mean() / paid2[k2].mean() if n2 else float("nan"), diff=float("nan"), z=float("nan"))
    a, d = net2[k2], net2[~k2]
    diff = a.mean() - d.mean()
    se = math.sqrt(a.var() / len(a) + d.var() / len(d)) if len(d) > 1 else float("nan")
    # floor the variance so a 100%-win cell does not get an infinite z from price variation alone
    return dict(t=t, q=q, n1=int(k1.sum()), w1=int((net1[k1] > 0).sum()), n2=n2, w2=int((a > 0).sum()),
                ret2=100 * a.mean() / paid2[k2].mean(), diff=diff, z=diff / se if se and se > 0 else float("nan"))


# ------------------------------------------------------------------------------------------------ a) robustness

def robust():
    rng = np.random.default_rng(11)
    rows = load()
    men3a = [r for r in rows if r["league"] == "men" and 90 <= r["mid"] < 99 and r["r_open"] is not None]
    first, second = halves(men3a)
    say("=" * 118)
    say("A. ROBUSTNESS of the men's opening-spread residual cell (3a: 90-99c favourites at the ask, taker fee paid)")
    say(f"   men 3a games with an opening spread: {len(men3a)}; first half {first[0]['day']}..{first[-1]['day']} (n {len(first)}), second half {second[0]['day']}..{second[-1]['day']} (n {len(second)})")
    b = pick_rule(np.array([r["r_open"] for r in first]), np.array([net_paid(r, 'ask')[0] for r in first]))
    t0 = b[1]
    say(f"   reproduction of the original rule: r_open >= {t0:+.3f} (first-half {b[2]}th percentile)")
    say(f"     first half kept    {fmt(stat([r for r in first if r['r_open'] >= t0], rng=rng))}")
    say(f"     second half kept   {fmt(stat([r for r in second if r['r_open'] >= t0], rng=rng))}")
    say(f"     second half dropped{fmt(stat([r for r in second if r['r_open'] < t0], rng=rng))}")
    say(f"     second half all    {fmt(stat(second, rng=rng))}")
    RES["reproduced"] = dict(threshold=t0, first=stat([r for r in first if r['r_open'] >= t0]), second=stat([r for r in second if r['r_open'] >= t0]))
    say("   NOTE on intervals: with 0 losses the bootstrap interval only reflects price variation. The exact bound is the honest one:")
    say(f"     63 of 63 -> 97.5% lower bound on the win rate {100*cp_low(63,63):.1f}%; break-even win rate at ask 95.5c + fee is about 95.8%.")
    say(f"     116 of 116 (both halves; the first half is in-sample) -> lower bound {100*cp_low(116,116):.1f}%.")

    say("\n   A1. nearby thresholds (cents of residual), second half / first half / both")
    tab = []
    for t in (1.0, 1.5, 2.0, 2.5, t0, 3.0, 3.5, 4.0, 5.0):
        s2, s1, sa = (stat([r for r in h if r["r_open"] >= t], rng=rng) for h in (second, first, men3a))
        say(f"     >= {t:+.2f}  H2 {fmt(s2)}")
        say(f"               H1 {fmt(s1)}")
        say(f"               all{fmt(sa)}")
        tab.append(dict(t=t, h2=s2, h1=s1, all=sa))
    RES["thresholds"] = tab
    say("   bands (not cumulative), all men 3a games: where do the losses sit?")
    edges = [-99, -4, -2, -1, 0, 1, 2, 2.7, 4, 99]
    for lo, hi in zip(edges[:-1], edges[1:]):
        say(f"     r_open in [{lo:+5.1f},{hi:+5.1f})  H1 {fmt(stat([r for r in first if lo <= r['r_open'] < hi], rng=rng))}")
        say(f"                             H2 {fmt(stat([r for r in second if lo <= r['r_open'] < hi], rng=rng))}")

    say("\n   A2. functional form of the residual fit (threshold re-picked on the first half by the same rule; and fixed at the 80th percentile)")
    for deg, lab in ((1, "linear"), (2, "quadratic (original)"), (3, "cubic")):
        rr = load(deg=deg)
        _variant(rr, lab, rng, t0)
    say("   opening spreads with the wrong sign in ESPN's feed (5 games, favourite shown as the underdog at the open) flipped back:")
    _variant(load(fix_open_sign=True), "sign-fixed, quadratic", rng, t0)

    say("\n   A3. a different sigma for spread -> probability (original: men 11.24)")
    for s in (9.0, 10.0, 11.24, 12.0, 13.0, 14.0):
        _variant(load(sigma=s), f"sigma {s:.2f}", rng, t0)
    say("   no sigma at all: residual of the opening spread IN POINTS on a quadratic in the Kalshi mid (same rule search)")
    rr = load()
    for lg in ("men", "women"):
        part = [r for r in rr if r["league"] == lg and r["open"] is not None]
        f1, _ = halves(part)
        fit = np.polyfit([r["mid"] for r in f1], [r["open"] for r in f1], 2)
        for r in part:
            r["r_open"] = float(r["open"] - np.polyval(fit, r["mid"]))
    _variant(rr, "points residual", rng, None)

    say("\n   A4. the second half in chronological thirds, and by season (rule fixed at the original threshold)")
    rows = load()
    men3a = [r for r in rows if r["league"] == "men" and 90 <= r["mid"] < 99 and r["r_open"] is not None]
    first, second = halves(men3a)
    k = len(second) // 3
    for i, part in enumerate((second[:k], second[k:2 * k], second[2 * k:])):
        say(f"     third {i+1} {part[0]['day']}..{part[-1]['day']}: kept    {fmt(stat([r for r in part if r['r_open'] >= t0], rng=rng))}")
        say(f"                                     dropped {fmt(stat([r for r in part if r['r_open'] < t0], rng=rng))}")
    for lab, test in (("2025-26 inside the fit/pick half", lambda r: r["archived"] and r in first), ("2025-26 after the pick half", lambda r: r["archived"] and r not in first),
                      ("November 2026", lambda r: not r["archived"])):
        part = [r for r in men3a if test(r)]
        say(f"     {lab:34s} kept    {fmt(stat([r for r in part if r['r_open'] >= t0], rng=rng))}")
        say(f"     {'':34s} dropped {fmt(stat([r for r in part if r['r_open'] < t0], rng=rng))}")
    RES["by_season"] = {lab: stat([r for r in men3a if test(r) and r["r_open"] >= t0]) for lab, test in
                        (("first", lambda r: r in first), ("late_2025_26", lambda r: r["archived"] and r not in first), ("nov_2026", lambda r: not r["archived"]))}
    say("   Quote timing: last season's Kalshi 'tip' quote ends 25 min before an ESTIMATED start; where that is after ESPN's scheduled start the quote may be in-game.")
    for lab, test in (("quote window ends before ESPN's scheduled start", lambda r: r["quote_end_vs_sched"] <= 0),
                      ("quote window ends AFTER the scheduled start", lambda r: r["quote_end_vs_sched"] > 0),
                      ("... more than 10 min after", lambda r: r["quote_end_vs_sched"] > 10)):
        part = [r for r in men3a if test(r)]
        say(f"     {lab:48s} games {len(part):4d} | kept {fmt(stat([r for r in part if r['r_open'] >= t0], rng=rng))}")
        say(f"     {'':48s}            | dropped {fmt(stat([r for r in part if r['r_open'] < t0], rng=rng))}")

    say("\n   A5. the same rule elsewhere (does the idea generalise?)")
    for lab, lg, code, field in (("women, opening spread residual, 90-99c", "women", "3a", "r_open"), ("women, BPI residual, 90-99c", "women", "3a", "r_bpi"),
                                 ("men, BPI residual, 90-99c", "men", "3a", "r_bpi"), ("men, closing spread residual, 90-99c", "men", "3a", "r_spread"),
                                 ("men, moneyline residual, 90-99c", "men", "3a", "r_ml"),
                                 ("men, opening spread residual, 80-90c", "men", "3c", "r_open"), ("women, opening spread residual, 80-90c", "women", "3c", "r_open"),
                                 ("men, opening spread residual, dip fills", "men", "3b", "r_open"), ("women, BPI residual, dip fills", "women", "3b", "r_bpi")):
        keep, cost = next((k, c) for cd, k, c in TESTS if cd == code)
        part = [r for r in rows if r["league"] == lg and keep(r) and r.get(field) is not None]
        f1, f2 = halves(part)
        p80 = float(np.percentile([r[field] for r in f1], 80))
        for tl, t in (("fixed +2.70", 2.7), (f"own 80th pct {p80:+.2f}", p80)):
            say(f"     {lab:42s} {tl:20s} H1 {fmt(stat([r for r in f1 if r[field] >= t], cost, rng))}")
            say(f"     {'':42s} {'':20s} H2 {fmt(stat([r for r in f2 if r[field] >= t], cost, rng))}")
            say(f"     {'':42s} {'':20s} H2 dropped {fmt(stat([r for r in f2 if r[field] < t], cost, rng))}")
        RES.setdefault("elsewhere", {})[lab] = dict(h1=stat([r for r in f1 if r[field] >= p80], cost), h2=stat([r for r in f2 if r[field] >= p80], cost))

    say("\n   A6. is 63 of 63 unusual for games at these prices? (no search correction; one cell only)")
    alls = [r for r in rows if r["league"] == "men" and 90 <= r["mid"] < 99]
    X = np.array([r["mid"] for r in alls]); y = np.array([r["payout"] for r in alls])
    beta = logit_fit(X, y)
    kept2 = [r for r in second if r["r_open"] >= t0]
    kept1 = [r for r in first if r["r_open"] >= t0]
    for lab, part in (("second-half cell (63)", kept2), ("first-half cell (53, in-sample)", kept1), ("both", kept1 + kept2)):
        q = 1 / (1 + np.exp(-(beta[0] + beta[1] * np.array([r["mid"] for r in part]))))
        fair = np.array([(r["ask"] + fee(r["ask"])) / 100 for r in part])
        say(f"     {lab:32s} P(all win) if each wins at the rate of men's 90-99c games of its price (logistic in mid; mean {100*q.mean():.1f}%): {np.prod(q):.3f}"
            f" | if each wins at its break-even rate (mean {100*fair.mean():.1f}%): {np.prod(fair):.4f}")
    n2, l2 = len(second), sum(1 - r["payout"] for r in second)
    say(f"     second half has {int(l2)} losses in {n2} games; a random 63 of them misses every loss with probability {hyper0(n2, int(l2), 63):.3f}")
    n1, l1 = len(first), sum(1 - r["payout"] for r in first)
    say(f"     first half has {int(l1)} losses in {n1} games; a random 53 misses every loss with probability {hyper0(n1, int(l1), 53):.3f}")
    RES["single_cell"] = dict(p_random_subset_h2=hyper0(n2, int(l2), 63), p_random_subset_h1=hyper0(n1, int(l1), 53))


def hyper0(n, losses, k):
    p = 1.0
    for i in range(losses):
        p *= (n - k - i) / (n - i)
    return p


def logit_fit(x, y, iters=50):
    X = np.column_stack([np.ones(len(x)), (x - 95.0)])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        b = b + np.linalg.solve(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2), X.T @ (y - p))
    return np.array([b[0] - 95.0 * b[1], b[1]])


def _variant(rows, lab, rng, t0):
    part = [r for r in rows if r["league"] == "men" and 90 <= r["mid"] < 99 and r["r_open"] is not None]
    f1, f2 = halves(part)
    s1 = np.array([r["r_open"] for r in f1]); n1 = np.array([net_paid(r, "ask")[0] for r in f1])
    b = pick_rule(s1, n1)
    p80 = float(np.percentile(s1, 80))
    opts = [(f"rule re-picked: >= {b[1]:+.2f} ({b[2]}th pct)", b[1]), (f"80th pct: >= {p80:+.2f}", p80)]
    if t0 is not None:
        opts.append((f"fixed >= {t0:+.2f}", t0))
    for tl, t in opts:
        say(f"     {lab:22s} {tl:34s} H1 {fmt(stat([r for r in f1 if r['r_open'] >= t], rng=rng))}")
        say(f"     {'':22s} {'':34s} H2 {fmt(stat([r for r in f2 if r['r_open'] >= t], rng=rng))}")
    RES.setdefault("variants", {})[lab] = dict(threshold=b[1], pct=b[2], h2=stat([r for r in f2 if r["r_open"] >= b[1]]), h2_p80=stat([r for r in f2 if r["r_open"] >= p80]))


# ------------------------------------------------------------------------------------------------ b) permutation

def perm(n_perm=2000):
    rows = load()
    mem = members(rows)
    say("\n" + "=" * 118)
    say(f"B. PERMUTATION TEST FOR MULTIPLICITY ({n_perm} shuffles; the original's full search re-run each time)")
    say(f"   search family = every keep-above rule the original scored: {len(mem)} cells "
        f"(3 tests x men/women/both x 4 measures x raw/residual, those with >= 40 games)")
    obs = {}
    for d in mem:
        obs[d["name"]] = score(d["sig1"], d["net1"], d["sig2"], d["net2"], d["paid2"])
    say("   observed cells, ranked by studentised (second-half kept minus dropped):")
    ranked = sorted((o["z"], n) for n, o in obs.items() if o and not math.isnan(o["z"]))[::-1]
    for z, n in ranked[:8]:
        o = obs[n]
        say(f"     {n:26s} threshold {o['t']:+6.2f} (pct {o['q']:2d})  H1 kept {o['w1']}/{o['n1']}  H2 kept {o['w2']}/{o['n2']} ret {o['ret2']:+6.2f}%  kept-dropped {o['diff']:+5.2f}c  z {z:5.2f}")
    target = obs["3a men r_open"]
    say(f"   target cell: 3a men r_open  H2 {target['w2']}/{target['n2']} ret {target['ret2']:+.2f}% diff {target['diff']:+.2f}c z {target['z']:.2f}")
    fam3a = [d for d in mem if d["code"] == "3a"]
    fam3a_men = [d for d in fam3a if d["lg"] == "men"]
    rng = np.random.default_rng(2026)
    N = len(rows)
    hits = dict(single_ret=0, single_perfect=0, single_z=0, men3a_ret=0, men3a_perfect=0, men3a_z=0, fam3a_ret=0, fam3a_perfect=0, fam3a_z=0,
                all_z=0, both_perfect=0, fam3a_both_perfect=0)
    best_ret, best_z = [], []
    for it in range(n_perm):
        u = rng.random(N)
        sc = {}
        for d in mem:
            sig = []
            for h in ("1", "2"):
                s = d["sig" + h].copy()
                idx, isw = d["idx" + h], d["isw" + h]
                for w in (False, True):   # shuffle within league and half
                    pos = np.where(isw == w)[0]
                    if len(pos) > 1:
                        s[pos] = d["sig" + h][pos[np.argsort(u[idx[pos]])]]
                sig.append(s)
            sc[d["name"]] = score(sig[0], d["net1"], sig[1], d["net2"], d["paid2"])
        def good_ret(o):   # "looks as good": at least as many bets, at least the return
            return o and o["n2"] >= 50 and o["ret2"] >= target["ret2"] - 1e-9
        def perfect(o):
            return o and o["n2"] >= target["n2"] and o["w2"] == o["n2"]
        def zok(o):
            return o and not math.isnan(o["z"]) and o["z"] >= target["z"] - 1e-9
        s = sc["3a men r_open"]
        hits["single_ret"] += bool(good_ret(s)); hits["single_perfect"] += bool(perfect(s)); hits["single_z"] += bool(zok(s))
        for key, fam in (("men3a", fam3a_men), ("fam3a", fam3a)):
            hits[key + "_ret"] += any(good_ret(sc[d["name"]]) for d in fam)
            hits[key + "_perfect"] += any(perfect(sc[d["name"]]) for d in fam)
            hits[key + "_z"] += any(zok(sc[d["name"]]) for d in fam)
        hits["all_z"] += any(zok(o) for o in sc.values())
        hits["both_perfect"] += bool(perfect(s) and s["w1"] == s["n1"] and s["n1"] >= target["n1"])
        hits["fam3a_both_perfect"] += any(perfect(sc[d["name"]]) and sc[d["name"]]["w1"] == sc[d["name"]]["n1"] and sc[d["name"]]["n1"] >= 50 for d in fam3a)
        best_ret.append(max([sc[d["name"]]["ret2"] for d in fam3a if sc[d["name"]] and sc[d["name"]]["n2"] >= 50] or [float("nan")]))
        best_z.append(np.nanmax([o["z"] for o in sc.values() if o] or [float("nan")]))
    P = {k: (v + 1) / (n_perm + 1) for k, v in hits.items()}
    say("   share of shuffles in which the search turns up something at least as good (p-values, +1 corrected):")
    say(f"     this one cell only (men 3a opening residual; threshold still searched on the first half):")
    say(f"        second-half kept return >= {target['ret2']:+.2f}% on >= 50 bets      p = {P['single_ret']:.4f}")
    say(f"        second-half kept all wins on >= {target['n2']} bets              p = {P['single_perfect']:.4f}")
    say(f"        kept-minus-dropped z >= {target['z']:.2f}                        p = {P['single_z']:.4f}")
    say(f"        all wins in BOTH halves (>= {target['n1']} and >= {target['n2']} bets)        p = {P['both_perfect']:.4f}")
    say(f"     men's 3a cells (8: 4 measures x raw/residual):")
    say(f"        return >= {target['ret2']:+.2f}% on >= 50 bets  p = {P['men3a_ret']:.4f} | all wins on >= {target['n2']} bets  p = {P['men3a_perfect']:.4f} | z  p = {P['men3a_z']:.4f}")
    say(f"     all 3a cells (men, women, both; {len(fam3a)}):")
    say(f"        return >= {target['ret2']:+.2f}% on >= 50 bets  p = {P['fam3a_ret']:.4f} | all wins on >= {target['n2']} bets  p = {P['fam3a_perfect']:.4f} | z  p = {P['fam3a_z']:.4f}")
    say(f"        some 3a cell perfect in BOTH halves (>= 50 and >= {target['n2']} bets)   p = {P['fam3a_both_perfect']:.4f}")
    say(f"     the whole search ({len(mem)} cells, 3a+3b+3c): some cell with z >= {target['z']:.2f}   p = {P['all_z']:.4f}")
    say(f"     best 3a second-half kept return (>= 50 bets) across shuffles: median {np.nanmedian(best_ret):+.2f}%, 95th pct {np.nanpercentile(best_ret,95):+.2f}%, max {np.nanmax(best_ret):+.2f}% (observed {target['ret2']:+.2f}%)")
    RES["perm"] = dict(n=n_perm, cells=len(mem), p=P, target=target)

    # second null: outcomes re-drawn from the Kalshi price alone (signals untouched), 3a family
    say("   Second null (signals kept on their own games; outcomes re-drawn so that only the Kalshi price matters, win rate = logistic in mid per league):")
    hits2 = dict(single_ret=0, single_perfect=0, fam_ret=0, fam_perfect=0, single_both=0, fam_both=0)
    beta = {}
    for lg in ("men", "women"):
        part = [r for r in rows if r["league"] == lg and 90 <= r["mid"] < 99]
        beta[lg] = logit_fit(np.array([r["mid"] for r in part]), np.array([r["payout"] for r in part]))
    q = np.array([1 / (1 + math.exp(-(beta[r["league"]][0] + beta[r["league"]][1] * r["mid"]))) for r in rows])
    ask = np.array([r["ask"] for r in rows]); cost = ask + 7 * (ask / 100) * (1 - ask / 100)
    for it in range(n_perm):
        win = rng.random(N) < q
        net = 100 * win - cost
        sc = {d["name"]: score(d["sig1"], net[d["idx1"]], d["sig2"], net[d["idx2"]], d["paid2"]) for d in fam3a}
        s = sc["3a men r_open"]
        g = lambda o: o and o["n2"] >= 50 and o["ret2"] >= target["ret2"] - 1e-9
        pf = lambda o: o and o["n2"] >= target["n2"] and o["w2"] == o["n2"]
        hits2["single_ret"] += bool(g(s)); hits2["single_perfect"] += bool(pf(s))
        hits2["single_both"] += bool(pf(s) and s["w1"] == s["n1"])
        hits2["fam_ret"] += any(g(o) for o in sc.values()); hits2["fam_perfect"] += any(pf(o) for o in sc.values())
        hits2["fam_both"] += any(pf(o) and o["w1"] == o["n1"] and o["n1"] >= 50 for o in sc.values())
    P2 = {k: (v + 1) / (n_perm + 1) for k, v in hits2.items()}
    say(f"     this cell: return >= {target['ret2']:+.2f}% p = {P2['single_ret']:.4f}; all wins on >= {target['n2']} p = {P2['single_perfect']:.4f}; all wins in both halves p = {P2['single_both']:.4f}")
    say(f"     any of the {len(fam3a)} 3a cells: return p = {P2['fam_ret']:.4f}; all wins on >= {target['n2']} bets p = {P2['fam_perfect']:.4f}; all wins in both halves (>= 50 and >= {target['n2']}) p = {P2['fam_both']:.4f}")
    say("   Reading: the return and z statistics under the shuffle are flattered by price (the cell's games are cheaper than average, 95.5c against 96.1c, and an")
    say("   all-win cell's return is just its price); the shuffle breaks that link. The price-free 'all wins' counts and the second null are the fair ones.")
    RES["perm_null2"] = P2


# ------------------------------------------------------------------------------------------------ c) planted edge

def plant(sims=400, draws=300):
    rows = load()
    rng = np.random.default_rng(99)
    say("\n" + "=" * 118)
    say(f"C. PLANTED EDGE ({sims} simulated worlds per line). Outcomes are re-drawn: every game returns the test's observed average, and games whose")
    say("   residual is above the first-half 80th percentile get an EXTRA true edge. Then the original method is run unchanged.")
    say("   detected (rule) = keep-above rule picked on half 1, second-half kept-minus-dropped 95% interval above zero (the original's own flag)")
    say("   detected (slope) = slope of net on the signal, all games, 95% interval above zero;   'kept>0' = second-half kept return interval above zero")
    out = {}
    for lab, lg, code, field, edges in (
            ("3a men, opening residual", "men", "3a", "r_open", (0, 1.5, 3.0, 4.0, 6.0, 99)),
            ("3a women, BPI residual", "women", "3a", "r_bpi", (0, 1.5, 3.0, 4.0, 6.0, 99)),
            ("3b men, opening residual (dip fills)", "men", "3b", "r_open", (0, 1.5, 3.0, 5, 8, 10, 12, 15, 99)),
            ("3b women, BPI residual (dip fills)", "women", "3b", "r_bpi", (0, 1.5, 3.0, 5, 8, 10, 12, 15, 99)),
            ("3c men, opening residual", "men", "3c", "r_open", (0, 1.5, 3.0, 5, 8, 10, 12, 15, 99)),
            ("3c women, BPI residual", "women", "3c", "r_bpi", (0, 1.5, 3.0, 5, 8, 10, 12, 15, 99))):
        keep, cost = next((k, c) for cd, k, c in TESTS if cd == code)
        part = [r for r in rows if r["league"] == lg and keep(r) and r.get(field) is not None]
        f1, f2 = halves(part)
        order = f1 + f2
        n1 = len(f1)
        sig = np.array([r[field] for r in order])
        paid = np.array([net_paid(r, cost)[1] for r in order])
        base = float(np.mean([net_paid(r, cost)[0] for r in order]) / paid.mean())
        thr = float(np.percentile(sig[:n1], 80))
        planted = sig >= thr
        say(f"\n   {lab}: games {len(order)} (halves {n1}/{len(order)-n1}); base return {100*base:+.2f}%; planted games {int(planted.sum())} ({int(planted[n1:].sum())} in the second half)")
        res = []
        for e in edges:
            q = np.clip(paid / 100 * (1 + base + np.where(planted, min(e, 99) / 100, 0.0)), 0, 1)
            if e == 99:
                q = np.where(planted, 1.0, np.clip(paid / 100 * (1 + base), 0, 1))
            true_edge = 100 * float(((100 * q - paid)[planted]).sum() / paid[planted].sum()) - 100 * base
            det_rule = det_slope = det_kept = 0
            for _ in range(sims):
                net = 100 * (rng.random(len(order)) < q) - paid
                b = pick_rule(sig[:n1], net[:n1])
                if b is not None:
                    k2 = sig[n1:] >= b[1]
                    a, d = net[n1:][k2], net[n1:][~k2]
                    if len(a) >= 5 and len(d) >= 5:
                        da = a[rng.integers(0, len(a), (draws, len(a)))].mean(1)
                        dd = d[rng.integers(0, len(d), (draws, len(d)))].mean(1)
                        det_rule += np.percentile(da - dd, 2.5) > 0
                        det_kept += np.percentile(da, 2.5) > 0
                idx = rng.integers(0, len(order), (draws, len(order)))
                xs, ys = sig[idx], net[idx]
                xm = xs - xs.mean(1, keepdims=True)
                sl = (xm * (ys - ys.mean(1, keepdims=True))).mean(1) / np.maximum(xm.var(1), 1e-12)
                det_slope += np.percentile(sl, 2.5) > 0
            lab_e = "always win" if e == 99 else f"+{e:.1f}%"
            say(f"     planted extra edge {lab_e:>10s} (realised {true_edge:+5.2f}% after the 100% cap): detected by rule {100*det_rule/sims:5.1f}%  by slope {100*det_slope/sims:5.1f}%  kept>0 {100*det_kept/sims:5.1f}%")
            res.append(dict(edge=e, realised=true_edge, rule=det_rule / sims, slope=det_slope / sims, kept=det_kept / sims))
        ok = [r for r in res if r["edge"] not in (0,) and max(r["rule"], r["slope"]) >= 0.8]
        mde = (f"about +{ok[0]['realised']:.1f}% (first tested level with >= 80% power)" if ok else "none: even a planted certainty is found less than 80% of the time")
        say(f"     minimum detectable extra edge (80% power, better of rule/slope): {mde}")
        out[lab] = dict(n=len(order), base=base, planted=int(planted.sum()), rows=res, mde=mde)
    RES["plant"] = out


# ------------------------------------------------------------------------------------------------ d) mechanism

def mech():
    rows = load()
    rng = np.random.default_rng(5)
    say("\n" + "=" * 118)
    say("D. MECHANISM: is the cell a Kalshi lag behind a line move (open -> close)?   move = closing minus opening spread, points for the favourite")
    part = [r for r in rows if r["league"] == "men" and 90 <= r["mid"] < 99 and r["r_open"] is not None and r["r_spread"] is not None]
    f1, f2 = halves([r for r in rows if r["league"] == "men" and 90 <= r["mid"] < 99 and r["r_open"] is not None])
    t0 = pick_rule(np.array([r["r_open"] for r in f1]), np.array([net_paid(r, 'ask')[0] for r in f1]))[1]
    cell = [r for r in part if r["r_open"] >= t0]
    rest = [r for r in part if r["r_open"] < t0]
    for lab, rs in (("cell (r_open >= %.2f)" % t0, cell), ("other men 3a games", rest)):
        mv = np.array([r["move"] for r in rs]); rc = np.array([r["r_spread"] for r in rs])
        say(f"   {lab:26s} n {len(rs):4d}: move mean {mv.mean():+.2f} pts (down {100*(mv<0).mean():.0f}%, flat {100*(mv==0).mean():.0f}%, up {100*(mv>0).mean():.0f}%); "
            f"opening spread mean {np.mean([r['open'] for r in rs]):.1f}, closing {np.mean([r['close'] for r in rs]):.1f}, Kalshi mid {np.mean([r['mid'] for r in rs]):.1f}; "
            f"closing residual mean {rc.mean():+.2f}c; corr(r_open, r_close) in group {np.corrcoef([r['r_open'] for r in rs], rc)[0,1]:+.2f}")
    say("   If Kalshi lagged the move, the edge would sit where the line moved; if the OPEN itself carried news the close/Kalshi lost, where it moved down.")
    say("   2x2, all men 3a games (both halves): opening residual high/low x direction of the move")
    for la, ta in (("r_open >= t", lambda r: r["r_open"] >= t0), ("r_open <  t", lambda r: r["r_open"] < t0)):
        for lb, tb in (("line moved DOWN (against fav)", lambda r: r["move"] < 0), ("line flat", lambda r: r["move"] == 0), ("line moved UP (toward fav)", lambda r: r["move"] > 0)):
            say(f"     {la} & {lb:30s} {fmt(stat([r for r in part if ta(r) and tb(r)], rng=rng))}")
    say("   the move on its own as a rule (all men 3a games; and second half only):")
    _, s2 = halves(part)
    for lb, tb in (("move <= -2", lambda r: r["move"] <= -2), ("move <= -1", lambda r: r["move"] <= -1), ("move < 0", lambda r: r["move"] < 0), ("move == 0", lambda r: r["move"] == 0),
                   ("move > 0", lambda r: r["move"] > 0), ("move >= +1", lambda r: r["move"] >= 1), ("move >= +2", lambda r: r["move"] >= 2)):
        say(f"     {lb:12s} all {fmt(stat([r for r in part if tb(r)], rng=rng))}")
        say(f"     {'':12s} H2  {fmt(stat([r for r in s2 if tb(r)], rng=rng))}")
    say("   the CLOSING residual at the same percentile and at the same threshold (does the close do the same job?):")
    c1, c2 = halves(part)
    p80 = float(np.percentile([r["r_spread"] for r in c1], 80))
    for tl, t in ((f"r_close >= {p80:+.2f} (80th pct)", p80), (f"r_close >= {t0:+.2f}", t0)):
        say(f"     {tl:30s} H1 {fmt(stat([r for r in c1 if r['r_spread'] >= t], rng=rng))}")
        say(f"     {'':30s} H2 {fmt(stat([r for r in c2 if r['r_spread'] >= t], rng=rng))}")
    say("   inside/outside split by BOTH residuals (all men 3a games):")
    for la, ta in (("r_open >= t", lambda r: r["r_open"] >= t0), ("r_open <  t", lambda r: r["r_open"] < t0)):
        for lb, tb in ((f"r_close >= {p80:+.2f}", lambda r: r["r_spread"] >= p80), (f"r_close <  {p80:+.2f}", lambda r: r["r_spread"] < p80)):
            say(f"     {la} & {lb:18s} {fmt(stat([r for r in part if ta(r) and tb(r)], rng=rng))}")
    # logistic regression of the win on mid, r_close, and the open-minus-close part
    y = np.array([r["payout"] for r in part])
    X = np.column_stack([np.ones(len(part)), [r["mid"] - 95 for r in part], [r["r_spread"] for r in part], [r["r_open"] - r["r_spread"] for r in part]])
    def fit(X, y):
        b = np.zeros(X.shape[1])
        for _ in range(60):
            p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
            W = p * (1 - p)
            b = b + np.linalg.solve(X.T @ (X * W[:, None]) + 1e-3 * np.eye(X.shape[1]), X.T @ (y - p) - 1e-3 * b)
        return b
    b = fit(X, y)
    bs = np.array([fit(X[i], y[i]) for i in rng.integers(0, len(y), (600, len(y)))])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    say("   logistic regression of the favourite winning (men 3a, all games, lightly ridge-stabilised; bootstrap 95%):")
    for i, nm in enumerate(("intercept", "Kalshi mid (per cent)", "closing residual (per cent)", "opening minus closing residual (per cent)")):
        say(f"     {nm:42s} {b[i]:+.3f} [{lo[i]:+.3f},{hi[i]:+.3f}]")
    # spread size: is the cell just "big opening spreads"?
    say("   is the cell just big spreads? men 3a by opening spread size, all games:")
    for lo_, hi_ in ((0, 14), (14, 17), (17, 20), (20, 24), (24, 99)):
        say(f"     open in [{lo_:2d},{hi_:2d}) {fmt(stat([r for r in part if lo_ <= r['open'] < hi_], rng=rng))}")
    say("   and at a matched Kalshi price: cell games vs other games in the same 1c ask band (all men 3a)")
    for lo_, hi_ in ((90, 94), (94, 96), (96, 97.5), (97.5, 100)):
        say(f"     ask [{lo_:5.1f},{hi_:5.1f})  cell {fmt(stat([r for r in cell if lo_ <= r['ask'] < hi_], rng=rng))}")
        say(f"     {'':18s} rest {fmt(stat([r for r in rest if lo_ <= r['ask'] < hi_], rng=rng))}")

    say("   does 'the line moved against the favourite' work anywhere else? (move < 0 against move >= 0; H1/H2 = chronological halves of each group)")
    for lab, lg, code in (("men 90-99c", "men", "3a"), ("men 80-90c", "men", "3c"), ("women 90-99c", "women", "3a"), ("women 80-90c", "women", "3c"), ("men dip fills", "men", "3b"), ("women dip fills", "women", "3b")):
        keep, cost = next((k, c) for cd, k, c in TESTS if cd == code)
        grp = [r for r in rows if r["league"] == lg and keep(r) and r["move"] is not None]
        g1, g2 = halves(grp)
        for hl, h in (("H1", g1), ("H2", g2)):
            say(f"     {lab:16s} {hl} moved down {fmt(stat([r for r in h if r['move'] < 0], cost, rng))}")
            say(f"     {'':16s}    flat or up {fmt(stat([r for r in h if r['move'] >= 0], cost, rng))}")
    say("   look-ahead probe: if ESPN's 'open' had been overwritten with later information it would predict the final margin better than the close does.")
    for lab, grp in (("men 90-99c", part), ("men 80-99c", [r for r in rows if r["league"] == "men" and r["move"] is not None and r["open"] > 0]),
                     ("men 80-99c Nov 2026", [r for r in rows if r["league"] == "men" and r["move"] is not None and r["open"] > 0 and not r["archived"]])):
        g = [r for r in grp if r["open"] > 0]
        yy = np.array([r["margin"] for r in g]); cc = np.array([r["close"] for r in g]); oo = np.array([r["open"] for r in g])
        XX = np.column_stack([np.ones(len(g)), cc, oo - cc])
        bb = np.linalg.lstsq(XX, yy, rcond=None)[0]
        se = np.sqrt(np.diag(np.linalg.inv(XX.T @ XX)) * (yy - XX @ bb).var(ddof=3))
        say(f"     {lab:20s} n {len(g):4d}: margin = a + {bb[1]:.2f} x close + {bb[2]:+.2f} (se {se[2]:.2f}) x (open - close); rmse of open {np.sqrt(((yy-oo)**2).mean()):.2f}, of close {np.sqrt(((yy-cc)**2).mean()):.2f} points")
    say("\n   FORWARD TEST SIZE (bets on the frozen rule: men, 90-99c, opening-spread residual >= %.2f, at the ask)" % t0)
    paid = float(np.mean([net_paid(r, "ask")[1] for r in cell]))
    nov = [r for r in cell if not r["archived"]]
    days = len({r["day"] for r in rows if r["league"] == "men" and not r["archived"]})
    say(f"     mean cost in the cell {paid:.2f}c -> break-even win rate {paid:.1f}%; the most the cell can ever pay is {100*(100-paid)/paid:+.2f}% (every game wins)")
    say(f"     rate: {len(nov)} qualifying games in {days} November 2026 days = {len(nov)/days:.1f} a day")
    for h0, lab0 in ((paid / 100, "break-even"), (0.971, "the plain basket's 97.1%")):
        for h1 in (0.985, 0.99, 0.995):
            n = need_n(h0, h1)
            say(f"     to tell a true win rate of {100*h1:.1f}% (return {100*(100*h1-paid)/paid:+.1f}%) from {lab0} ({100*h0:.1f}%) with 5% false alarm and 80% power: about {n} bets (~{n/(len(nov)/days):.0f} days at the November rate)")
    for p0 in (paid / 100, 0.971):
        n = math.ceil(math.log(0.05) / math.log(p0))
        say(f"     quick kill: if the cell is no better than {100*p0:.1f}%, a loss-free run of {n} bets happens only 5% of the time; expect the first loss within about {1/(1-p0):.0f} bets")
    RES["forward"] = dict(paid=paid, per_day=len(nov) / days)


def need_n(p0, p1, alpha=0.05, power=0.8):
    """Smallest n for an exact one-sided binomial test of p0 against p1 > p0."""
    for n in range(20, 6000):
        # critical k: smallest k with P(X >= k | p0) <= alpha
        k = n
        while k > 0 and 1 - binom_cdf(k - 2, n, p0) <= alpha:
            k -= 1
        if 1 - binom_cdf(k - 1, n, p0) > alpha:
            continue
        if 1 - binom_cdf(k - 1, n, p1) >= power:
            return n
    return None


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if what in ("robust", "all"):
        robust()
    if what in ("perm", "all"):
        perm(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
    if what in ("plant", "all"):
        plant()
    if what in ("mech", "all"):
        mech()
    name = "stress" if what == "all" else "stress_" + what
    open(os.path.join(OUT, name + ".txt"), "w").write("\n".join(LINES) + "\n")
    json.dump(RES, open(os.path.join(OUT, name + ".json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
