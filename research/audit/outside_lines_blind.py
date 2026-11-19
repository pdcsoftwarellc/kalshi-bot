"""Blind rebuild of research/outside_lines.py (audit; written WITHOUT reading that script's body or its report).

Inputs (read-only): data/processed/outside-lines/matched.json (Kalshi quote, outcome, ESPN ids) and the raw ESPN cache
data/raw/outside-lines/{men,women}/{scoreboard,odds,predictor}. Every ESPN number is re-derived here from the raw cache
(spread sign, favourite orientation, open vs close, moneyline, BPI, winner); matched.json supplies only the Kalshi side
and the event pairing. No network access.

Output: data/processed/audit/outside_lines/blind.txt
"""
import difflib
import glob
import json
import math
import os
import random
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data/raw/outside-lines")
MATCHED = os.path.join(ROOT, "data/processed/outside-lines/matched.json")
OUT = os.path.join(ROOT, "data/processed/audit/outside_lines")
MEASURES = ("close", "ml", "open", "bpi")
NBOOT = 4000


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm(s):
    s = s.lower().replace("&", " and ")
    s = re.sub(r"\bst\.?(?=\s|$)", "state", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def american(x):
    """American point spread / moneyline string -> float (EVEN/PK -> 0 for spreads, +100 for moneylines handled by caller)."""
    if x is None:
        return None
    x = str(x).strip().upper()
    if x in ("EVEN", "PK", "PICK", "EV"):
        return 0.0
    try:
        return float(x)
    except ValueError:
        return None


def ml_prob(a):
    if a is None:
        return None
    if a == 0:
        a = 100.0
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def load_scoreboards(league):
    idx = {}
    for f in glob.glob(os.path.join(RAW, league, "scoreboard", "*.json")):
        for g in json.load(open(f)):
            idx[g["id"]] = g
    return idx


def team_id(ref):
    m = re.search(r"/teams/(\d+)", ref or "")
    return m.group(1) if m else None


def side_numbers(side):
    """(open spread, close spread, open ml, close ml) for one team's odds block, each from that team's own point of view."""
    def ps(ph):
        return american(((side.get(ph) or {}).get("pointSpread") or {}).get("american"))

    def ml(ph):
        return american(((side.get(ph) or {}).get("moneyLine") or {}).get("american"))
    return ps("open"), ps("close"), ml("open"), ml("close")


def rebuild_row(r, sb):
    """Independent ESPN numbers for one matched row. Everything from the favourite's point of view."""
    lg = r["league"]
    out = dict(ok=True, notes=[])
    g = sb.get(r["espn"])
    if g is None:
        out["ok"] = False
        out["notes"].append("no scoreboard game")
        return out
    t = g["teams"]
    # which ESPN team is the Kalshi favourite: best name similarity over the name variants
    def best(name, team):
        return max(sim(name, team[k]) for k in ("location", "short", "display") if team.get(k))
    s0 = best(r["fav"], t[0]) - best(r["opp"], t[0])
    s1 = best(r["fav"], t[1]) - best(r["opp"], t[1])
    fi = 0 if s0 > s1 else 1
    out["orient_margin"] = abs(s0 - s1)
    fav, opp = t[fi], t[1 - fi]
    out["fav_espn"] = fav["location"]
    out["fav_home"] = fav["home"]
    out["neutral"] = g.get("neutral")
    try:
        out["margin"] = float(fav["score"]) - float(opp["score"])
    except (TypeError, ValueError):
        out["margin"] = None
    out["won"] = bool(fav.get("winner"))
    out["espn_start"] = g["date"]
    for k in ("open", "close", "ml", "ml_open", "bpi", "provider"):
        out[k] = None
    # odds
    p = os.path.join(RAW, lg, "odds", r["espn"] + ".json")
    if os.path.exists(p):
        items = [i for i in json.load(open(p)).get("items", []) if "live" not in i["provider"]["name"].lower()]
        items.sort(key=lambda i: (i["provider"]["name"] != r.get("provider"), i["provider"].get("priority", 9)))
        out["n_providers"] = len(items)
        if items:
            it = items[0]
            out["provider"] = it["provider"]["name"]
            h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
            # check the team refs agree with the scoreboard's home/away
            home_sb = [x for x in t if x["home"]][0]
            hid = team_id((h.get("team") or {}).get("$ref"))
            if hid and hid != home_sb["id"]:
                out["notes"].append("odds home team id != scoreboard home id")
            fs, os_ = (h, a) if fav["home"] else (a, h)
            fo, fc, fmo, fmc = side_numbers(fs)
            oo, oc, omo, omc = side_numbers(os_)
            # favourite's points = minus the favourite's own point spread; cross-check with the other side
            if fc is not None:
                out["close"] = -fc
                if oc is not None and abs(oc + fc) > 1e-9:
                    out["notes"].append("close spreads of the two sides do not mirror")
            if fo is not None:
                out["open"] = -fo
                if oo is not None and abs(oo + fo) > 1e-9:
                    out["notes"].append("open spreads of the two sides do not mirror")
            top = it.get("spread")  # top-level spread is from the home side (negative = home favoured)
            if top is not None:
                out["top_spread_fav"] = -top if fav["home"] else top
            out["details"] = it.get("details")
            if fmc is not None and omc is not None:
                a_, b_ = ml_prob(fmc), ml_prob(omc)
                out["ml"] = 100 * a_ / (a_ + b_)
            if fmo is not None and omo is not None:
                a_, b_ = ml_prob(fmo), ml_prob(omo)
                out["ml_open"] = 100 * a_ / (a_ + b_)
    # BPI
    p = os.path.join(RAW, lg, "predictor", r["espn"] + ".json")
    if os.path.exists(p):
        d = json.load(open(p))
        side = d.get("homeTeam") if fav["home"] else d.get("awayTeam")
        for s in (side or {}).get("statistics", []):
            if s["name"] == "gameProjection" and s.get("value") is not None:
                out["bpi"] = float(s["value"])
        out["bpi_modified"] = d.get("lastModified")
        hid = team_id(((d.get("homeTeam") or {}).get("team") or {}).get("$ref"))
        home_sb = [x for x in t if x["home"]][0]
        if hid and hid != home_sb["id"]:
            out["notes"].append("predictor home team id != scoreboard home id")
    return out


def iso_ts(s):
    import calendar, time
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%MZ")) if s else None


def build():
    rows = json.load(open(MATCHED))
    data = {}
    for lg in ("men", "women"):
        sb = load_scoreboards(lg)
        rs = sorted([r for r in rows if r["league"] == lg], key=lambda r: (r["start"], r["event"]))
        half = len(rs) // 2
        for i, r in enumerate(rs):
            r["b"] = rebuild_row(r, sb)
            r["half"] = 1 if i < half else 2
            b = r["b"]
            # BPI recomputed after the scheduled start is unusable (ESPN's start time)
            if b.get("bpi") is not None and b.get("bpi_modified") and b.get("espn_start"):
                b["bpi_late"] = iso_ts(b["bpi_modified"]) > iso_ts(b["espn_start"])
                if b["bpi_late"]:
                    b["bpi"] = None
        # sigma: sd of (favourite margin - closing spread) on the first half
        e = [r["b"]["margin"] - r["b"]["close"] for r in rs if r["half"] == 1 and r["b"].get("close") is not None
             and r["b"].get("margin") is not None]
        sigma = float(np.std(e, ddof=1))
        for r in rs:
            b = r["b"]
            r["P"] = {
                "close": 100 * phi(b["close"] / sigma) if b.get("close") is not None else None,
                "open": 100 * phi(b["open"] / sigma) if b.get("open") is not None else None,
                "ml": b.get("ml"),
                "bpi": b.get("bpi"),
            }
        data[lg] = dict(rows=rs, sigma=sigma, n_sigma=len(e), mean_err=float(np.mean(e)))
    return data


def fee(p_cents):
    p = p_cents / 100
    return 7 * p * (1 - p)


def ret_ask(r):
    cost = r["ask"] + fee(r["ask"])
    return 100 * (100 * r["payout"] - cost) / cost


def ret_dip(r, maker_fee=0.0):
    cost = 80 + maker_fee
    return 100 * (100 * r["payout"] - cost) / cost


TESTS = {
    "3a fav 90-99c at ask": (lambda r: 90 <= r["mid"] < 99, ret_ask),
    "3b dip fills at 80c": (lambda r: r["fill"], ret_dip),
    "3c fav 80-90c at ask": (lambda r: 80 <= r["mid"] < 90, ret_ask),
}


def fit_resid(rs, m, deg=2, fit_half=1):
    """Residual of outside prob on a polynomial in Kalshi's mid, fitted on one chronological half of the league."""
    tr = [r for r in rs if r["half"] == fit_half and r["P"][m] is not None]
    if len(tr) < 10:
        return None
    c = np.polyfit([r["mid"] for r in tr], [r["P"][m] for r in tr], deg)
    return c


def signal(r, m, kind, coef):
    p = r["P"][m]
    if p is None:
        return None
    if kind == "raw":
        return p - r["ask"]
    if coef is None:
        return None
    return p - float(np.polyval(coef, r["mid"]))


def boot(x, rng, n=NBOOT):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), (n, len(x)))
    return tuple(np.percentile(x[idx].mean(1), [2.5, 97.5]))


def boot_diff(a, b, rng, n=NBOOT):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    da = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    return tuple(np.percentile(da - db, [2.5, 97.5]))


def cell(x, rng):
    if not len(x):
        return "n=0"
    lo, hi = boot(x, rng)
    return f"n={len(x):4d} wins={sum(1 for v in x if v > 0):4d} ret={np.mean(x):+6.2f}% [{lo:+6.2f},{hi:+6.2f}]"


def spot_check(data, w):
    w("=" * 110)
    w("DATA CHECK: every ESPN number re-derived from the raw cache and compared with matched.json")
    for lg, d in data.items():
        rs = d["rows"]
        c = dict(n=len(rs), winner_bad=0, margin_bad=0, close_bad=0, open_bad=0, ml_bad=0, bpi_bad=0, orient_weak=0,
                 notes=0, close_neg=0, open_neg=0, top_vs_close=0, provider_diff=0, late=0)
        bad = []
        for r in rs:
            b = r["b"]
            if not b["ok"]:
                bad.append((r["event"], "no scoreboard"))
                continue
            if b["won"] != (r["payout"] == 1.0):
                c["winner_bad"] += 1; bad.append((r["event"], "winner", b["won"], r["payout"]))
            if b["margin"] != r["margin"]:
                c["margin_bad"] += 1; bad.append((r["event"], "margin", b["margin"], r["margin"]))
            for k, tol in (("close", 1e-9), ("open", 1e-9), ("ml", 1e-6), ("bpi", 1e-6)):
                mine, theirs = b.get(k), r.get(k if k != "bpi" else "p_bpi")
                if (mine is None) != (theirs is None) or (mine is not None and abs(mine - theirs) > tol):
                    c[k + "_bad"] += 1; bad.append((r["event"], k, mine, theirs, b.get("provider"), r.get("provider")))
            if b["orient_margin"] < 0.3:
                c["orient_weak"] += 1
            if b["notes"]:
                c["notes"] += 1; bad.append((r["event"], "notes", b["notes"]))
            if b.get("close") is not None and b["close"] < 0: c["close_neg"] += 1
            if b.get("open") is not None and b["open"] < 0: c["open_neg"] += 1
            if b.get("top_spread_fav") is not None and b.get("close") is not None and abs(b["top_spread_fav"] - b["close"]) > 1e-9:
                c["top_vs_close"] += 1
            if b.get("provider") != r.get("provider"): c["provider_diff"] += 1
            if b.get("bpi_late"): c["late"] += 1
        w(f"\n{lg}: {c}")
        w(f"  sigma (first half, n={d['n_sigma']}): {d['sigma']:.3f}   mean(margin - close) = {d['mean_err']:+.2f}")
        # their sigma implied by p_spread
        imp = []
        from statistics import NormalDist
        for r in rs:
            if r.get("p_spread") and r.get("close") and 1 < r["p_spread"] < 99.99:
                imp.append(r["close"] / NormalDist().inv_cdf(r["p_spread"] / 100))
        if imp:
            w(f"  sigma implied by matched.json p_spread: {np.median(imp):.3f} (min {min(imp):.3f}, max {max(imp):.3f})")
        for x in bad[:25]:
            w(f"  MISMATCH {x}")
        if len(bad) > 25:
            w(f"  ... {len(bad) - 25} more")
        # correlation sanity: favourite-oriented spread must correlate positively with margin and with Kalshi mid
        for k in ("close", "open"):
            xs = [(r["b"][k], r["b"]["margin"], r["mid"]) for r in rs if r["b"].get(k) is not None]
            if len(xs) > 5:
                a = np.array(xs)
                w(f"  corr({k}, margin) = {np.corrcoef(a[:,0], a[:,1])[0,1]:+.3f}   corr({k}, kalshi mid) = {np.corrcoef(a[:,0], a[:,2])[0,1]:+.3f}   n={len(xs)}")
        xs = [(r["P"]["bpi"], r["mid"]) for r in rs if r["P"]["bpi"] is not None]
        a = np.array(xs)
        w(f"  corr(bpi, kalshi mid) = {np.corrcoef(a[:,0], a[:,1])[0,1]:+.3f}   n={len(xs)}")
        # quote time relative to ESPN start: 'hours' in matched.json is |kalshi start - espn start|
        late = [r for r in rs if r["b"]["ok"] and r["start"] - iso_ts(r["b"]["espn_start"]) > 600]
        w(f"  Kalshi 'start' more than 10 min AFTER ESPN's scheduled start: {len(late)} of {len(rs)}")
    # 25 random rows printed for eyeballing
    w("\n25 random matched rows, raw-cache view (fav = Kalshi favourite):")
    rng = random.Random(20261119)
    allrows = [r for d in data.values() for r in d["rows"] if r["b"]["ok"]]
    for r in rng.sample(allrows, 25):
        b = r["b"]
        w(f"  {r['event']:<38} kalshi fav={r['fav']!r:<24} espn fav={b['fav_espn']!r:<22} home={b['fav_home']!s:<5} "
          f"details={b.get('details')!r:<14} open={b.get('open')} close={b.get('close')} (json {r['open']}/{r['close']}) "
          f"ml={None if b.get('ml') is None else round(b['ml'],1)} bpi={b.get('bpi')} margin={b['margin']} won={b['won']} payout={r['payout']} mid={r['mid']}")


def analyse(data, w):
    rng = np.random.default_rng(7)
    head = []
    for lg, d in data.items():
        rs = d["rows"]
        w("\n" + "=" * 110)
        w(f"{lg.upper()}  games={len(rs)}  first half ends {rs[len(rs)//2-1]['day']}  second half starts {rs[len(rs)//2]['day']}")
        for tname, (flt, retf) in TESTS.items():
            sub = [r for r in rs if flt(r)]
            w(f"\n--- {lg} {tname}: base")
            for lab, hs in (("all", (1, 2)), ("H1", (1,)), ("H2", (2,))):
                x = [retf(r) for r in sub if r["half"] in hs]
                w(f"  {lab:<3} {cell(x, rng)}")
            for m in MEASURES:
                coef = fit_resid(rs, m)
                for kind in ("raw", "resid"):
                    sg = [(signal(r, m, kind, coef), r) for r in sub]
                    sg = [(s, r) for s, r in sg if s is not None]
                    h1 = [s for s, r in sg if r["half"] == 1]
                    if len(h1) < 9 or len(sg) < 30:
                        w(f"  {m:<5} {kind:<5} too few rows (n={len(sg)}, first half {len(h1)})")
                        continue
                    for cutlab, cutsrc in (("cuts from first half", h1), ("cuts from all games", [s for s, _ in sg])):
                        lo_c, hi_c = np.percentile(cutsrc, [100 / 3, 200 / 3])
                        w(f"  {m:<5} {kind:<5} n={len(sg)}  {cutlab}: {lo_c:+.2f} / {hi_c:+.2f}")
                        for lab, hs in (("all", (1, 2)), ("H1", (1,)), ("H2", (2,))):
                            T = [[], [], []]
                            for s, r in sg:
                                if r["half"] in hs:
                                    T[0 if s < lo_c else (1 if s < hi_c else 2)].append(retf(r))
                            for ti, nm in enumerate(("low", "mid", "high")):
                                w(f"      {lab:<3} {nm:<4} {cell(T[ti], rng)}")
                            if T[0] and T[2]:
                                dl, dh = boot_diff(T[2], T[0], rng)
                                diff = np.mean(T[2]) - np.mean(T[0])
                                w(f"      {lab:<3} high-low {diff:+6.2f}% [{dl:+6.2f},{dh:+6.2f}]")
                                if cutlab.startswith("cuts from first"):
                                    head.append((lg, tname, m, kind, lab, len(T[0]), len(T[2]), np.mean(T[0]), np.mean(T[2]), diff, dl, dh,
                                                 sum(1 for v in T[2] if v > 0)))
    w("\n" + "=" * 110)
    w("HEADLINE: high tercile minus low tercile (tercile cuts fixed on the first half), return % per bet, 95% bootstrap")
    w(f"{'league':<6} {'test':<22} {'meas':<5} {'kind':<5} {'smp':<3} {'nLow':>4} {'nHigh':>5} {'winsHi':>6} {'low%':>7} {'high%':>7} {'diff%':>7}  interval")
    for h in head:
        flag = "  <-- excludes 0" if (h[10] > 0 or h[11] < 0) else ""
        w(f"{h[0]:<6} {h[1]:<22} {h[2]:<5} {h[3]:<5} {h[4]:<3} {h[5]:>4} {h[6]:>5} {h[12]:>6} {h[7]:>+7.2f} {h[8]:>+7.2f} {h[9]:>+7.2f}  [{h[10]:+.2f},{h[11]:+.2f}]{flag}")
    n = len(head)
    k = sum(1 for h in head if h[10] > 0 or h[11] < 0)
    w(f"\n{k} of {n} headline comparisons have an interval excluding zero (about {0.05*n:.0f} expected by chance if nothing is there).")


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def w(s=""):
        lines.append(s)
    data = build()
    w("BLIND REBUILD of the outside-lines study (written before reading the original script body, report or summary)")
    w("Return = (100 x settlement - ask - fee) / (ask + fee), fee = 7 x p x (1-p) cents; dip fills: bought at 80c, no fee assumed.")
    w("Halves: chronological halves of ALL matched games of the league. Residual: outside prob minus quadratic in Kalshi mid, fitted on half 1.")
    spot_check(data, w)
    analyse(data, w)
    open(os.path.join(OUT, "blind.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
