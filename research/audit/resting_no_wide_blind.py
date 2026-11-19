"""Blind rebuild (audit) of the resting-NO-bid test outside mention markets.

Written before reading research/resting_no_wide.py beyond its first 30 docstring
lines or anything under data/processed/resting-no-wide/. Offline: reads only the
cached files under data/raw/resting-no-wide/.

Bar layout (checked against one public candlestick call):
  [end_ts, yes_bid_close, yes_ask_close, trade_low, trade_high, volume, yes_bid_high, yes_ask_low]

Order: when a bar closes with 1 <= bid < ask <= 99 and ask-bid >= 2, rest a YES
offer at ask-1 (= NO bid at 100-ask+1). Filled only if a LATER bar's trade high is
strictly above ask-1. Mirror: YES bid at bid+1, filled if a later trade low is
strictly below it. Held to settlement. Maker fee 0.0175*p*(1-p) only on
maker-fee series. Returns are % of stake; intervals resample events.
"""
import json, sys, random, collections, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/resting-no-wide"
OUT = ROOT / "data/processed/audit/resting_no_wide"
WEEK = 7 * 86400


def load_all():
    s = json.load(open(RAW / "sample.json"))
    rh = json.load(open(RAW / "recent-hourly.json"))
    rd = json.load(open(RAW / "recent-daily.json"))
    ar = json.load(open(RAW / "archive-bars.json"))
    out = []
    for period in ("recent", "archive"):
        for m in s[period]:
            t = m["ticker"]
            if m.get("result") not in ("yes", "no"):
                continue
            if period == "recent":
                h, d = rh.get(t) or [], rd.get(t) or []
            else:
                b = ar.get(t) or {}
                h, d = b.get("hourly") or [], b.get("daily") or []
            out.append((m, merge(h, d)))
    return out


def merge(h, d):
    """Hourly bars, then daily bars that start after the last hourly bar ends."""
    h = sorted(h)
    last = h[-1][0] if h else 0
    tail = [b for b in sorted(d) if b[0] - 86400 >= last]
    return h + tail


def fee(price, maker):
    return 0.0175 * price * (100 - price) / 100 if maker else 0.0   # cents per contract


def try_order(m, bars, i, until):
    """Quote at bars[i]; orders rest over bars after i with end_ts <= until. Returns (no_row, yes_row) or None."""
    ts, bid, ask = bars[i][0], bars[i][1], bars[i][2]
    if bid is None or ask is None or not (1 <= bid < ask <= 99) or ask - bid < 2:
        return None
    offer, ybid = ask - 1, bid + 1
    nofill = yesfill = False
    vol_no = 0.0
    for b in bars[i + 1:]:
        if b[0] > until:
            break
        lo, hi = b[3], b[4]
        if hi is not None and hi > offer:
            nofill = True; vol_no += b[5] or 0
        if lo is not None and lo < ybid:
            yesfill = True
    res = m["result"]
    cno, cyes = 100 - offer, ybid
    rno = ((100 if res == "no" else 0) - cno - fee(cno, m["maker_fee"])) / cno * 100
    ryes = ((100 if res == "yes" else 0) - cyes - fee(cyes, m["maker_fee"])) / cyes * 100
    base = dict(ticker=m["ticker"], event=m["event"], family=m["family"], ts=ts, bid=bid, ask=ask, period=m["period"])
    return (dict(base, fill=nofill, ret=rno, cost=cno, vol=vol_no), dict(base, fill=yesfill, ret=ryes, cost=cyes, vol=0))


def orders(data, rule):
    no, yes = [], []
    for m, bars in data:
        if not bars:
            continue
        idx = []
        if rule == "first2c":
            for i, b in enumerate(bars):
                if b[0] <= m["open"]:
                    continue
                if b[1] is not None and b[2] is not None and 1 <= b[1] < b[2] <= 99 and b[2] - b[1] >= 2:
                    idx.append((i, m["close"])); break
        elif rule == "weekly":
            # every Monday 05:00 UTC while open; quote = last bar ending at or before that time (within 24h)
            t = (m["open"] // WEEK) * WEEK + 4 * 86400 + 5 * 3600   # epoch day 0 is Thursday -> +4d = Monday
            while t <= m["open"]:
                t += WEEK
            j = 0
            while t < m["close"]:
                while j + 1 < len(bars) and bars[j + 1][0] <= t:
                    j += 1
                if bars[j][0] <= t and t - bars[j][0] <= 86400:
                    idx.append((j, min(t + WEEK, m["close"])))
                t += WEEK
        for i, until in idx:
            r = try_order(m, bars, i, until)
            if r:
                no.append(r[0]); yes.append(r[1])
    return no, yes


def boot(rows, n=2000, seed=7):
    """Event-clustered bootstrap of the mean return per fill and the stake-weighted return."""
    ev = collections.defaultdict(list)
    for r in rows:
        ev[r["event"]].append(r)
    keys = list(ev)
    if len(keys) < 2:
        return (float("nan"),) * 2
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = c = 0.0
        for _ in keys:
            for r in ev[rng.choice(keys)]:
                s += r["ret"]; c += 1
        means.append(s / c)
    means.sort()
    return means[int(.025 * n)], means[int(.975 * n)]


def stat(rows):
    f = [r for r in rows if r["fill"]]
    if not f:
        return None
    mean = sum(r["ret"] for r in f) / len(f)
    stake = sum(r["ret"] * r["cost"] for r in f) / sum(r["cost"] for r in f)
    lo, hi = boot(f)
    f.sort(key=lambda r: r["ts"])
    evs = []
    for r in f:
        if r["event"] not in evs:
            evs.append(r["event"])
    first = set(evs[:len(evs) // 2])
    h1 = [r["ret"] for r in f if r["event"] in first]
    h2 = [r["ret"] for r in f if r["event"] not in first]
    m = lambda x: sum(x) / len(x) if x else float("nan")
    return dict(orders=len(rows), fills=len(f), events=len(evs), mean=mean, stake=stake, lo=lo, hi=hi, h1=m(h1), h2=m(h2),
                n1=len(h1), n2=len(h2), vol=sum(r["vol"] for r in f), cost=sum(r["cost"] for r in f) / len(f),
                win=sum(r["ret"] > 0 for r in f) / len(f))


def main():
    data = load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"markets with a result: {len(data)}; with bars: {sum(1 for _, b in data if b)}"]
    for rule in ("first2c", "weekly"):
        no, yes = orders(data, rule)
        fams = sorted({r["family"] for r in no})
        table = []
        for fam in fams + ["ALL"]:
            a = stat([r for r in no if fam == "ALL" or r["family"] == fam])
            b = stat([r for r in yes if fam == "ALL" or r["family"] == fam])
            if a:
                table.append((fam, a, b))
        table.sort(key=lambda x: -(x[1]["mean"] * x[1]["fills"]))
        lines.append(f"\n== rule {rule}: NO bid at 100-ask+1 (ranked by mean return x fills) ==")
        lines.append(f"{'family':36s} {'ord':>5s} {'fill':>5s} {'ev':>4s} {'cost':>5s} {'win%':>5s} {'mean%':>7s} {'95% CI':>17s} {'stake%':>7s} {'h1%':>7s} {'h2%':>7s} {'mirror%':>8s} {'mirror CI':>17s} {'mfill':>5s} pass")
        for fam, a, b in table:
            ok = a["lo"] > 0 and a["h1"] > 0 and a["h2"] > 0
            bm = f"{b['mean']:8.1f} [{b['lo']:6.1f},{b['hi']:6.1f}] {b['fills']:5d}" if b else " " * 32
            lines.append(f"{fam:36s} {a['orders']:5d} {a['fills']:5d} {a['events']:4d} {a['cost']:5.1f} {100*a['win']:5.1f} {a['mean']:7.1f} [{a['lo']:6.1f},{a['hi']:6.1f}] {a['stake']:7.1f} {a['h1']:7.1f} {a['h2']:7.1f} {bm} {'PASS' if ok else ''}")
        json.dump(dict(no=no, yes=yes), open(OUT / f"blind-{rule}.json", "w"))
    (OUT / "blind.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
