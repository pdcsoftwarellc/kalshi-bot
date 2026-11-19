"""Blind rebuild of the short-dated study (audit). Written before reading research/short_dated.py's body or its report.

Reads only the cached bars under data/raw/short-dated. No network, no orders.
  v1 bars: [minute_end_ts, open_bid, open_ask, close_bid, close_ask, volume]   (YES side, cents)
  v2 bars: [minute_end_ts, close_bid, close_ask, trade_low, trade_high, volume]

Test A (taker): first bar in a time-to-close bucket where a side's closing ask is in a band; buy at the next
  minute's opening ask (v1) if still in band, else (v2-only families) at the decision bar's closing ask; pay
  0.07*p*(1-p); hold to settlement. Return in % of stake (price + fee).
Test B (resting bid): first bar in a bucket where the favourite's closing bid is 90-99; rest a bid at
  join / bid-3 / 80c; filled only if a LATER bar's trade range prints strictly through it. No maker fee
  (series fee_type is plain "quadratic"). Return in % of the bid price.
Intervals: 2,000 bootstrap resamples of events (pooled: of close timestamps). Halves: chronological by close.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/short-dated"
OUT = ROOT / "data/processed/audit/short_dated"
FAMILIES = {  # name: (v1 files, v2 file)
    "btc15m": (["btc15m-bars.json"], "v2/btc15m.json"),
    "eth15m": (["eth15m-bars.json"], "v2/eth15m.json"),
    "sol15m": ([], "v2/sol15m.json"),
    "xrp15m": ([], "v2/xrp15m.json"),
    "btc_hourly": (["btc_hourly-bars.json", "btc_hourly-bars-2026-10-29.json"], "v2/btc_hourly.json"),
    "eth_hourly": (["eth_hourly-bars.json"], "v2/eth_hourly.json"),
    "sp_hourly": (["index_hourly-bars.json"], "v2/sp_hourly.json"),
    "nasdaq_hourly": ([], "v2/nasdaq_hourly.json"),
    "btc_daily": ([], "v2/btc_daily.json"),
    "eth_daily": ([], "v2/eth_daily.json"),
    "sp_daily": ([], "v2/sp_daily.json"),
}
BUCKETS = [(1, 2), (3, 5), (6, 15), (16, 60), (61, 360), (361, 1500)]
BANDS = [(60, 80), (80, 90), (90, 95), (95, 98), (98, 99.5)]
RNG = np.random.default_rng(7)


def fee(p):
    return 0.07 * p * (100 - p) / 100


def load(name):
    v1files, v2file = FAMILIES[name]
    v1 = {}
    for f in v1files:
        for m in json.load(open(RAW / f)):
            v1[m["ticker"]] = {b[0]: b for b in m["bars"]}
    return json.load(open(RAW / v2file)), v1


def stats(rows, cluster_key=1):
    """rows: (ret_pct, event, close, win). Returns dict."""
    if not rows:
        return None
    r = np.array([x[0] for x in rows])
    keys = [x[cluster_key] for x in rows]
    idx = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    c = np.array([idx[k] for k in keys])
    n = len(idx)
    s = np.bincount(c, r, n); k = np.bincount(c, minlength=n).astype(float)
    pick = RNG.integers(0, n, (2000, n))
    boot = s[pick].sum(1) / k[pick].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    closes = np.array([x[2] for x in rows])
    med = np.median(np.array(sorted({(x[1], x[2]) for x in rows}, key=lambda t: t[1]))[:, 1].astype(float))
    h1, h2 = r[closes <= med], r[closes > med]
    return dict(n=len(r), ev=n, win=100 * np.mean([x[3] for x in rows]), mean=r.mean(), lo=lo, hi=hi,
                h1=h1.mean() if len(h1) else float("nan"), h2=h2.mean() if len(h2) else float("nan"),
                n1=len(h1), n2=len(h2))


def fmt(label, s):
    if s is None:
        return f"{label:<46} n=0"
    flag = "  *" if (s["lo"] > 0 and s["h1"] > 0 and s["h2"] > 0) else ("  -" if s["hi"] < 0 else "")
    return (f"{label:<46} n={s['n']:>6} ev={s['ev']:>5} win={s['win']:5.1f}% ret={s['mean']:+7.2f}% "
            f"[{s['lo']:+7.2f},{s['hi']:+7.2f}] halves {s['h1']:+7.2f}% ({s['n1']}) / {s['h2']:+7.2f}% ({s['n2']}){flag}")


def bucket_of(mins):
    for lo, hi in BUCKETS:
        if lo <= mins <= hi:
            return (lo, hi)
    return None


def run():
    lines = []
    taker = defaultdict(list)   # (family, bucket, band) -> rows
    rest = defaultdict(list)    # (family, bucket, kind) -> rows
    restn = defaultdict(int)    # placements
    for name in FAMILIES:
        markets, v1 = load(name)
        for m in markets:
            if m["outcome"] not in (0.0, 1.0):
                continue
            bars = m["bars"]; close = m["close"]; o1 = v1.get(m["ticker"])
            seenA, seenB = set(), set()
            ts_index = {b[0]: i for i, b in enumerate(bars)}
            for i, b in enumerate(bars):
                ts, cb, ca = b[0], b[1], b[2]
                if cb is None or ca is None or ts >= close:
                    continue
                bk = bucket_of(round((close - ts) / 60))
                if bk is None:
                    continue
                # ---- test A
                for side, ask in (("yes", ca), ("no", 100 - cb)):
                    for band in BANDS:
                        if not (band[0] <= ask < band[1]) or (bk, band) in seenA:
                            continue
                        if o1 is not None:
                            nb = o1.get(ts + 60)
                            if nb is None:
                                continue
                            p = nb[2] if side == "yes" else 100 - nb[1]
                            if not (band[0] <= p < band[1]):
                                continue
                        else:
                            if ts + 60 not in ts_index and ts + 60 <= close:
                                pass  # sparse bars: quote unchanged is assumed
                            p = ask
                        win = m["outcome"] == (1.0 if side == "yes" else 0.0)
                        stake = p + fee(p)
                        taker[(name, bk, band)].append(((100 * win - stake) / stake * 100, m["event"], close, win))
                        seenA.add((bk, band))
                # ---- test B
                if bk in seenB:
                    continue
                if cb >= 90 and cb <= 99:
                    side, fb = "yes", cb
                elif 100 - ca >= 90 and 100 - ca <= 99:
                    side, fb = "no", 100 - ca
                else:
                    continue
                seenB.add(bk)
                win = m["outcome"] == (1.0 if side == "yes" else 0.0)
                later = [x for x in bars[i + 1:] if x[3] is not None]
                for kind, q in (("join", fb), ("bid-3", fb - 3), ("80c", 80.0)):
                    restn[(name, bk, kind)] += 1
                    if side == "yes":
                        hit = any(x[3] < q for x in later)
                    else:
                        hit = any(x[4] > 100 - q for x in later)
                    if hit:
                        rest[(name, bk, kind)].append(((100 * win - q) / q * 100, m["event"], close, win))
        print(name, "done", file=sys.stderr)

    lines.append("BLIND REBUILD - short-dated. Returns in % of stake; 95% event-clustered bootstrap; * = CI>0 and both halves >0; - = CI<0")
    lines.append("")
    lines.append("TEST A: taker buys favourite-side ask in band (next-minute open ask where v1 bars exist), fee 0.07p(1-p), hold")
    for name in FAMILIES:
        for bk in BUCKETS:
            for band in BANDS:
                rows = taker.get((name, bk, band))
                if rows and len(rows) >= 30:
                    lines.append(fmt(f"A {name} {bk[0]}-{bk[1]}m ask {band[0]}-{band[1]}", stats(rows)))
    lines.append("")
    lines.append("TEST A pooled by family (all buckets <=60m, ask 80-99.5), one row per market-bucket-band")
    groups = {"crypto15m": ["btc15m", "eth15m", "sol15m", "xrp15m"], "crypto_hourly": ["btc_hourly", "eth_hourly"],
              "index_hourly": ["sp_hourly", "nasdaq_hourly"], "daily": ["btc_daily", "eth_daily", "sp_daily"]}
    for g, fams in groups.items():
        for band in BANDS:
            rows = [r for (n, bk, bd), v in taker.items() if n in fams and bd == band for r in v]
            lines.append(fmt(f"A pooled {g} ask {band[0]}-{band[1]} (cluster=close ts)", stats(rows, 2)))
        rows = [r for (n, bk, bd), v in taker.items() if n in fams and bd[0] >= 80 for r in v]
        lines.append(fmt(f"A pooled {g} ask 80-99.5 ALL (cluster=close ts)", stats(rows, 2)))
    lines.append("")
    lines.append("TEST B: resting bid on favourite with bid 90-99; fill only on strict print-through in a later bar; no fee")
    for name in FAMILIES:
        for bk in BUCKETS:
            for kind in ("join", "bid-3", "80c"):
                rows = rest.get((name, bk, kind))
                if rows and len(rows) >= 30:
                    lines.append(fmt(f"B {name} {bk[0]}-{bk[1]}m {kind} fill {len(rows)}/{restn[(name, bk, kind)]}", stats(rows)))
    lines.append("")
    lines.append("TEST B pooled (cluster=close ts)")
    for g, fams in groups.items():
        for kind in ("join", "bid-3", "80c"):
            rows = [r for (n, bk, kd), v in rest.items() if n in fams and kd == kind for r in v]
            lines.append(fmt(f"B pooled {g} {kind}", stats(rows, 2)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "blind.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
