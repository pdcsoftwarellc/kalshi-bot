"""Blind rebuild of the overround study (audit). Research only, cached public data, no orders.

Written from the first 30 lines of research/overround.py's docstring only.
Reads data/raw/overround/*.json (the original's download cache). Bars are
[end_ts, yes_bid_close, yes_ask_close, trade_low, trade_high, volume] (checked
against trades.json: a trade lands in the bar whose ts is the hour's END).

A. Taker: at T = modal close - 24h, buy NO at 100 - YES bid on every outcome whose
   bid exceeds the taker fee 7*p*(1-p)*mult cents. Held to settlement.
B. Maker: YES offer at ask-1 on outcomes with 1 <= bid < ask <= 99, spread >= 2.
   Filled only if a LATER hourly bar's trade high is strictly above the offer.
   Mirror: YES bid at bid+1 filled by a later low strictly below.
Returns are sum(profit)/sum(stake) in %, event-clustered bootstrap, halves by event close.
"""
import json, glob, os, sys, random, collections, math

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data/raw/overround/")
OUT = os.path.join(ROOT, "data/processed/audit/overround/blind.txt")


def load():
    series = {s["ticker"]: s for s in json.load(open(RAW + "series.json"))}
    ev = json.load(open(RAW + "events.json"))
    bars = {}
    for f in ("archive-bars", "recent-bars"):
        bars.update({k: v for k, v in json.load(open(RAW + f + ".json")).items() if v})
    out = []
    for s, v in ev.items():
        for e in v["events"]:
            ms = e["markets"]
            if not e["me"] or len(ms) < 4:
                continue
            if not all(m["result"] in ("yes", "no") for m in ms):
                continue
            if sum(m["volume"] for m in ms) < 5000:
                continue
            if sum(m["result"] == "yes" for m in ms) > 1:
                continue
            if not all(m["ticker"] in bars for m in ms):
                continue
            closes = collections.Counter(m["close"] for m in ms)
            close = closes.most_common(1)[0][0]
            out.append(dict(series=s, event=e["event"], close=close, markets=ms,
                            cat=series.get(s, {}).get("category", "?"),
                            ftype=series.get(s, {}).get("fee_type", "quadratic"),
                            mult=series.get(s, {}).get("fee_multiplier", 1) or 1))
    return out, bars


def quote_idx(b, T, stale=48 * 3600):
    i = -1
    for j, r in enumerate(b):
        if r[0] <= T:
            i = j
        else:
            break
    if i < 0 or T - b[i][0] > stale or b[i][1] is None or b[i][2] is None:
        return None
    return i


def legs(events, bars, lead):
    """One row per (event, outcome) with quote and later-print info."""
    rows = []
    for e in events:
        T = e["close"] - lead
        q = []
        for m in e["markets"]:
            b = bars[m["ticker"]]
            i = quote_idx(b, T)
            if i is None:
                q = None
                break
            later = [r for r in b[i + 1:] if r[3] is not None]
            hi = max((r[4] for r in later), default=None)
            lo = min((r[3] for r in later), default=None)
            q.append(dict(bid=b[i][1], ask=b[i][2], hi=hi, lo=lo, yes=m["result"] == "yes"))
        if q is None:
            continue
        rows.append((e, q))
    return rows


def tfee(p, mult):
    return 7 * (p / 100) * (1 - p / 100) * mult


def mfee(p, e):
    return 1.75 * (p / 100) * (1 - p / 100) * e["mult"] if e["ftype"] == "quadratic_with_maker_fees" else 0.0


def boot(per_event, n=2000, seed=1):
    """per_event: list of (profit, stake). Ratio of sums, cluster bootstrap."""
    per_event = [x for x in per_event if x[1] > 0]
    if not per_event:
        return None
    tot = lambda xs: 100 * sum(p for p, s in xs) / sum(s for p, s in xs)
    rng = random.Random(seed)
    k = len(per_event)
    sims = sorted(tot([per_event[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return tot(per_event), sims[int(.025 * n)], sims[int(.975 * n)]


def line(name, rows_pe, nb):
    """rows_pe: list of (close, profit, stake, bets) per event, chronological halves."""
    rows_pe = sorted([r for r in rows_pe if r[2] > 0])
    if len(rows_pe) < 4:
        return f"{name:44s} n_events={len(rows_pe)} (too few)"
    h = len(rows_pe) // 2
    f = lambda rs: boot([(r[1], r[2]) for r in rs])
    a, h1, h2 = f(rows_pe), f(rows_pe[:h]), f(rows_pe[h:])
    return (f"{name:44s} events={len(rows_pe):5d} bets={sum(r[3] for r in rows_pe):6d} "
            f"ret={a[0]:+7.2f}% [{a[1]:+7.2f},{a[2]:+7.2f}]  H1={h1[0]:+7.2f}% [{h1[1]:+.2f},{h1[2]:+.2f}]  "
            f"H2={h2[0]:+7.2f}% [{h2[1]:+.2f},{h2[2]:+.2f}]")


def study(events, bars, lead, label, out):
    R = legs(events, bars, lead)
    out.append(f"\n=== clock {label}: {len(R)} events with a quote on every outcome ===")
    sb = sorted(sum(x["bid"] for x in q) for e, q in R)
    sa = sorted(sum(x["ask"] for x in q) for e, q in R)
    md = lambda v: v[len(v) // 2]
    out.append(f"sum YES bids median {md(sb):.0f} (>100: {sum(v > 100 for v in sb)}), sum YES asks median {md(sa):.0f} (>100: {sum(v > 100 for v in sa)})")

    def taker(filt):
        pe = []
        for e, q in R:
            L = [x for x in q if x["bid"] >= 1 and x["bid"] > tfee(x["bid"], e["mult"])]
            locked = sum(x["bid"] - tfee(x["bid"], e["mult"]) for x in L) - 100
            if not L or not filt(e, q, locked):
                continue
            stake = sum(100 - x["bid"] + tfee(x["bid"], e["mult"]) for x in L)
            pay = sum(0 if x["yes"] else 100 for x in L)
            pe.append((e["close"], pay - stake, stake, len(L)))
        return pe
    out.append("A. TAKER buy NO at 100 - YES bid, + fee, on every outcome")
    out.append(line("  all events", taker(lambda e, q, l: True), 0))
    out.append(line("  sum YES bids > 100", taker(lambda e, q, l: sum(x['bid'] for x in q) > 100), 0))
    out.append(line("  locked after fee > 0", taker(lambda e, q, l: l > 0), 0))

    def maker(filt, side="no", allfill=False, legf=lambda x: True):
        pe = []
        for e, q in R:
            if not filt(e, q):
                continue
            p = s = n = 0
            for x in q:
                if not (1 <= x["bid"] < x["ask"] <= 99 and x["ask"] - x["bid"] >= 2) or not legf(x):
                    continue
                if side == "no":
                    o = x["ask"] - 1
                    if not allfill and not (x["hi"] is not None and x["hi"] > o):
                        continue
                    fee = mfee(o, e)
                    p += o - (100 if x["yes"] else 0) - fee
                    s += 100 - o + fee
                else:
                    o = x["bid"] + 1
                    if not allfill and not (x["lo"] is not None and x["lo"] < o):
                        continue
                    fee = mfee(o, e)
                    p += (100 if x["yes"] else 0) - o - fee
                    s += o + fee
                n += 1
            pe.append((e["close"], p, s, n))
        return pe
    T = lambda e, q: True
    over = lambda e, q: sum(x["ask"] - 1 if x["ask"] - x["bid"] >= 2 else x["ask"] for x in q) > 100
    out.append("B. MAKER YES offer at ask-1 (= NO bid), fill on later print strictly above")
    out.append(line("  all events", maker(T), 0))
    out.append(line("  events with sum of offers > 100", maker(over), 0))
    out.append(line("  IF every order filled (no selection)", maker(T, allfill=True), 0))
    for lo, hi in ((2, 10), (11, 30), (31, 70), (71, 98)):
        out.append(line(f"  offer {lo}-{hi}c", maker(T, legf=lambda x: lo <= x['ask'] - 1 <= hi), 0))
    for c in sorted({e["cat"] for e, q in R}):
        out.append(line(f"  category {c}", maker(lambda e, q: e["cat"] == c), 0))
    out.append("   Mirror control: YES bid at bid+1, fill on later print strictly below")
    out.append(line("  all events", maker(T, side="yes"), 0))
    out.append(line("  IF every order filled", maker(T, side="yes", allfill=True), 0))
    # fill composition
    nf = nw = no = ow = 0
    for e, q in R:
        for x in q:
            if 1 <= x["bid"] < x["ask"] <= 99 and x["ask"] - x["bid"] >= 2:
                no += 1; ow += x["yes"]
                if x["hi"] is not None and x["hi"] > x["ask"] - 1:
                    nf += 1; nw += x["yes"]
    out.append(f"  maker orders {no}, winners among orders {100*ow/max(no,1):.1f}%; filled {nf}, winners among fills {100*nw/max(nf,1):.1f}%")


def main():
    events, bars = load()
    out = [f"Blind rebuild, overround. Universe: mutually exclusive, 4+ settled outcomes, 5000+ contracts, <=1 YES, hourly bars on every outcome: {len(events)} events",
           "Returns = sum profit / sum stake in %, 95% event-clustered bootstrap, halves chronological by event close."]
    study(events, bars, 24 * 3600, "24h before close", out)
    study(events, bars, 7 * 24 * 3600, "7d before close", out)
    study(events, bars, 3600, "1h before close", out)
    txt = "\n".join(out)
    print(txt)
    open(OUT, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
