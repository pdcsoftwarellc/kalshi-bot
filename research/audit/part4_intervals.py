"""Audit helper (inventory part 4): cluster-bootstrap intervals from SAVED rows only. Offline, read-only.

Reads data/processed/{favorites,far-strikes,weather-state}/ and data/raw/crypto-forward/*.jsonl.
Prints to stdout; writes nothing. Run: .venv/bin/python research/audit/part4_intervals.py [section ...]
Sections: fav rest elect far weather crypto
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data/processed"
DRAWS = 4000


def boot(rows, num, den, key, draws=DRAWS, seed=11):
    """Ratio sum(num)/sum(den) with a 95% interval resampling whole clusters. Returns (est, lo, hi, clusters)."""
    if not rows:
        return None
    sums = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        s = sums[key(r)]
        s[0] += num(r)
        s[1] += den(r)
    c = np.array(list(sums.values()))
    est = c[:, 0].sum() / c[:, 1].sum()
    if len(c) < 5:
        return est, float("nan"), float("nan"), len(c)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(c), (draws, len(c)))
    d = c[pick, 1].sum(1)
    b = c[pick, 0].sum(1) / np.where(d == 0, np.nan, d)
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return est, float(lo), float(hi), len(c)


def fmt(x):
    if x is None:
        return "n/a"
    est, lo, hi, n = x
    return f"{est:+7.2f} [{lo:+7.2f},{hi:+7.2f}] cl={n}"


def week(ts):
    d = datetime.fromtimestamp(ts, timezone.utc).isocalendar()
    return f"{d[0]}-W{d[1]:02d}"


def split60(rows, when="ref_ts"):
    first = {}
    for r in rows:
        first[r["event"]] = min(r[when], first.get(r["event"], float("inf")))
    order = sorted(first, key=lambda e: (first[e], e))
    early = set(order[:int(round(.6 * len(order)))])
    return [r for r in rows if r["event"] in early], [r for r in rows if r["event"] not in early]


def halves(rows, when="ref_ts"):
    first = {}
    for r in rows:
        first[r["event"]] = min(r[when], first.get(r["event"], float("inf")))
    order = sorted(first, key=lambda e: (first[e], e))
    a = set(order[:len(order) // 2])
    return [r for r in rows if r["event"] in a], [r for r in rows if r["event"] not in a]


def line(name, rows, net, stake, extra=""):
    """cents per bet and % per bet (total net / total stake), by event, series and ISO week clusters."""
    if not rows:
        print(f"  {name:34} no rows")
        return
    one = lambda r: 1.0
    ev = len({r["event"] for r in rows})
    c_ev = boot(rows, net, one, lambda r: r["event"])
    p_ev = boot(rows, lambda r: 100 * net(r), stake, lambda r: r["event"])
    p_se = boot(rows, lambda r: 100 * net(r), stake, lambda r: r["series"])
    p_wk = boot(rows, lambda r: 100 * net(r), stake, lambda r: week(r["ref_ts"]))
    print(f"  {name:34} n={len(rows):4} ev={ev:4} | c/bet {fmt(c_ev)} | %/bet event {fmt(p_ev)} | series {fmt(p_se)} | week {fmt(p_wk)} {extra}")


def fav():
    rows = json.loads((P / "favorites/entries.json").read_text())
    net = lambda r: r["net_cents"]
    stake = lambda r: r["price"] + r["fee_cents"]
    print("FAVORITES TAKER (entries.json): %/bet = total net / total (price + fee); mean price",
          round(float(np.mean([r["price"] for r in rows])), 2))
    print("  fee types of entries:", Counter((r["fee_type"], r["fee_multiplier"]) for r in rows))
    early, late = split60(rows)
    h1, h2 = halves(rows)
    for name, part in (("all rules, whole sample", rows), ("all rules, earliest 60% events", early), ("all rules, latest 40% events", late),
                       ("all rules, first half events", h1), ("all rules, second half events", h2)):
        line(name, part, net, stake)
    # one equal-weight number per event (events carry up to 15 rule rows each)
    def per_event(part):
        by = defaultdict(list)
        for r in part:
            by[r["event"]].append(r)
        vals = np.array([sum(net(r) for r in g) / sum(stake(r) for r in g) * 100 for g in by.values()])
        rng = np.random.default_rng(3)
        b = vals[rng.integers(0, len(vals), (DRAWS, len(vals)))].mean(1)
        return f"{vals.mean():+.2f}% [{np.percentile(b, 2.5):+.2f},{np.percentile(b, 97.5):+.2f}] events={len(vals)}"
    print("  equal weight per event (mean of event returns): all", per_event(rows), "| early60", per_event(early), "| late40", per_event(late))
    print("  rows per event:", Counter(Counter(r["event"] for r in rows).values()).most_common())
    for b in ("90-94", "95-97", "98-99"):
        sub = [r for r in rows if r["band"] == b]
        e, l = split60(sub)
        line(f"band {b} all", sub, net, stake)
        line(f"band {b} early60", e, net, stake)
        line(f"band {b} late40", l, net, stake)
    for d in (1, 3, 7, 14, 30):
        sub = [r for r in rows if r["days"] == d]
        e, l = split60(sub)
        line(f"D{d} all", sub, net, stake, f"| early60 c {np.mean([net(r) for r in e]):+.2f} late40 c {np.mean([net(r) for r in l]):+.2f}")
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        e, l = split60(sub)
        ee = f"{np.mean([net(r) for r in e]):+.2f}" if e else "n/a"
        ll = f"{np.mean([net(r) for r in l]):+.2f}" if l else "n/a"
        line(f"cat {cat[:22]} all", sub, net, stake, f"| early60 c {ee} late40 c {ll}")
    liquid = [r for r in rows if r["volume_24h"] >= 100 and r["spread"] <= 3]
    line("liquid subset of plain rows", liquid, net, stake)

    print("\n  LATE-PERIOD CONCENTRATION (latest 40% of events, all rules)")
    tot = sum(net(r) for r in late)
    print(f"   late rows {len(late)} events {len({r['event'] for r in late})} total net {tot:+.1f}c mean {tot / len(late):+.3f}c")
    by = defaultdict(list)
    for r in late:
        by[r["event"]].append(r)
    losers = sorted(((sum(net(r) for r in g), e, g) for e, g in by.items() if any(r["payout"] < .5 for r in g)))
    print(f"   events with a full loss: {len(losers)}; losing rows: {sum(r['payout'] < .5 for r in late)}")
    for s, e, g in losers:
        print(f"    {e:34} {g[0]['category'][:14]:14} rows {len(g)} lost {sum(r['payout'] < .5 for r in g)} net {s:+8.1f}c "
              f"week {week(g[0]['ref_ts'])} rules {sorted((r['days'], r['band']) for r in g if r['payout'] < .5)}")
    print("   loss events by series:", Counter(g[0]["series"] for _, _, g in losers).most_common())
    print("   loss events by week:", Counter(week(g[0]["ref_ts"]) for _, _, g in losers).most_common())
    print("   loss events by category:", Counter(g[0]["category"] for _, _, g in losers).most_common())
    for k in (1, 2, 3, 5, 8):
        drop = {e for _, e, _ in losers[:k]}
        keep = [r for r in late if r["event"] not in drop]
        print(f"   late mean without the {k} worst events: {np.mean([net(r) for r in keep]):+.3f}c  %/bet {fmt(boot(keep, lambda r: 100 * net(r), stake, lambda r: r['event']))}")
    for cat in ("Commodities", "Economics", "Entertainment", "World", "Financials"):
        keep = [r for r in late if r["category"] != cat]
        print(f"   late mean without category {cat}: {np.mean([net(r) for r in keep]):+.3f}c  %/bet {fmt(boot(keep, lambda r: 100 * net(r), stake, lambda r: r['event']))}")
    # early period: same list
    eby = defaultdict(list)
    for r in early:
        eby[r["event"]].append(r)
    el = [(e, g) for e, g in eby.items() if any(r["payout"] < .5 for r in g)]
    print(f"   EARLY 60%: events with a full loss {len(el)} of {len(eby)}; late: {len(losers)} of {len(by)}")
    # Fisher-style check on event loss rate early vs late by bootstrap
    ev_loss = [(1.0 if any(r["payout"] < .5 for r in g) else 0.0) for g in eby.values()]
    lv_loss = [(1.0 if any(r["payout"] < .5 for r in g) else 0.0) for g in by.values()]
    print(f"   event loss rate early {np.mean(ev_loss):.3f} late {np.mean(lv_loss):.3f}")
    print("   late period span:", datetime.fromtimestamp(min(r["ref_ts"] for r in late), timezone.utc).date(),
          datetime.fromtimestamp(max(r["ref_ts"] for r in late), timezone.utc).date(),
          "| early:", datetime.fromtimestamp(min(r["ref_ts"] for r in early), timezone.utc).date(),
          datetime.fromtimestamp(max(r["ref_ts"] for r in early), timezone.utc).date())


def rest():
    rows = json.loads((P / "favorites/resting.json").read_text())
    print("\nFAVORITES RESTING BIDS (resting.json): bid = ask0 - 1, 24h, fill = print strictly through")
    print("  orders", len(rows), "events", len({r["event"] for r in rows}), "truncated tapes", sum(r["truncated"] for r in rows),
          "fee types", Counter(r["fee_type"] for r in rows))
    hit = [r for r in rows if r["filled"]]
    miss = [r for r in rows if not r["filled"]]
    net = lambda r: r["maker_net"]
    stake = lambda r: r["bid"]
    print("  maker fee actually charged (c) on filled rows by fee type:",
          {k: round(float(np.mean([100 * r['payout'] - r['bid'] - r['maker_net'] for r in hit if r['fee_type'] == k])), 4)
           for k in {r['fee_type'] for r in hit}})
    early, late = split60(rows)
    h1, h2 = halves(rows)
    for name, part in (("filled, whole sample", hit), ("filled, early60", [r for r in early if r["filled"]]),
                       ("filled, late40", [r for r in late if r["filled"]]), ("filled, first half", [r for r in h1 if r["filled"]]),
                       ("filled, second half", [r for r in h2 if r["filled"]])):
        line(name, part, net, stake)
    for b in ("90-94", "95-97", "98-99"):
        line(f"filled band {b}", [r for r in hit if r["band"] == b], net, stake)
    for cat in sorted({r["category"] for r in rows}):
        line(f"filled cat {cat[:22]}", [r for r in hit if r["category"] == cat], net, stake)
    # per order placed (unfilled = 0)
    print("  per ORDER placed (unfilled earn 0), c:", fmt(boot(rows, lambda r: r["maker_net"] if r["filled"] else 0.0, lambda r: 1.0, lambda r: r["event"])))
    # adverse-selection claim: loss rate filled minus unfilled, event-clustered
    def diff(part, key, seed=5):
        groups = defaultdict(list)
        for r in part:
            groups[key(r)].append(r)
        g = list(groups.values())
        rng = np.random.default_rng(seed)
        out = []
        arr = np.array([[sum(r["filled"] and r["payout"] < .5 for r in x), sum(bool(r["filled"]) for r in x),
                         sum((not r["filled"]) and r["payout"] < .5 for r in x), sum(not r["filled"] for r in x)] for x in g], float)
        est = arr[:, 0].sum() / arr[:, 1].sum() - arr[:, 2].sum() / arr[:, 3].sum()
        pick = rng.integers(0, len(arr), (DRAWS, len(arr)))
        s = arr[pick].sum(1)
        b = s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]
        return f"{100 * est:+.2f}pp [{100 * np.nanpercentile(b, 2.5):+.2f},{100 * np.nanpercentile(b, 97.5):+.2f}] clusters={len(arr)}"
    print(f"  loss rate filled {np.mean([r['payout'] < .5 for r in hit]):.4f} ({sum(r['payout'] < .5 for r in hit)}/{len(hit)}) "
          f"unfilled {np.mean([r['payout'] < .5 for r in miss]):.4f} ({sum(r['payout'] < .5 for r in miss)}/{len(miss)})")
    print("  filled-minus-unfilled loss rate: by event", diff(rows, lambda r: r["event"]), "| by series", diff(rows, lambda r: r["series"]),
          "| by week", diff(rows, lambda r: week(r["ref_ts"])))
    print("  distinct EVENTS behind the losses: filled", len({r["event"] for r in hit if r["payout"] < .5}),
          "unfilled", len({r["event"] for r in miss if r["payout"] < .5}),
          "both", len({r["event"] for r in hit if r["payout"] < .5} & {r["event"] for r in miss if r["payout"] < .5}))
    # like for like: same price, filled vs unfilled valued at settlement
    for name, part in (("filled", hit), ("unfilled", miss)):
        print(f"  settlement value minus bid, {name}: c", fmt(boot(part, net, lambda r: 1.0, lambda r: r["event"])))
    # no-print orders
    quiet = [r for r in rows if r["prints"] == 0]
    print(f"  orders with no print at all in 24h: {len(quiet)} ({len(quiet) / len(rows):.1%}); among unfilled {sum(r['prints'] == 0 for r in miss)}/{len(miss)}")
    # touched-but-not-through
    t = [r for r in rows if r["touched"] and not r["filled"]]
    print(f"  touched but not through: {len(t)}; if those were fills, filled net c:",
          fmt(boot([r for r in rows if r["touched"]], net, lambda r: 1.0, lambda r: r["event"])),
          "%/bet", fmt(boot([r for r in rows if r["touched"]], lambda r: 100 * net(r), stake, lambda r: r["event"])))
    lby = defaultdict(list)
    for r in [x for x in late if x["filled"]]:
        lby[r["event"]].append(r)
    worst = sorted((sum(net(r) for r in g), e) for e, g in lby.items())[:8]
    print("  late filled: worst events (sum maker net c):", [(e, round(s)) for s, e in worst])
    lf = [x for x in late if x["filled"]]
    for k in (1, 3, 5):
        drop = {e for _, e in worst[:k]}
        keep = [r for r in lf if r["event"] not in drop]
        print(f"   late filled mean without {k} worst events: {np.mean([net(r) for r in keep]):+.3f}c")


def elect():
    rows = json.loads((P / "favorites/elections-entries.json").read_text())
    net = lambda r: r["net_cents"]
    stake = lambda r: r["price"] + r["fee_cents"]
    print("\nELECTIONS SUPPLEMENT (elections-entries.json; hindsight clock = actual close)")
    print("  series:", Counter(r["series"] for r in rows), "losses", sum(r["payout"] < .5 for r in rows),
          "mean price", round(float(np.mean([r["price"] for r in rows])), 2))
    line("all rules", rows, net, stake)
    e, l = split60(rows)
    line("early60", e, net, stake)
    line("late40", l, net, stake)
    print("  dates:", sorted({str(datetime.fromtimestamp(r['ref_ts'], timezone.utc).date()) for r in rows}))


def far():
    rows = json.loads((P / "far-strikes/rows.json").read_text())
    print("\nFAR STRIKES (rows.json). taker net = 100*win - ask - 7p(1-p); maker = filled rows, 100*win - rest, no fee")
    print("  family x ref rows:", sorted(Counter((r["family"], r["ref"]) for r in rows).items()))
    fee = lambda p: 7 * (p / 100) * (1 - p / 100)
    tnet = lambda r: 100 * r["win"] - r["ask"] - fee(r["ask"])
    mnet = lambda r: 100 * r["win"] - r["rest"]
    wk = lambda r: "-".join(map(str, datetime.strptime(r["day"], "%Y-%m-%d").isocalendar()[:2]))
    month = lambda r: r["day"][:7]

    def show(name, sub):
        if len(sub) < 5:
            print(f"  {name:46} n={len(sub)}")
            return
        days = sorted({r["day"] for r in sub})
        half = set(days[:len(days) // 2])
        a = [r for r in sub if r["day"] in half]
        b = [r for r in sub if r["day"] not in half]
        stake = lambda r: r["ask"] + fee(r["ask"])
        pd_ = boot(sub, lambda r: 100 * tnet(r), stake, lambda r: r["day"])
        pw = boot(sub, lambda r: 100 * tnet(r), stake, wk)
        pm = boot(sub, lambda r: 100 * tnet(r), stake, month)
        ha = sum(tnet(r) for r in a) / sum(stake(r) for r in a) * 100 if a else float("nan")
        hb = sum(tnet(r) for r in b) / sum(stake(r) for r in b) * 100 if b else float("nan")
        f = [r for r in sub if r.get("filled") and r.get("rest") is not None]
        mk = boot(f, lambda r: 100 * mnet(r), lambda r: r["rest"], lambda r: r["day"]) if len(f) >= 5 else None
        mw = boot(f, lambda r: 100 * mnet(r), lambda r: r["rest"], wk) if len(f) >= 5 else None
        print(f"  {name:46} n={len(sub):5} days={len(days):4} losses={sum(1 - r['win'] for r in sub):3} ask {np.mean([r['ask'] for r in sub]):5.1f} | TAKER %/bet day {fmt(pd_)} week {fmt(pw)} month {fmt(pm)} "
              f"halves {ha:+.2f}/{hb:+.2f} | MAKER fills {len(f)} %/fill day {fmt(mk)} week {fmt(mw)}")

    for fam, ref in sorted({(r["family"], r["ref"]) for r in rows}):
        for side in ("NO", "YES"):
            for lo, hi in ((90, 99), (80, 90)):
                sub = [r for r in rows if r["family"] == fam and r["ref"] == ref and r["side"] == side and lo <= r["mid"] < hi
                       and r["ask"] <= 99]
                show(f"{fam[:24]} @{ref} {side} {lo}-{hi}", sub)
    gas = [r for r in rows if r["family"].startswith("Gas+diesel daily") and r["ref"] == "6h" and r["side"] == "NO" and 90 <= r["mid"] < 99 and r["ask"] <= 99]
    print("  gas NO 90-99 @6h by series:", Counter(r["series"] for r in gas), "archived:", Counter(r["archived"] for r in gas),
          "losses:", [(r["ticker"], r["day"]) for r in gas if not r["win"]])
    for s in ("KXAAAGASD", "KXDIESELD"):
        show(f"  {s} NO 90-99 @6h", [r for r in gas if r["series"] == s])
    show("  gas NO 90-99 @6h archived only", [r for r in gas if r["archived"]])
    show("  gas NO 90-99 @6h recent only", [r for r in gas if not r["archived"]])
    print("  gas NO 90-99 @6h next-hour volume median", np.median([r["next_hour_volume"] for r in gas]), "day volume median",
          np.median([r["day_volume"] for r in gas]), "rows/day", len(gas) / len({r["day"] for r in gas}))


def weather():
    rows = json.loads((P / "weather-state/trades.json").read_text())
    fee = lambda p: 7 * (p / 100) * (1 - p / 100)
    print("\nWEATHER STATE (trades.json): rule cells overlap (caps 90/95/98 nested, hours 15-18, drop 2/4), so pooled sets are de-duplicated")
    print("  cities:", Counter(r["city"] for r in rows), "date span", min(r["day"] for r in rows), max(r["day"] for r in rows))
    fams = sorted({r["rule"][0] for r in rows})
    for fam in fams:
        sub = [r for r in rows if r["rule"][0] == fam]
        # widest cell per hour: cap 98, loosest condition; one trade per (city, day, ticker, hour)
        seen, uniq = set(), []
        for r in sorted(sub, key=lambda r: (r["day"], str(r["rule"][1]))):
            k = (r["city"], r["day"], r["ticker"], r["rule"][1])
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        first, f1 = set(), []
        for r in uniq:                                     # first qualifying hour per city-day
            k = (r["city"], r["day"])
            if k not in first:
                first.add(k)
                f1.append(r)
        for name, part in ((f"{fam}: unique ticker-hours", uniq), (f"{fam}: first per city-day", f1),
                           (f"{fam}: first per city-day, price>=50", [r for r in f1 if r["price"] >= 50]),
                           (f"{fam}: first per city-day, price 50-89", [r for r in f1 if 50 <= r["price"] < 90]),
                           (f"{fam}: first per city-day, price>=90", [r for r in f1 if r["price"] >= 90]),
                           (f"{fam}: first per city-day, price<=5", [r for r in f1 if r["price"] <= 5])):
            if len(part) < 3:
                print(f"  {name:62} n={len(part)}")
                continue
            stake = lambda r: r["price"] + fee(r["price"])
            c = boot(part, lambda r: r["net"], lambda r: 1.0, lambda r: r["day"])
            p = boot(part, lambda r: 100 * r["net"], stake, lambda r: r["day"])
            days = sorted({r["day"] for r in part})
            half = set(days[:len(days) // 2])
            a = [r for r in part if r["day"] in half]
            b = [r for r in part if r["day"] not in half]
            ha = sum(r["net"] for r in a) / sum(stake(r) for r in a) * 100
            hb = sum(r["net"] for r in b) / sum(stake(r) for r in b) * 100
            print(f"  {name:62} n={len(part):4} days={len(days):3} lost={sum(not r['won'] for r in part):3} price {np.mean([r['price'] for r in part]):5.1f} "
                  f"| c/bet {fmt(c)} | %/bet day-cluster {fmt(p)} | halves % {ha:+.2f}/{hb:+.2f}")
        cheap = [r for r in uniq if r["price"] <= 5]
        print(f"   {fam}: unique rows at price<=5c: {len(cheap)} of {len(uniq)}; of those official high > observed max: "
              f"{sum(r['final'] > r['top'] for r in cheap)}, < observed max: {sum(r['final'] < r['top'] for r in cheap)}, won: {sum(r['won'] for r in cheap)}")


def crypto():
    print("\nCRYPTO FORWARD (data/raw/crypto-forward/*.jsonl)")
    rows = []
    for path in sorted((ROOT / "data/raw/crypto-forward").glob("*.jsonl")):
        for l in open(path):
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass
    print("  row types:", Counter(r["type"] for r in rows))
    obs = [r for r in rows if r["type"] == "obs"]
    print(f"  obs span hours {(obs[-1]['t'] - obs[0]['t']) / 3600:.2f}; events observed {len({r['ticker'] for r in obs})}")
    settled = {r["ticker"]: r["value"] for r in rows if r["type"] == "settle"}
    checks = {(r["event"], r["rule"]): r for r in rows if r["type"] == "recheck"}
    fee = lambda p: 7 * (p / 100) * (1 - p / 100)
    for e in [r for r in rows if r["type"] == "entry"]:
        v = settled.get(e["ticker"])
        pay = None if v is None else (v if e["side"] == "yes" else 1 - v)
        ck = checks.get((e["event"], e["rule"]))
        net = None if pay is None else 100 * pay - e["ask"] - fee(e["ask"])
        print(f"   {e['rule']:6} {e['ticker']:28} {e['side']:3} ask {e['ask']} size {e['ask_size']} z {e['z'] and round(e['z'], 2)} "
              f"filled2s {ck and ck['filled']} payout {pay} net {net if net is None else round(net, 2)}")


if __name__ == "__main__":
    want = sys.argv[1:] or ["fav", "rest", "elect", "far", "weather", "crypto"]
    for name in want:
        globals()[name]()
