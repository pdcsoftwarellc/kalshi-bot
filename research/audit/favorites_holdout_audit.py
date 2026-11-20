"""Score the two pre-registered rules (data/processed/audit/inventory/holdout-prereg.txt) on the part of
research/favorites.py's seeded sample that the original run never fetched.

  bars    read from data/processed/audit/inventory/favorites-holdout/bars (favorites_holdout_download.py)
  trades  `--trades SECONDS` downloads the public prints the resting-bid test needs (one request every 3 s,
          60-300 s back-off on HTTP 429), cached under favorites-holdout/trades
  score   decisions(), one_per_event(), split() and resting() are the original's functions, unmodified
`--original` scores the original 939 markets instead, as a check on this script.
"""
import json, math, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import favorites as f, favorites_report as fr

HOLD = ROOT/"data/processed/audit/inventory/favorites-holdout"
OUT = ROOT/"data/processed/audit/inventory/favorites-holdout-audit.json"
BASE = "https://api.elections.kalshi.com/trade-api/v2"
GAP = 3.0
RULES = {"(a) pooled 90-99c, one day before close": lambda r: r["days"] == 1,
         "(b) 90-94c, one day before close": lambda r: r["days"] == 1 and r["band"] == "90-94"}


def build(original):
    series = {s["ticker"]: s for s in fr.series_list(None)}
    markets = f.references(f.load_universe())
    picked = f.sample(markets)
    rows, got = [], []
    for m in picked:
        if original:
            bars = fr.bars_for(None, m, False)
        else:
            p = HOLD/"bars"/(m["ticker"]+".json")
            bars = json.loads(p.read_text()) if p.exists() else None
        if bars:
            got.append(m)
            rows += fr.decisions(m, bars, series[m["series"]])
    return markets, picked, got, rows


def cost(r):
    return r["price"]+r["fee_cents"]


def pct(rows, net, c):
    return 100*sum(r[net] for r in rows)/sum(c(r) for r in rows)


def boot(rows, net, c, draws=3000):
    by = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        by[r["event"]][0] += r[net]
        by[r["event"]][1] += c(r)
    g = list(by.values())
    if len(g) < 10:
        return None
    rng = random.Random(11)
    out = sorted(100*sum(a for a, _ in s)/sum(b for _, b in s) for s in (rng.choices(g, k=len(g)) for _ in range(draws)))
    return [round(out[int(.025*draws)], 2), round(out[int(.975*draws)], 2)]


def describe(rows, net="net_cents", c=cost, price="price"):
    if not rows:
        return dict(bets=0)
    order = sorted(rows, key=lambda r: (r["ref_ts"], r["event"]))
    h = len(order)//2
    a, b = fr.split(rows)
    one = lambda rs: dict(bets=len(rs), pct=round(pct(rs, net, c), 2), ci=boot(rs, net, c)) if rs else None
    return dict(bets=len(rows), events=len({r["event"] for r in rows}), wins=sum(r["payout"] == 1 for r in rows),
                full_losses=sum(r["payout"] < .5 for r in rows), fractional=sum(0 < r["payout"] < 1 for r in rows),
                mean_price=round(mean(r[price] for r in rows), 2), pct=round(pct(rows, net, c), 2), ci=boot(rows, net, c),
                first_half=one(order[:h]), second_half=one(order[h:]), early60=one(a), late40=one(b),
                by_category={k: one([r for r in rows if r["category"] == k]) for k in sorted({r["category"] for r in rows})},
                median_volume_24h=round(median(r["volume_24h"] for r in rows)),
                volume_24h_quartiles=[round(q) for q in (sorted(r["volume_24h"] for r in rows)[len(rows)//4], sorted(r["volume_24h"] for r in rows)[3*len(rows)//4])],
                sd_pct=round(pstdev([100*r[net]/c(r) for r in rows]), 1),
                first=time.strftime("%Y-%m-%d", time.gmtime(min(r["ref_ts"] for r in rows))),
                last=time.strftime("%Y-%m-%d", time.gmtime(max(r["ref_ts"] for r in rows))))


def show(name, s):
    if not s.get("bets"):
        print(f"  {name}: no bets")
        return
    o = lambda x: "none" if not x else f"{x['pct']:+.2f}% {x['ci']} (n={x['bets']})"
    print(f"  {name}\n    bets {s['bets']}, events {s['events']}, wins {s['wins']}, full losses {s['full_losses']}, fractional {s['fractional']},"
          f" mean price {s['mean_price']}c, per-bet sd {s['sd_pct']}%, {s['first']} to {s['last']}\n"
          f"    return {s['pct']:+.2f}% per bet, 95% interval {s['ci']}\n"
          f"    equal halves: {o(s['first_half'])} / {o(s['second_half'])}    60/40: {o(s['early60'])} / {o(s['late40'])}\n"
          f"    prior-24h volume: median {s['median_volume_24h']} contracts, quartiles {s['volume_24h_quartiles']}")
    for k, v in s["by_category"].items():
        print(f"      {k:26} {o(v)}")


def binom_tail(wins, n, p):
    return sum(math.comb(n, k)*p**k*(1-p)**(n-k) for k in range(wins, n+1))


def need(sd, edge):
    return math.ceil((2.8*sd/edge)**2)


def trade_file(row):
    return HOLD/"trades"/f"{row['ticker']}-{int(row['t'])}.json"


def get_trades(budget, orders):
    import httpx
    (HOLD/"trades").mkdir(parents=True, exist_ok=True)
    stop, last, limited, streak, n = time.time()+budget, 0.0, None, 0, Counter()
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        for row in orders:
            out = trade_file(row)
            if out.exists():
                continue
            if time.time() > stop:
                break
            start, end = row["t"], min(row["t"]+fr.DAY, row["close_ts"])
            tiers = (["/historical/trades"] if start < fr.CUTOFF else [])+(["/markets/trades"] if end >= fr.CUTOFF else [])
            got, truncated, failed = [], False, False
            for path in tiers:
                cursor = ""
                for page in range(8):
                    params = dict(ticker=row["ticker"], min_ts=int(start), max_ts=int(end), limit=1000)
                    if cursor:
                        params["cursor"] = cursor
                    data = None
                    for attempt in range(3):
                        wait = last+GAP-time.time()
                        if wait > 0:
                            time.sleep(wait)
                        last = time.time()
                        try:
                            r = client.get(BASE+path, params=params)
                        except httpx.RequestError:
                            time.sleep(30)
                            continue
                        if r.status_code == 429:
                            n["http429"] += 1
                            limited = limited or time.time()
                            if time.time()-limited > 900:
                                print("STOP: rate-limit responses for more than 15 minutes", dict(n), flush=True)
                                return
                            pause = min(60*(streak+1), 300)
                            streak += 1
                            print(f"429: waiting {pause}s", flush=True)
                            time.sleep(pause)
                            continue
                        limited, streak = None, 0
                        if r.status_code == 200:
                            data = fr.slim_trades(r.json())
                        elif r.status_code < 500:
                            data = dict(cursor=None, trades=[])
                        else:
                            time.sleep(30)
                            continue
                        break
                    if data is None:
                        failed = True
                        break
                    got += data["trades"]
                    cursor = data.get("cursor")
                    if not cursor or not data["trades"]:
                        break
                else:
                    truncated = True
            if failed:
                n["failed"] += 1
                continue
            out.write_text(json.dumps(dict(truncated=truncated, trades=got), separators=(",", ":")))
            n["done"] += 1
            if n["done"] % 50 == 0:
                print(dict(n), flush=True)
    print("trades pass finished", dict(n), "still missing", sum(not trade_file(r).exists() for r in orders), flush=True)


def main():
    original = "--original" in sys.argv
    markets, picked, got, rows = build(original)
    taken = fr.one_per_event([r for r in rows if r["net_cents"] is not None])
    orders = [r for r in fr.one_per_event(rows) if r["days"] == 1]          # resting orders: the choice at T, before the delay
    if "--trades" in sys.argv:
        get_trades(float(sys.argv[sys.argv.index("--trades")+1]), orders)
        return
    months = (f.END-f.START).days/30.44
    scale = {}
    for s in {m["series"] for m in got}:
        allev = {m["event_ticker"] for m in markets if m["series"] == s and f.usable(m)}
        gotev = {m["event_ticker"] for m in got if m["series"] == s}
        scale[s] = len(allev)/len(gotev)
    print(f"{'ORIGINAL 939 (check)' if original else 'HOLDOUT'}: markets with bars {len(got)}, events {len({m['event_ticker'] for m in got})},"
          f" decisions {len(rows)}, entries {len(taken)}")
    out = dict(markets=len(got))
    if original:
        rest_rows = [r for r in json.loads((ROOT/'data/processed/favorites/resting.json').read_text()) if r["days"] == 1]
    else:
        rest_rows, missing = [], 0
        for r in orders:
            p = trade_file(r)
            if not p.exists():
                missing += 1
                continue
            d = json.loads(p.read_text())
            start, end = r["t"], min(r["t"]+fr.DAY, r["close_ts"])
            tr = [t for t in d["trades"] if start < t[0] <= end and t[1] is not None]
            rest_rows.append(dict(r, truncated=d["truncated"], **fr.resting(r, tr)))
        print(f"resting orders {len(orders)}, with a trade record {len(rest_rows)}, missing {missing}")
    mcost = lambda r: r["bid"]+(100*r["payout"]-r["bid"]-r["maker_net"])
    for name, keep in RULES.items():
        print(f"\n{name}")
        bets = [r for r in taken if keep(r)]
        s = describe(bets)
        show("TAKER: ask one hour after the snapshot plus the taker fee", s)
        if bets:
            ev = {r["event"]: r for r in bets}
            s["bets_per_month_scaled"] = round(sum(scale[r["series"]] for r in ev.values())/months, 1)
            s["bets_per_month_in_sample"] = round(len(bets)/months, 1)
            be = mean(cost(r) for r in bets)/100
            wins = sum(r["payout"] == 1 for r in bets)
            s["breakeven_win_rate"], s["binomial_p_one_sided"] = round(be, 4), binom_tail(wins, len(bets), be)
            s["need_for_3pct"], s["need_for_1p5pct"] = need(s["sd_pct"], 3), need(s["sd_pct"], 1.5)
            print(f"    events with a bet per month, scaled to every eligible event: {s['bets_per_month_scaled']} (in this sample: {s['bets_per_month_in_sample']})\n"
                  f"    break-even win rate {100*be:.2f}%; won {wins} of {len(bets)}; exact one-sided binomial p = {s['binomial_p_one_sided']:.4f}\n"
                  f"    bets for 80% power: {s['need_for_3pct']} to detect +3%, {s['need_for_1p5pct']} to detect +1.5%")
        placed = [r for r in rest_rows if keep(r)]
        fills = [r for r in placed if r["filled"]]
        m = describe(fills, "maker_net", mcost, "bid")
        show(f"RESTING: bid 1c under the ask for 24h, strict print-through; {len(fills)} fills of {len(placed)} orders"
             f" ({100*len(fills)/max(len(placed), 1):.0f}%), {sum(r['prints'] == 0 for r in placed)} orders saw no print", m)
        if placed:
            unf = [r for r in placed if not r["filled"]]
            m["orders"], m["losers_filled"], m["losers_unfilled"] = len(placed), sum(r["payout"] < .5 for r in fills), sum(r["payout"] < .5 for r in unf)
            print(f"    full losses: filled {m['losers_filled']} of {len(fills)}, unfilled {m['losers_unfilled']} of {len(unf)}")
        out[name] = dict(taker=s, resting=m)
    if not original:
        OUT.write_text(json.dumps(out, indent=1, default=str))
        print("\nsaved", OUT)


if __name__ == "__main__":
    main()
