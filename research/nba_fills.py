"""Check whether the NBA late-game favourite rule could really have been filled.

The plain rule (4th quarter or overtime, leader's ask 97-99c, buy at the next
minute's opening ask, hold to settlement) was tested on one-minute quote bars.
This script pulls Kalshi's public trade tape around every entry and asks:

1. how many contracts printed at our price or better in the entry minute and
   the two minutes after (taker bought our side = an offer we could have hit);
2. what is left of the edge if we pay 1c more, or only count entries where at
   least 100 contracts traded at our price or better;
3. whether a bid resting 1c below the ask for 3 minutes would have filled
   (a taker sold our side strictly through the bid) and what those fills earn.

Entries and the 60/40 chronological split come unchanged from rule_search.py.
The leader's side (YES or NO of the canonical ticker) is recovered from the
row's `won` flag and the settlement, which is only bookkeeping: the side was
known at the time from the score. Public reads only, two requests at a time.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean, median
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fetch, pages, save, timestamp
from research.game_state import kalshi_games
from research.rule_search import cushion, entries, interval

DATA = ROOT / "data/processed/game-state"
OUT = ROOT / "data/processed/nba-fills"
PLAIN = (97, 99, 1, 2880, 0, 0)
BEFORE, AFTER = 60, 120            # the entry minute is [ts-60, ts); two more minutes follow
REST = 180                         # a resting bid lives for three minutes from the entry time


def fee(price):
    return 100*.07*price/100*(1-price/100)


def split():
    rows = json.loads((DATA/"nba-minutes.json").read_text())
    by_game = {}
    for r in sorted(rows, key=lambda r: r["ts"]):
        r["cushion"] = cushion("nba", r)
        by_game.setdefault(r["event"], []).append(r)
    games = sorted(by_game.values(), key=lambda g: g[0]["ts"])
    cut = int(len(games)*.6)
    return ([dict(h, part="earlier") for h in entries("nba", games[:cut], PLAIN)]
            + [dict(h, part="later") for h in entries("nba", games[cut:], PLAIN)])


def tape(client, ticker, start, end, cutoff):
    path = "/historical/trades" if start < cutoff else "/markets/trades"
    rows = []
    for t in pages(client, path, "trades", ticker=ticker, min_ts=start, max_ts=end, limit=1000):
        when = timestamp(t["created_time"])
        if start <= when < end and not t.get("is_block_trade"):
            rows.append((when, t["taker_side"], round(100*float(t["yes_price_dollars"])), float(t["count_fp"])))
    return sorted(rows)


def measure(hit, trades, side):
    """Trade-tape facts for one entry. `side` is what we buy on this ticker; prices are in that side's cents."""
    ask, start = hit["ask"], hit["ts"]-BEFORE
    mine = [(when, taker == side, yes if side == "yes" else 100-yes, size) for when, taker, yes, size in trades]
    out = dict(total=sum(s for *_, s in mine),
               at_or_below=sum(s for _, _, p, s in mine if p <= ask),
               takeable=sum(s for _, buy, p, s in mine if buy and p <= ask),
               takeable_first_minute=sum(s for w, buy, p, s in mine if buy and p <= ask and w < hit["ts"]),
               takeable_plus1=sum(s for _, buy, p, s in mine if buy and p <= ask+1),
               total_first_minute=sum(s for w, *_, s in mine if w < hit["ts"]))
    # A bid one cent under the ask fills only when a taker sells our side strictly below it.
    bid = ask-1
    through = [(w, s) for w, buy, p, s in mine if not buy and p < bid and w < start+REST]
    touched = [(w, s) for w, buy, p, s in mine if not buy and p <= bid and w < start+REST]
    out.update(rest_through=sum(s for _, s in through), rest_touched=sum(s for _, s in touched),
               rest_wait=through[0][0]-start if through else None)
    for secs in (10, 30):
        out[f"takeable_{secs}s"] = sum(s for w, buy, p, s in mine if buy and p <= ask and w < start+secs)
    for size in (100, 1000):
        # Seconds until takers had bought `size` contracts at our price or better, and what the
        # first `size` contracts bought by takers on our side cost at whatever price they paid.
        got, wait, need, paid = 0.0, None, float(size), 0.0
        for w, buy, p, s in mine:
            if not buy:
                continue
            if p <= ask and wait is None:
                got += s
                if got >= size:
                    wait = w-start
            if need > 0:
                take = min(s, need)
                paid, need = paid+take*p, need-take
        out[f"wait_{size}"] = wait
        out[f"first{size}_price"] = paid/size if need <= 0 else None
    return out


def collect(part_order):
    hits = split()
    hits.sort(key=lambda h: (part_order.index(h["part"]), h["ts"]))
    with httpx.Client(timeout=45) as client:
        cutoff = timestamp(fetch(client, "/historical/cutoff", {})["trades_created_ts"])
        games = {g["event"]: g for g in kalshi_games(client, "KXNBAGAME")}

        def one(hit):
            g = games[hit["event"]]
            leader_yes = hit["won"] == (g["outcome"] == 1)
            start, end = hit["ts"]-BEFORE, hit["ts"]+AFTER
            row = dict(hit, ticker=g["ticker"], side="yes" if leader_yes else "no")
            try:
                row["main"] = measure(hit, tape(client, g["ticker"], start, end, cutoff), row["side"])
            except (RuntimeError, KeyError, httpx.HTTPError) as exc:
                row["error"] = type(exc).__name__
            return row
        done = []
        with ThreadPoolExecutor(2) as pool:           # the live recorder shares this API: two requests at a time
            for i, row in enumerate(pool.map(one, hits), 1):
                done.append(row)
                if i % 100 == 0:
                    print(f"{i}/{len(hits)} entries", flush=True)
    save(OUT/"entries.json", done)
    return done


def stats(trades):
    """trades: list of (price paid in cents, won, net cents)."""
    if not trades:
        return dict(trades=0)
    price = mean(t[0] for t in trades)/100
    nets = [dict(net=t[2]) for t in trades]
    return dict(trades=len(trades), losses=sum(not t[1] for t in trades), win_rate=mean(t[1] for t in trades),
                mean_price=100*price, net_cents=mean(t[2] for t in trades), total_cents=sum(t[2] for t in trades),
                ci95=interval(nets))


def needed(s, taker=True):
    p = s["mean_price"]/100
    return p+(.07*p*(1-p) if taker else 0)


def buy(row, price, taker=True):
    return (price, row["won"], 100*row["won"]-price-(fee(price) if taker else 0))


def quantiles(values):
    v = sorted(values)
    pick = lambda q: v[min(len(v)-1, int(q*len(v)))]
    return dict(n=len(v), p10=pick(.10), p25=pick(.25), median=median(v), p75=pick(.75), mean=mean(v),
                **{f"share_ge_{k}": mean(x >= k for x in v) for k in (1, 10, 100, 1000)})


def report(rows):
    def m(r, key):
        return r["main"][key]
    out = {}
    for part in ("earlier", "later"):
        part_rows = [r for r in rows if r["part"] == part and "main" in r]
        if not part_rows:
            continue
        o = out[part] = dict(entries=len(part_rows), errors=sum(r["part"] == part and "error" in r for r in rows))
        o["liquidity"] = {k: quantiles([m(r, k) for r in part_rows])
                          for k in ("takeable", "takeable_first_minute", "takeable_30s", "takeable_10s",
                                    "at_or_below", "total", "takeable_plus1")}
        o["wait_seconds"] = {size: dict(reached=mean(m(r, f"wait_{size}") is not None for r in part_rows),
                                        within_10s=mean((m(r, f"wait_{size}") or 999) <= 10 for r in part_rows),
                                        within_60s=mean((m(r, f"wait_{size}") or 999) <= 60 for r in part_rows),
                                        median=median([m(r, f"wait_{size}") for r in part_rows if m(r, f"wait_{size}") is not None]))
                             for size in (100, 1000)}
        o["liquidity_by_ask"] = {a: quantiles([m(r, "takeable") for r in part_rows if r["ask"] == a])
                                 for a in (97, 98, 99) if any(r["ask"] == a for r in part_rows)}
        v = o["variants"] = {}
        v["bar open ask (backtest)"] = stats([buy(r, r["ask"]) for r in part_rows])
        v["pay 1c more"] = stats([buy(r, r["ask"]+1) for r in part_rows])
        v["pay 1c more, skip 99c entries"] = stats([buy(r, r["ask"]+1) for r in part_rows if r["ask"] < 99])
        for k in (10, 100, 1000):
            v[f">= {k} takeable at our price"] = stats([buy(r, r["ask"]) for r in part_rows if m(r, "takeable") >= k])
            v[f"fewer than {k} takeable (the rest)"] = stats([buy(r, r["ask"]) for r in part_rows if m(r, "takeable") < k])
        v[">= 100 takeable in first minute"] = stats([buy(r, r["ask"]) for r in part_rows if m(r, "takeable_first_minute") >= 100])
        v[">= 100 takeable at our price + 1c, pay 1c more"] = stats(
            [buy(r, r["ask"]+1) for r in part_rows if m(r, "takeable_plus1") >= 100])
        o["slippage_cents"] = {}
        for size in (100, 1000):
            got = [r for r in part_rows if m(r, f"first{size}_price") is not None]
            v[f"first {size} taker-bought contracts, price actually paid"] = stats(
                [buy(r, m(r, f"first{size}_price")) for r in got])
            o["slippage_cents"][size] = quantiles([m(r, f"first{size}_price")-r["ask"] for r in got])
        for k in (100, 1000):
            v[f">= {k} takeable within 10 seconds"] = stats([buy(r, r["ask"]) for r in part_rows if m(r, "takeable_10s") >= k])
        filled = [r for r in part_rows if m(r, "rest_through") > 0]
        missed = [r for r in part_rows if m(r, "rest_through") == 0]
        o["resting_bid"] = dict(
            fill_rate=len(filled)/len(part_rows),
            fill_rate_100=mean(m(r, "rest_through") >= 100 for r in part_rows),
            touched_rate=mean(m(r, "rest_touched") > 0 for r in part_rows),
            median_wait_s=median(r["main"]["rest_wait"] for r in filled if r["main"]["rest_wait"] is not None)
            if any(r["main"]["rest_wait"] is not None for r in filled) else None,
            filled_maker_fee_zero=stats([buy(r, r["ask"]-1, taker=False) for r in filled]),
            filled_taker_fee=stats([buy(r, r["ask"]-1) for r in filled]),
            filled_100_maker_fee_zero=stats([buy(r, r["ask"]-1, taker=False) for r in part_rows if m(r, "rest_through") >= 100]),
            unfilled_if_bought_at_ask=stats([buy(r, r["ask"]) for r in missed]),
            filled_if_bought_at_ask=stats([buy(r, r["ask"]) for r in filled]),
            per_entry_maker_fee_zero=sum(buy(r, r["ask"]-1, taker=False)[2] for r in filled)/len(part_rows),
            per_entry_taker_rule=mean(buy(r, r["ask"])[2] for r in part_rows))
    return out


def show(rep, title):
    print(f"\n=== {title} ===")
    for part, o in rep.items():
        print(f'\n{part}: {o["entries"]} entries ({o["errors"]} fetch errors)')
        for k, q in o["liquidity"].items():
            print(f'  {k:22} p10 {q["p10"]:8.0f} p25 {q["p25"]:8.0f} median {q["median"]:8.0f} p75 {q["p75"]:9.0f} | '
                  f'>=1 {q["share_ge_1"]:.3f} >=10 {q["share_ge_10"]:.3f} >=100 {q["share_ge_100"]:.3f} >=1000 {q["share_ge_1000"]:.3f}')
        for a, q in o["liquidity_by_ask"].items():
            print(f'  takeable at ask {a}: n {q["n"]:3} p25 {q["p25"]:7.0f} median {q["median"]:7.0f} '
                  f'>=100 {q["share_ge_100"]:.3f} >=1000 {q["share_ge_1000"]:.3f}')

        def line(tag, s, taker=True):
            if not s["trades"]:
                print(f"  {tag}: no trades")
                return
            ci = s["ci95"] and [round(x, 2) for x in s["ci95"]]
            print(f'  {tag:55} n {s["trades"]:4} losses {s["losses"]:2} win {s["win_rate"]:.4f} need {needed(s, taker):.4f} '
                  f'price {s["mean_price"]:.2f} net {s["net_cents"]:+.2f}c ci {ci}')
        for tag, s in o["variants"].items():
            line(tag, s)
        for size, q in o["slippage_cents"].items():
            print(f'  first-{size} price minus bar ask: median {q["median"]:+.2f}c mean {q["mean"]:+.2f}c p75 {q["p75"]:+.2f}c (n {q["n"]})')
        for size, w in o["wait_seconds"].items():
            print(f'  seconds until {size} taken at our price or better: reached {w["reached"]:.3f} within 10s {w["within_10s"]:.3f} '
                  f'within 60s {w["within_60s"]:.3f} median {w["median"]:.1f}s')
        b = o["resting_bid"]
        print(f'  resting bid 1c under: fill rate {b["fill_rate"]:.3f} (>=100 through: {b["fill_rate_100"]:.3f}; '
              f'touched {b["touched_rate"]:.3f}), median wait {b["median_wait_s"]}s')
        line("  filled, maker fee 0", b["filled_maker_fee_zero"], taker=False)
        line("  filled, full taker fee", b["filled_taker_fee"])
        line("  filled >=100 through, maker fee 0", b["filled_100_maker_fee_zero"], taker=False)
        line("  filled subset if it had paid the ask", b["filled_if_bought_at_ask"])
        line("  unfilled subset if it had paid the ask", b["unfilled_if_bought_at_ask"])
        print(f'  profit per rule entry: resting bid {b["per_entry_maker_fee_zero"]:+.2f}c vs paying the ask {b["per_entry_taker_rule"]:+.2f}c')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earlier-first", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="reuse data/processed/nba-fills/entries.json")
    args = parser.parse_args()
    if args.report_only:
        rows = json.loads((OUT/"entries.json").read_text())
    else:
        rows = collect(["earlier", "later"] if args.earlier_first else ["later", "earlier"])
    summary = dict(canonical=report(rows))
    show(summary["canonical"], "canonical ticker only")
    save(OUT/"summary.json", summary)


if __name__ == "__main__":
    main()
