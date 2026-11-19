"""How many dollars can the favourite-dip bid absorb? Contracts printed through the bid, by sport, level and month.

The rule: a favourite priced 90-99c before play gets a resting bid (80c in the
traded version) once the game is live, held to settlement. A bid is filled only
when a trade prints STRICTLY below it (paper/rules.py `prints`). This script
measures the size behind those fills, which the earlier studies did not: how
many contracts printed through each of a ladder of bids (85, 80, 75, 70, 65,
60c), whether the big dips are the bad ones, and what a fixed bid of Q
contracts a game would have deployed, earned and reserved.

Orders (nothing re-derived; the same orders as the earlier backtests)
  men     men's college basketball, KXNCAAMBGAME: data/processed/jit-bids/orders.pkl, group CBB
  tennis  first-round qualifying on every tour: the same file, group FRQ
  women   women's college basketball, KXNCAAWBGAME: research/wcbb_backfill.py, view "tip", 90-99c

Size through a bid, per game and level
  * One-minute bars (cached) give every in-play minute's low, high and volume.
    A minute whose low and high have no ladder level between them is exact:
    all of its volume is below a level or none is.
  * Minutes that straddle a level are read from the public trade record
    (GET /markets/trades, /historical/trades before 2026-09-18): the windows
    already cached by research/jit_bids.py around the 80c fills, plus new
    reads cached under data/raw/dip-capacity/ (one file per market).
  * A straddling minute whose trades were not read is carried as a range: none
    of its volume through the straddled levels (low) or all of it (high).
  * "through" = contracts printed strictly below the level from the start of
    play to settlement (what `prints` calls size_through). "5m"/"30m" = the
    same within 5/30 minutes of the first such print. "band" = contracts
    printed below the level but not below the next level down: with the whole
    ladder resting, a contract can fill only one of our bids.

Sizing: a bid of Q contracts is filled for min(Q, SHARE x through), SHARE =
25%. Money put in = contracts x level. Capital reserved = Q x level for every
90-99c favourite, filled or not, from its start to its settlement.

    .venv/bin/python research/dip_capacity.py download --minutes 9      (repeat until nothing is left)
    .venv/bin/python research/dip_capacity.py report

Public market data only, one request at a time, PACE seconds apart, long
pauses after HTTP 429 (live programs share the limit). No orders, no
authentication. Output: data/processed/dip-capacity/.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, midnight, timestamp
from research.near_end import cents
from research import jit_bids as jit

RAW = ROOT / "data/raw/dip-capacity"
OUT = ROOT / "data/processed/dip-capacity"
LEVELS = (85, 80, 75, 70, 65, 60)
SHARE, PACE, GAP = 0.25, 2.0, 300
CUTOFF = midnight(date(2026, 9, 18))
EASTERN = ZoneInfo("America/New_York")
GROUPS = (("men", "men's college basketball"), ("women", "women's college basketball"), ("tennis", "tennis first-round qualifying"))
SIZES = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 10**9)
RNG = np.random.default_rng(7)
last_request = [0.0]
buffer = io.StringIO()


def say(text=""):
    print(text, flush=True)
    buffer.write(text+"\n")


# ---------------------------------------------------------------- orders

def bars_of(file, yes, start):
    out = []
    for b in sorted(json.loads(file.read_text()).get("candlesticks") or [], key=lambda b: b["end_period_ts"]):
        lo, hi = cents(b, "price", "low"), cents(b, "price", "high")
        if lo is None or hi is None or b["end_period_ts"] <= start+60:
            continue
        if not yes:
            lo, hi = 100-hi, 100-lo
        out.append((b["end_period_ts"], lo, hi, float(b.get("volume_fp") or b.get("volume") or 0)))
    return out


def orders():
    """One dict per 90-99c favourite: group, event, ticker, yes, start, settle, payout, price, in-play bars
    [(end of minute, low, high, contracts)] in the favourite's price, and trade windows already cached by jit_bids."""
    out = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404))) as offline:
        for r in jit.load():
            if r["group"] not in ("CBB", "FRQ"):
                continue
            o = dict(group="men" if r["group"] == "CBB" else "tennis", event=r["event"], ticker=r["ticker"], yes=r["yes"], start=r["start"],
                     settle=r["settle"], payout=r["payout"], price=r["mid"], series=r["series"],
                     bars=bars_of(jit.bar_file(r["series"], r["ticker"], r["settle"]), r["yes"], r["start"]), cached=[])
            if r["fill_ts"]:
                lo, hi = jit.windows(r)
                raw = jit.trade_pages(offline, r["ticker"], r["settle"], lo, hi, False, 0)
                if raw:
                    o["cached"].append((lo, hi, [t for t in jit.parse(r, raw) if lo < t[0] <= hi]))
            out.append(o)
    from research import wcbb_backfill as w
    for v in w.band(w.view(w.load(), "tip"), 90, 99):
        out.append(dict(group="women", event=v["event"], ticker=v["ticker"], yes=v["yes"], start=v["est_start"], settle=v["settle"],
                        payout=v["payout"], price=v["price"], series="KXNCAAWBGAME", cached=[],
                        bars=[(x[0], x[1], x[2], x[3]) for x in v["plays"] if x[0] > v["est_start"]+60]))
    for o in out:
        o["day"] = (datetime.fromtimestamp(o["start"], EASTERN)-timedelta(hours=5)).strftime("%Y-%m-%d")
        o["month"] = o["day"][:7]
    return sorted(out, key=lambda o: (o["settle"], o["event"]))


def straddles(bar):
    return any(bar[1] < level <= bar[2] for level in LEVELS)


def covered(o, bar, spans=()):
    return any(lo <= bar[0]-60 and bar[0] <= hi for lo, hi, _ in list(o["cached"])+list(spans))


def wanted(o):
    """Windows (lo, hi] still to read: runs of straddling minutes not inside a cached window, joined across gaps of GAP seconds."""
    need = [b for b in o["bars"] if b[1] < LEVELS[0] and straddles(b) and not covered(o, b)]
    spans = []
    for b in need:
        if spans and b[0]-spans[-1][1] <= GAP:
            spans[-1][1] = b[0]
        else:
            spans.append([b[0]-60, b[0]])
    return [(lo, hi, sum(b[3] for b in o["bars"] if lo < b[0] <= hi)) for lo, hi in spans]


# ---------------------------------------------------------------- download

def store(o):
    return RAW / (o["ticker"]+".json")


def read_store(o):
    return json.loads(store(o).read_text()) if store(o).exists() else {}


def get(client, path, params):
    for pause in (30, 60, 120, 240):
        wait = last_request[0]+PACE-time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params)
        except httpx.RequestError:
            resp = None
        last_request[0] = time.monotonic()
        if resp is not None and resp.status_code == 200:
            return resp.json()
        if resp is not None and resp.status_code != 429 and resp.status_code < 500:
            return dict(trades=[], error=resp.status_code)
        print(f"   {time.strftime('%H:%M:%S')} throttled ({resp.status_code if resp is not None else 'network'}): pausing {pause}s", flush=True)
        time.sleep(pause)
        last_request[0] = time.monotonic()
    raise RuntimeError(f"Read failed after long back-off: {path}")


def span_trades(client, o, lo, hi, count):
    """Every trade of one market in (lo, hi], as [time, favourite's price, contracts, seller-initiated]."""
    paths = ["/historical/trades", "/markets/trades"] if o["settle"] < CUTOFF else ["/markets/trades", "/historical/trades"]
    for path in paths:
        raw, cursor = [], ""
        while True:
            params = dict(ticker=o["ticker"], min_ts=int(lo), max_ts=int(hi)+1, limit=1000)
            if cursor:
                params["cursor"] = cursor
            data = get(client, path, params)
            count["requests"] += 1
            raw += data.get("trades") or []
            cursor = data.get("cursor")
            if not cursor or not data.get("trades"):
                break
        if raw:
            out = []
            for t in raw:
                price = round(100*float(t["yes_price_dollars"]))
                taker = t.get("taker_side") or t.get("taker_outcome_side")
                out.append([timestamp(t["created_time"]), price if o["yes"] else 100-price, float(t["count_fp"]), (taker == "no") if o["yes"] else (taker == "yes")])
            return sorted(t for t in out if lo < t[0] <= hi)
    return []


def download(minutes):
    RAW.mkdir(parents=True, exist_ok=True)
    todo = []
    for o in orders():
        have = read_store(o)
        todo += [(volume, o, lo, hi) for lo, hi, volume in wanted(o) if f"{lo}-{hi}" not in have]
    todo.sort(key=lambda x: (x[0], x[1]["ticker"], x[2]))          # small windows first: they are the dips that came back
    print(f"windows still to read {len(todo)}, contracts in them {sum(x[0] for x in todo):,.0f}", flush=True)
    deadline, count = time.monotonic()+60*minutes, Counter()
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        for volume, o, lo, hi in todo:
            if time.monotonic() > deadline:
                break
            trades = span_trades(client, o, lo, hi, count)
            have = read_store(o)
            have[f"{lo}-{hi}"] = trades
            temp = store(o).with_suffix(".tmp")
            temp.write_text(json.dumps(have))
            temp.replace(store(o))
            count["windows"] += 1
            count["empty"] += not trades
    left = sum(f"{lo}-{hi}" not in read_store(o) for o in orders() for lo, hi, _ in wanted(o))
    print(f"done: {dict(count)}; windows still to read {left}")


# ---------------------------------------------------------------- per-game measures

def prints(o):
    """In-play prints as (time, lowest possible price, highest possible price, contracts, seller-initiated or None), and
    a tally of how the volume was resolved (bar exact / trade record / straddling minute not read)."""
    have = read_store(o)
    spans = [(float(k.split("-")[0]), float(k.split("-")[1]), v) for k, v in have.items()]
    out, tally = [], Counter()
    for b in o["bars"]:
        if covered(o, b, spans):
            tally["bar volume replaced by trades"] += b[3]
            continue
        kind = "unread" if straddles(b) and b[1] < LEVELS[0] else "bar"
        tally[kind] += b[3]
        out.append((b[0], b[1], b[2], b[3], None))
    for lo, hi, trades in list(o["cached"])+spans:
        for t in trades:
            if t[0] > o["start"]:
                tally["trades"] += t[2]
                out.append((t[0], t[1], t[1], t[2], bool(t[3])))
    return sorted(out, key=lambda p: p[0]), tally


def measure(o):
    stream, tally = prints(o)
    o["tally"], o["levels"] = tally, {}
    lower = {level: nxt for level, nxt in zip(LEVELS, LEVELS[1:]+(0,))}
    for level in LEVELS:
        below = [p for p in stream if p[2] < level]            # certainly below
        maybe = [p for p in stream if p[1] < level]            # possibly below (an unread straddling minute)
        cell = dict(filled=bool(maybe), through=sum(p[3] for p in below), through_high=sum(p[3] for p in maybe))
        if maybe:
            t0 = maybe[0][0]
            cell.update(t0=t0, minutes=(t0-o["start"])/60,
                        m5=sum(p[3] for p in below if p[0] <= t0+300), m30=sum(p[3] for p in below if p[0] <= t0+1800),
                        band=sum(p[3] for p in below if p[2] >= lower[level]), band_high=sum(p[3] for p in maybe if p[1] >= lower[level]),
                        taped=sum(p[3] for p in below if p[4] is not None), sold=sum(p[3] for p in below if p[4]),
                        first=sum(p[3] for p in below if p[0] == t0), low=min(p[1] for p in maybe))
        o["levels"][level] = cell
    return o


# ---------------------------------------------------------------- statistics

def boot(values, weights=None, draws=4000):
    values = np.asarray(values, float)
    if len(values) < 10:
        return None
    pick = RNG.integers(0, len(values), (draws, len(values)))
    if weights is None:
        means = values[pick].mean(axis=1)
    else:
        weights = np.asarray(weights, float)
        means = (values[pick]*weights[pick]).sum(axis=1)/np.maximum(weights[pick].sum(axis=1), 1e-9)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def band(ci, scale=1.0, unit="c"):
    return "[  n<10   ]" if ci is None else f"[{scale*ci[0]:+5.1f},{scale*ci[1]:+5.1f}]{unit}"


def fills(part, level):
    return [o for o in part if o["levels"][level]["filled"]]


def line(part, level):
    """fills, win rate, net c/fill with interval, % return on the level."""
    if not part:
        return "fills   0"
    nets = [100*o["payout"]-level for o in part]
    text = f'fills {len(part):3} win {100*np.mean([o["payout"] for o in part]):5.1f}% net {np.mean(nets):+5.1f}c {band(boot(nets))} = {100*np.mean(nets)/level:+5.1f}%'
    return text+("  (too few)" if len(part) < 10 else "")


def halves(part):
    part = sorted(part, key=lambda o: (o["settle"], o["event"]))
    return part[:len(part)//2], part[len(part)//2:]


def pct(values, qs=(10, 25, 50, 75, 90)):
    return " ".join(f"{v:9,.0f}" for v in np.percentile(values, qs)) if len(values) else "n/a"


def sized(part, level, size, key="through", share=SHARE):
    """Contracts and net cents per contract for a bid of `size` contracts on each filled game."""
    got = np.array([min(size, share*o["levels"][level][key]) for o in part])
    net = np.array([100*o["payout"]-level for o in part])
    return got, net


def peak(part):
    """Largest number of orders resting at once, and the day on which it happens."""
    marks = sorted([(o["start"], 1, o["day"]) for o in part]+[(o["settle"], -1, o["day"]) for o in part], key=lambda m: (m[0], m[1]))
    best, now, day = 0, 0, None
    for _, step, d in marks:
        now += step
        if now > best:
            best, day = now, d
    return best, day


def days_in(part, month):
    days = sorted({o["day"] for o in part if o["month"] == month})
    return (date.fromisoformat(days[-1])-date.fromisoformat(days[0])).days+1 if days else 0


# ---------------------------------------------------------------- report

def report():
    rows = [measure(o) for o in orders()]
    OUT.mkdir(parents=True, exist_ok=True)
    say("=== COVERAGE of the size measurement (contracts in in-play minutes with a print under 85c)")
    for key, name in GROUPS:
        part = [o for o in rows if o["group"] == key]
        tally = sum((o["tally"] for o in part), Counter())
        left = sum(1 for o in part for lo, hi, _ in wanted(o) if f"{lo}-{hi}" not in read_store(o))
        say(f"  {name:32} orders {len(part):4} | from exact bars {tally['bar']:13,.0f} | from the trade record {tally['trades']:13,.0f} "
            f"(bars said {tally['bar volume replaced by trades']:13,.0f}) | straddling minutes not read {tally['unread']:11,.0f} in {left} windows")
    say("  'exact bars' counts every in-play minute outside the trade windows, including minutes wholly above 85c.")

    say("\n=== 0. THE BACKTEST BEING SIZED (80c bid, strict fills)")
    for key, name in GROUPS:
        part = [o for o in rows if o["group"] == key]
        a, b = halves(part)
        say(f"  {name:32} orders {len(part):4} {line(fills(part, 80), 80)}")
        say(f"  {'':32} first half  {line(fills(a, 80), 80)}")
        say(f"  {'':32} second half {line(fills(b, 80), 80)}")

    say("\n=== 1. CONTRACTS PRINTED STRICTLY THROUGH 80c, per filled game (percentiles 10 / 25 / 50 / 75 / 90, then the mean)")
    summary = {}
    for key, name in GROUPS:
        part = fills([o for o in rows if o["group"] == key], 80)
        cells = [o["levels"][80] for o in part]
        say(f"  {name} ({len(part)} fills)")
        for label, values in (("whole game (size_through)", [c["through"] for c in cells]), ("  upper bound incl. unread minutes", [c["through_high"] for c in cells]),
                              ("within 30 minutes of the first print", [c["m30"] for c in cells]), ("within 5 minutes of the first print", [c["m5"] for c in cells]),
                              ("at the very first print", [c["first"] for c in cells]),
                              ("whole game, favourite WON", [o["levels"][80]["through"] for o in part if o["payout"] > .5]),
                              ("whole game, favourite LOST", [o["levels"][80]["through"] for o in part if o["payout"] < .5]),
                              ("dollars through (whole game x 80c)", [.8*c["through"] for c in cells])):
            say(f"    {label:38} n {len(values):3} | {pct(values)} | mean {np.mean(values):11,.0f}")
        taped = sum(c["taped"] for c in cells)
        say(f"    seller-initiated share of the volume under 80c read from the trade record: {100*sum(c['sold'] for c in cells)/max(taped, 1):.0f}% "
            f"(trade record covers {100*taped/max(sum(c['through'] for c in cells), 1):.0f}% of the volume under 80c)")
        say(f"    games with fewer than 10 / 100 / 1,000 contracts through: {100*np.mean([c['through'] < 10 for c in cells]):.0f}% / "
            f"{100*np.mean([c['through'] < 100 for c in cells]):.0f}% / {100*np.mean([c['through'] < 1000 for c in cells]):.0f}%")
        for name_half, half in zip(("first half", "second half"), halves(part)):
            say(f"    {name_half:38} n {len(half):3} | {pct([o['levels'][80]['through'] for o in half])}")
        summary[key] = dict(fills=len(part), through_percentiles=[float(v) for v in np.percentile([c["through"] for c in cells], [10, 25, 50, 75, 90])])

    say("\n=== 2. DOES THE RETURN DEPEND ON HOW MUCH PRINTED THROUGH? 80c fills by tercile of size")
    say("    (size through the whole game is known only at the end and is large BECAUSE a loser keeps trading under 80c; the 5-minute size is the cleaner test)")
    for key, name in GROUPS:
        part = fills([o for o in rows if o["group"] == key], 80)
        for label, field in (("whole game", "through"), ("first 5 minutes", "m5")):
            values = np.array([o["levels"][80][field] for o in part])
            cuts = np.percentile(values, [100/3, 200/3])
            rank = np.argsort(np.argsort(values)).astype(float)
            wins = np.array([o["payout"] for o in part])
            rho = float(np.corrcoef(rank, wins)[0, 1]) if len(set(wins)) > 1 else float("nan")
            say(f"  {name}, by contracts through in the {label}: cuts {cuts[0]:,.0f} and {cuts[1]:,.0f}; rank correlation of size with winning {rho:+.2f} (noise +/-{2/len(part)**.5:.2f})")
            a, b = halves(part)
            for tercile, test in (("small ", lambda x: x <= cuts[0]), ("middle", lambda x: cuts[0] < x <= cuts[1]), ("large ", lambda x: x > cuts[1])):
                pick = lambda group: [o for o in group if test(o["levels"][80][field])]
                say(f"    {tercile} {line(pick(part), 80)}")
                say(f"           first half {line(pick(a), 80)} | second half {line(pick(b), 80)}")
    say("\n  2b. Return per CONTRACT for a bid of Q contracts at 80c, filled for min(Q, 25% of the contracts through in the whole game)")
    for key, name in GROUPS:
        part = fills([o for o in rows if o["group"] == key], 80)
        say(f"  {name}")
        for size in SIZES:
            got, net = sized(part, 80, size)
            text = f"    Q {'no cap' if size >= 10**9 else format(size, ','):>7}: mean contracts {got.mean():10,.0f} full {100*np.mean(got >= size):3.0f}% | net per contract {(got*net).sum()/got.sum():+6.1f}c {band(boot(net, got))} = {100*(got*net).sum()/got.sum()/80:+6.1f}%"
            for half in halves(part):
                g, n = sized(half, 80, size)
                text += f" | {(g*n).sum()/max(g.sum(), 1e-9):+6.1f}c"
            say(text+"  (halves)")

    say("\n=== 3. THE LADDER: one bid per level on every 90-99c favourite, strict fills")
    for key, name in GROUPS:
        part = [o for o in rows if o["group"] == key]
        say(f"  {name} ({len(part)} orders)")
        base = {o["event"] for o in fills(part, 80)}
        for level in LEVELS:
            hit = fills(part, level)
            a, b = halves(part)
            through = [o["levels"][level]["through"] for o in hit]
            bands = [o["levels"][level]["band"] for o in hit]
            say(f"    bid {level}c {line(hit, level)} | halves {np.mean([100*o['payout']-level for o in fills(a, level)]):+5.1f}c ({len(fills(a, level))}) / "
                f"{np.mean([100*o['payout']-level for o in fills(b, level)]):+5.1f}c ({len(fills(b, level))})")
            say(f"            contracts through: median {np.median(through):9,.0f} quartiles {np.percentile(through, 25):9,.0f}-{np.percentile(through, 75):9,.0f} total {sum(through):13,.0f} | "
                f"of which not through the next level down: median {np.median(bands):8,.0f} total {sum(bands):12,.0f} | games also filled at 80c: {sum(o['event'] in base for o in hit)}/{len(hit)}")
        only85 = [o for o in fills(part, 85) if not o["levels"][80]["filled"]]
        say(f"    games filled at 85c but never at 80c: {line(only85, 85)}")
        for lo_level in LEVELS[2:]:
            stop = [o for o in fills(part, 80) if not o["levels"][lo_level]["filled"]]
            say(f"    games filled at 80c that never printed under {lo_level}c: {line(stop, 80)}")
        say("    ladder as one book, Q contracts at every level, each contract counted once (25% of its band), against Q at 80c alone:")
        for size in (100, 1000, 10000):
            totals = {}
            for label, levels, field in (("80c alone", (80,), "through"), ("85+80", (85, 80), "band"), ("85..60", LEVELS, "band"), ("80..60", LEVELS[1:], "band")):
                money = earned = 0.0
                per_game = defaultdict(lambda: [0.0, 0.0])
                for level in levels:
                    last = level == levels[-1]
                    for o in fills(part, level):
                        got = min(size, SHARE*o["levels"][level]["through" if last or field == "through" else "band"])
                        per_game[o["event"]][0] += got*level/100
                        per_game[o["event"]][1] += got*(100*o["payout"]-level)/100
                money, earned = sum(v[0] for v in per_game.values()), sum(v[1] for v in per_game.values())
                ci = boot([v[1]/max(v[0], 1e-9) for v in per_game.values()], [v[0] for v in per_game.values()])
                totals[label] = f"{label}: in ${money:11,.0f} earned ${earned:+10,.0f} = {100*earned/max(money, 1e-9):+5.1f}% {band(ci, 100, '%')}"
            say(f"      Q {size:6,}: "+" | ".join(totals.values()))

    say("\n=== 4. DOLLARS PER MONTH: bid Q contracts at 80c on every 90-99c favourite, filled for min(Q, 25% of the contracts through)")
    say("    'in' = money paid for fills; 'earned' = realised at settlement; 'reserve' = Q x 80c x the most orders resting at once (every order rests from start to settlement)")
    say("  4a. Observed months, as sampled (men's 2025-26 season is a 57% sample of games; women's 2025-26 is 96%; November 2026 is complete to the 18th)")
    for key, name in GROUPS:
        part = [o for o in rows if o["group"] == key]
        say(f"  {name}")
        for month in sorted({o["month"] for o in part}):
            sub = [o for o in part if o["month"] == month]
            got = fills(sub, 80)
            top, day = peak(sub)
            text = f"    {month} days {days_in(part, month):2} orders {len(sub):3} fills {len(got):3} most resting at once {top:3} ({day})"
            for size in (100, 1000, 10**9):
                g, n = sized(got, 80, size) if got else (np.zeros(1), np.zeros(1))
                text += f" | Q {'no cap' if size >= 10**9 else format(size, ',')}: in ${.8*g.sum():11,.0f} earned ${(g*n).sum()/100:+10,.0f}"
            say(text)

    say("\n  4b. Per 30 days at the current frequency (college basketball: November 2026 scaled to 30 days; tennis: mean of May-October 2026),")
    say("      with dollars per fill taken from EVERY fill in the sample. 'at backtest' = money in x the group's return per fill; 'half' = half of that.")
    freq = {}
    for key, name in GROUPS:
        part = [o for o in rows if o["group"] == key]
        if key == "tennis":
            use = [m for m in sorted({o["month"] for o in part}) if "2026-05" <= m <= "2026-10"]
            scale = 1/len(use)
        else:
            use = ["2026-11"]
            scale = 30/days_in(part, "2026-11")
        sub = [o for o in part if o["month"] in use]
        freq[key] = dict(orders=len(sub)*scale, fills=len(fills(sub, 80))*scale, peak=max(peak([o for o in sub if o["month"] == m])[0] for m in use))
        hit = fills(part, 80)
        rate = np.mean([100*o["payout"]-80 for o in hit])/80
        f = freq[key]
        say(f"  {name}: {f['orders']:.0f} orders and {f['fills']:.0f} fills per 30 days, most resting at once {f['peak']}; return per fill {100*rate:+.1f}%")
        before = (0.0, 0.0)
        for size in SIZES:
            got, net = sized(hit, 80, size)
            dollars = got*net/100
            ci = boot(dollars)
            money, earned = f["fills"]*.8*got.mean(), f["fills"]*dollars.mean()
            marginal = 100*(earned-before[1])/max(money-before[0], 1e-9)
            before = (money, earned)
            say(f"    Q {'no cap' if size >= 10**9 else format(size, ','):>7}: in ${money:11,.0f} | earned ${earned:+10,.0f} [{f['fills']*ci[0]:+10,.0f},{f['fills']*ci[1]:+10,.0f}] = {100*earned/money:+6.1f}% "
                f"| at backtest ${money*rate:+10,.0f} half ${money*rate/2:+9,.0f} | return on the extra money {marginal:+6.1f}% | reserve ${.8*size*f['peak'] if size < 10**9 else float('nan'):12,.0f}")
            a, b = halves(hit)
            ga, na = sized(a, 80, size)
            gb, nb = sized(b, 80, size)
            say(f"             halves: dollars per fill {np.mean(ga*na)/100:+9.2f} / {np.mean(gb*nb)/100:+9.2f}")
    say("  For reference, last season's frequency: men (December-February, sample / 0.57) "
        f"{sum(len(fills([o for o in rows if o['group'] == 'men' and o['month'] == m], 80)) for m in ('2025-12', '2026-01', '2026-02'))/3/.57:.0f} fills a month; "
        f"women (January-February / 0.956) {sum(len(fills([o for o in rows if o['group'] == 'women' and o['month'] == m], 80)) for m in ('2026-01', '2026-02'))/2/.956:.0f} fills a month")

    say("\n=== 5. ACCOUNT SIZE: one Q for all three sports at 80c, Q = account / (80c x most orders resting at once)")
    now = [o for o in rows if o["month"] == "2026-11"]
    top, day = peak(now)
    by_day = Counter(o["day"] for o in now)
    say(f"  November 2026, all three sports together: most orders resting at once {top} (on {day}); orders per day median {np.median(list(by_day.values())):.0f}, most {max(by_day.values())}")
    for label, tops in (("busiest moment", top), ("if only half of that needed funding", top/2)):
        say(f"  reserve sized for the {label}:")
        for account in (100, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 1000000):
            size = account/(.8*tops)
            text, total, total_in, total_bt, draws = f"    account ${account:9,} Q {size:8,.0f}", 0.0, 0.0, 0.0, np.zeros(4000)
            for key, name in GROUPS:
                hit = fills([o for o in rows if o["group"] == key], 80)
                got, net = sized(hit, 80, size)
                dollars = got*net/100
                total += freq[key]["fills"]*dollars.mean()
                total_in += freq[key]["fills"]*.8*got.mean()
                total_bt += freq[key]["fills"]*.8*got.mean()*np.mean([100*o["payout"]-80 for o in hit])/80
                draws += freq[key]["fills"]*dollars[RNG.integers(0, len(hit), (4000, len(hit)))].mean(axis=1)
                text += f" | {key} ${freq[key]['fills']*dollars.mean():+9,.0f}"
            lo, hi = np.percentile(draws, [2.5, 97.5])
            say(text+f" | all ${total:+10,.0f} a month [{lo:+10,.0f},{hi:+10,.0f}] = {100*total/account:+6.1f}% of the account | money in ${total_in:11,.0f} | at backtest ${total_bt:+10,.0f}, half ${total_bt/2:+9,.0f}")
    say("\n  5b. A $10,000 account by sport and by the share of the through-volume we are assumed to get (reserve sized for that sport set's busiest moment in November 2026)")
    for label, keys in (("all three", ("men", "women", "tennis")), ("men + tennis", ("men", "tennis")), ("men only", ("men",)), ("women only", ("women",)), ("tennis only (peak of May-October)", ("tennis",))):
        tops = freq["tennis"]["peak"] if keys == ("tennis",) else peak([o for o in now if o["group"] in keys])[0]
        size = 10000/(.8*tops)
        text = f"    {label:34} most resting {tops:3} Q {size:6,.0f}"
        for share in (.05, .10, .25, .50, 1.0):
            total, draws = 0.0, np.zeros(4000)
            for key in keys:
                hit = fills([o for o in rows if o["group"] == key], 80)
                got, net = sized(hit, 80, size, share=share)
                dollars = got*net/100
                total += freq[key]["fills"]*dollars.mean()
                draws += freq[key]["fills"]*dollars[RNG.integers(0, len(hit), (4000, len(hit)))].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            text += f" | share {100*share:3.0f}%: ${total:+7,.0f} [{lo:+7,.0f},{hi:+7,.0f}]"
        say(text)
    say("\n  5c. Worst days at Q = 321 (the $10,000 account, all three sports): realised dollars per day over the whole sample, as sampled")
    daily = defaultdict(float)
    for o in rows:
        if o["levels"][80]["filled"]:
            daily[o["day"]] += min(321, SHARE*o["levels"][80]["through"])*(100*o["payout"]-80)/100
    values = sorted(daily.values())
    say(f"    days with a fill {len(values)}; worst five {[round(v) for v in values[:5]]}; best five {[round(v) for v in values[-5:]]}; losing days {100*np.mean([v < 0 for v in values]):.0f}%")
    json_rows = [dict(group=o["group"], event=o["event"], ticker=o["ticker"], start=o["start"], settle=o["settle"], day=o["day"], price=o["price"],
                      payout=o["payout"], levels=o["levels"]) for o in rows]
    (OUT/"rows.json").write_text(json.dumps(json_rows))
    (OUT/"report.txt").write_text(buffer.getvalue())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("download", "report"))
    parser.add_argument("--minutes", type=float, default=9)
    args = parser.parse_args()
    if args.command == "download":
        download(args.minutes)
    else:
        report()


if __name__ == "__main__":
    main()
