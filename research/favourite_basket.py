"""A nightly basket of heavy pre-event favourites: does it pay, how smoothly, and how much money can it hold?

The idea: every day buy every pre-event favourite priced 90-99c shortly before
its event starts, in the families where such favourites win more often than
their price, hold each to settlement and reuse the money the next day.

Offline only. Nothing is downloaded, no orders, no authentication. It reuses
the per-event rows other studies already built and writes
data/processed/favourite-basket/ (table.json, summary.json, report.txt).

    .venv/bin/python research/favourite_basket.py

Sources (one row per Kalshi event; an event found in several keeps the first)
  1 tennis-contexts/rows.json        six tennis tours. Price = the ASK on the last one-minute bar before the detected
                                     start of play (spread at most 4c); band on that bar's midpoint. "price kind: last bar".
                                     The start is detected after the fact from the first 2c move, so the choice of bar
                                     looks ahead by a minute or two; the trader sees the same quote in real time.
  2 favourite-contexts cbb rows      men's and women's college basketball. Price = median midpoint of the last 10-15
    + wcbb-backfill/rows.json        one-minute bars ending 2 minutes before the scheduled start (25 minutes for last
                                     season, whose start is estimated) plus half their mean spread. "tip". The backfill
                                     is read if present, so a rerun picks up whatever that download has added.
  3 dip-new-families/rows-*.json     NBA, WNBA, other basketball, soccer, cricket, combat. Price = last midpoint at or
                                     before the scheduled start plus half the warm-up spread. "at start". Boxing and
                                     MMA have no schedule: their price is the warm-up one and they are flagged.
  4 favourite-contexts other rows,   soccer, esports, cricket, other basketball, college football and the rest: only the
    calibration-map/rows-*.json      warm-up quote exists (median midpoint of the first 15 bars of the window, two to
                                     three hours before the start) plus half its spread. "warm-up". Calibration-map
                                     rows must pass that script's timing check.
  extra nba-backfill/rows.json       Polymarket midpoints for NBA games before Kalshi listed them; shown separately.

Conventions
* Band: 90 <= midpoint < 99 on the price above. Gap = 100 x settlement - midpoint. Cancelled events were dropped upstream.
* Taker: pays the price above plus 0.07*p*(1-p) per contract; in the day-by-day simulation contracts are whole and the
  fee is rounded UP to the cent per order. Return = profit / (price + fee).
* Day: the US Eastern date of (start - 5 hours), so a game tipping after midnight belongs to the evening before.
* Simulation: a fixed bankroll each day (no compounding) and, separately, a compounding one. All of a day's positions
  are assumed back in cash before the next day's purchases (the share that is not is reported). Unfilled size is
  ignored; capacity is examined on its own.
* Sizing rules: "equal" splits the bankroll across the day's favourites; "fixed" stakes a set share of the bankroll
  per game and scales down when the day would exceed the bankroll; "family" splits the bankroll across the families
  present and then across each family's games.
* Membership: (a) every family; (b) the contexts with prior support (first-round tennis qualifying, men's and women's
  college basketball, NBA regular season); (c) the cells a rule applied to the EARLIER half alone would pick (at least
  MIN_CELL favourites and a positive net return at the ask after the fee), scored on the later half.
* Intervals: Wilson on the win rate for single cells, bootstrap resampling DAYS (2,000 draws) for baskets.
* Dip bid: an 80c resting bid from the start, filled when the in-play low is strictly below 80c, held to settlement;
  80c reserved per contract on every order. Maker fee 0.0175*p*(1-p) where the series charges one.
* Maker basket: research/favourite_gap.py Q5 rows (tennis favourites at 90-99c at the warm-up; a bid joined at the
  best bid there, filled only when a later pre-match minute prints strictly below it; shown with and without the fills
  in the minute play was detected, which a cancel at the start may or may not avoid). "Hybrid" pays the last ask
  before play when the bid never filled. Other families have no pre-start bars in the rows on disk, so tennis's fill
  rate and loss-rate penalty are applied to them as a stated assumption.
* Forward expectation: cell edges are shrunk towards the precision-weighted mean of every 90-99c cell with a
  normal-normal model (as research/edge_reality.py does across tours), then nights are simulated by resampling the
  line-ups of real nights (the full-coverage window; last season's sampled January-March; May-September) with every
  favourite winning with probability (cost + edge) / 100. A simulated month is 30 active nights.
"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROC = ROOT / "data/processed"
OUT = PROC / "favourite-basket"
EASTERN = ZoneInfo("America/New_York")
RNG = np.random.default_rng(31)
BANKROLL, MIN_CELL, DRAWS = 1000.0, 20, 2000
SUPPORTED = ("tennis: qualifying, first round", "cbb-men", "cbb-women", "nba: regular")
MAKER_SERIES = {"KXNCAAMBGAME", "KXNBAGAME", "KXATPMATCH", "KXWTAMATCH"}
SUBBANDS = (("90-93", 90, 93), ("93-96", 93, 96), ("96-99", 96, 99), ("85-90 control", 85, 90))
out_buffer = io.StringIO()


def say(text=""):
    print(text, flush=True)
    out_buffer.write(text+"\n")


# ---------------------------------------------------------------- the table

def day_of(ts):
    return (datetime.fromtimestamp(ts, EASTERN)-timedelta(hours=5)).date().isoformat()


def row(event, series, family, context, start, settle, mid, spread, ask, kind, payout, source, archived,
        pre_volume=None, volume_kind=None, low=None, warm_mid=None):
    return dict(event=event, series=series, family=family, context=context, cell=f"{family}: {context}" if context else family,
                start=int(start), settle=int(settle), day=day_of(start), month=day_of(start)[:7], mid=float(mid), spread=float(spread),
                ask=float(ask), price_kind=kind, payout=float(payout), source=source, archived=bool(archived),
                pre_volume=pre_volume, volume_kind=volume_kind, low_in_play=low, warm_mid=warm_mid,
                maker_fee=series in MAKER_SERIES)


TENNIS_CONTEXT = {}


def tennis_rows():
    from research import tennis_contexts
    rows = []
    for r in tennis_contexts.prepare():
        if r.get("stable"):
            TENNIS_CONTEXT[r["event"]] = ("qualifying, first round" if r["qual"] and r["stage"] in ("qual Q1", "qual Q") else "other")
        if not r.get("stable") or r.get("start_ts") is None or r.get("ask_before_play") is None or r.get("bid_before_play") is None:
            continue
        bid, ask = r["bid_before_play"], r["ask_before_play"]
        if not 0 < bid <= ask < 100 or ask-bid > 4 or r["pay_fav"] is None:
            continue
        itf = r["tour"].startswith("ITF")
        if not r["stage"]:
            context = "ITF (no round listed)" if itf else "round unknown"
        elif r["qual"]:
            context = "qualifying, first round" if r["stage"] in ("qual Q1", "qual Q") else "qualifying, later round"
        elif r["stage"] in ("QF", "SF", "F"):
            context = "quarterfinal on"
        else:
            context = "main draw, first round" if r["first_main"] else "main draw, before the quarterfinal"
        x = row(r["event"], r["series"], "tennis", context, r["start_ts"], r["settle"], (bid+ask)/2, ask-bid, ask, "last bar",
                r["pay_fav"], "tennis-contexts", r["archived"], r.get("volume_60m"), "contracts in the hour before play",
                r.get("low_in_play"), r["start"])
        x["tour"], x["warm_ask"] = r["tour"], r["ask0"]
        rows.append(x)
    return rows


def college_rows():
    rows, seen = [], set()
    files = [(PROC/"favourite-contexts/rows-cbb-men.json", "favourite-contexts"), (PROC/"favourite-contexts/rows-cbb-women.json", "favourite-contexts"),
             (PROC/"wcbb-backfill/rows.json", "wcbb-backfill")]
    counts = Counter()
    for path, source in files:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for r in (data.values() if isinstance(data, dict) else data):
            if r.get("skip") or r["event"] in seen or not r.get("full"):
                continue
            seen.add(r["event"])
            if r.get("tip_price") is None or not r.get("begin"):
                counts[f'{r["family"]} no clean quote at the tip'] += 1
                continue
            rows.append(row(r["event"], r["series"], r["family"], None, r["begin"], r["settle"], r["tip_price"], r["tip_spread"],
                            r["tip_price"]+r["tip_spread"]/2, "tip", r["payout"], source, r["archived"], r.get("pre_volume"),
                            "contracts between the warm-up and the start", r.get("low_in_play"), r["start"]))
            counts[source+" "+r["family"]] += 1
    return rows, counts


SLUGS = {"nba": "nba", "wnba": "wnba", "college-basketball-women": None, "college-basketball-men": None, "college-football": "college-football",
         "boxing": "combat (no schedule)", "ufc": "combat (no schedule)", "mma-other": "combat (no schedule)",
         "cs2": "esports", "dota-2": "esports", "league-of-legends": "esports", "valorant": "esports", "esports-other": "esports", "euroleague": "basketball-other"}


def family_of(slug):
    if slug in SLUGS:
        return SLUGS[slug]
    for head in ("basketball", "soccer", "cricket"):
        if slug.startswith(head):
            return "basketball-other" if head == "basketball" else head
    if "tennis" in slug or slug.startswith(("atp", "wta", "itf")):
        return None
    return slug


def other_rows(seen):
    rows = []
    volumes = {}
    for fam in ("soccer", "esports", "cricket", "basketball-other"):
        path = PROC/f"favourite-contexts/rows-{fam}.json"
        if path.exists():
            volumes.update({k: v for k, v in json.loads(path.read_text()).items() if not v.get("skip")})
    for path in sorted((PROC/"dip-new-families").glob("rows-*.json")):
        family = family_of(path.stem[5:])
        if family is None:
            continue
        for r in json.loads(path.read_text()):
            if r["event"] in seen:
                continue
            seen.add(r["event"])
            scheduled = r["kind"] != "none"
            mid = r["at_start"] if scheduled and r.get("at_start") is not None else r["pre"]
            context = r.get("phase") if family == "nba" else None
            extra = volumes.get(r["event"], {})
            rows.append(row(r["event"], r["series"], family, context, r["t0"], r["settle"], mid, r["spread"], mid+r["spread"]/2,
                            "at start" if scheduled else "warm-up", r["payout"], "dip-new-families", r["archived"], extra.get("pre_volume"),
                            "contracts between the warm-up and the start" if extra.get("pre_volume") is not None else None,
                            r.get("min_after_start"), r["pre"]))
    for event, r in volumes.items():
        if event in seen or not r.get("begin"):
            continue
        seen.add(event)
        rows.append(row(event, r["series"], r["family"], None, r["begin"], r["settle"], r["start"], r["spread"], r["start"]+r["spread"]/2,
                        "warm-up", r["payout"], "favourite-contexts", r["archived"], r.get("pre_volume"),
                        "contracts between the warm-up and the start", r.get("low_in_play"), r["start"]))
    for path in sorted((PROC/"calibration-map").glob("rows-*.json")):
        family = family_of(path.stem[5:])
        if family is None:
            continue
        data = json.loads(path.read_text())
        for r in (data.values() if isinstance(data, dict) else data):
            if r.get("skip") or r["event"] in seen:
                continue
            if r.get("sched"):
                ok, start = r["warm_end"] <= r["sched"], r["sched"]
            elif r.get("expected"):
                ok, start = r["settle"] <= r["expected"]+2*3600, r["expected"]-3*3600
            else:
                ok, start = family.startswith("combat"), r["warm_end"]
            if not ok:
                continue
            seen.add(r["event"])
            rows.append(row(r["event"], r["series"], family, None, max(start, r["warm_end"]), r["settle"], r["start"], r["spread"],
                            r["start"]+r["spread"]/2, "warm-up", r["payout"], "calibration-map", r["archived"], None, None, None, r["start"]))
    return rows


def load():
    tennis = tennis_rows()
    college, counts = college_rows()
    seen = {r["event"] for r in tennis+college}
    for path in (PROC/"dip-new-families").glob("rows-college-basketball-*.json"):      # college games not priced at the tip stay out
        seen.update(r["event"] for r in json.loads(path.read_text()))
    for name in ("college-basketball-men", "college-basketball-women", "nba"):
        path = PROC/f"calibration-map/rows-{name}.json"
        if path.exists() and name != "nba":
            data = json.loads(path.read_text())
            seen.update(r["event"] for r in (data.values() if isinstance(data, dict) else data) if not r.get("skip"))
    rows = tennis+college+other_rows(seen)
    rows = [r for r in rows if 85 <= r["mid"] < 99 and r["ask"] < 100]
    rows.sort(key=lambda r: r["start"])
    assert len({r["event"] for r in rows}) == len(rows)
    return rows, counts


# ---------------------------------------------------------------- statistics

def fee_cents(ask):
    return 7*ask/100*(1-ask/100)


def net_cents(r):
    return 100*r["payout"]-r["ask"]-fee_cents(r["ask"])


def wilson(wins, n, z=1.96):
    if not n:
        return None, None
    p = wins/n
    d = 1+z*z/n
    c, h = (p+z*z/(2*n))/d, z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h, c+h


def boot_days(rows, value, draws=DRAWS):
    """95% interval of the mean of `value` over rows, resampling days."""
    by = defaultdict(list)
    for r in rows:
        by[r["day"]].append(value(r))
    if len(by) < 5:
        return None
    sums, ns = np.array([sum(v) for v in by.values()]), np.array([len(v) for v in by.values()])
    pick = RNG.integers(0, len(sums), (draws, len(sums)))
    means = sums[pick].sum(1)/ns[pick].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def cell(rows):
    if not rows:
        return dict(n=0)
    ask, pay = np.array([r["ask"] for r in rows]), np.array([r["payout"] for r in rows])
    mid = np.array([r["mid"] for r in rows])
    cost = ask+7*ask/100*(1-ask/100)
    net = 100*pay-cost
    lo, hi = wilson(pay.sum(), len(rows))
    se = max(net.std(ddof=1)/math.sqrt(len(rows)) if len(rows) > 1 else 0, 100*math.sqrt(np.mean(mid/100*(1-mid/100))/len(rows)))
    return dict(n=len(rows), mid=float(mid.mean()), spread=float(np.mean([r["spread"] for r in rows])), ask=float(ask.mean()), won=float(pay.mean()), losses=float(len(rows)-pay.sum()),
                gap=float(100*pay.mean()-mid.mean()), gap_wilson=[100*lo-float(mid.mean()), 100*hi-float(mid.mean())],
                net=float(net.mean()), net_wilson=[100*lo-float(cost.mean()), 100*hi-float(cost.mean())], ret=float(100*(net/cost).mean()),
                ret_wilson=[100*(100*lo-float(cost.mean()))/float(cost.mean()), 100*(100*hi-float(cost.mean()))/float(cost.mean())],
                net_days=boot_days(rows, net_cents), se=float(se), first=rows[0]["day"], last=rows[-1]["day"])


def band(ci, unit="c", digits=1):
    return "      -      " if not ci or ci[0] is None else f"[{ci[0]:+5.{digits}f},{ci[1]:+5.{digits}f}]{unit}"


def line(label, c, width=44):
    if not c["n"]:
        return f"  {label[:width]:{width}} n    0"
    return (f'  {label[:width]:{width}} n {c["n"]:5} ask {c["ask"]:5.1f} won {100*c["won"]:5.1f}% | return {c["ret"]:+5.2f}% {band(c["ret_wilson"], "%")} '
            f'| gap {c["gap"]:+5.2f}c net {c["net"]:+5.2f}c {band(c["net_wilson"])} days {band(c["net_days"])}')


def halves(rows):
    rows = sorted(rows, key=lambda r: r["start"])
    return rows[:len(rows)//2], rows[len(rows)//2:]


def in_band(rows, lo=90, hi=99):
    return [r for r in rows if lo <= r["mid"] < hi]


# ---------------------------------------------------------------- the daily simulation

def stakes(day_rows, rule, bankroll, share):
    n = len(day_rows)
    if rule == "equal":
        return [bankroll/n]*n
    if rule == "fixed":
        return [min(share*bankroll, bankroll/n)]*n
    fams = Counter(r["family"] for r in day_rows)
    return [bankroll/len(fams)/fams[r["family"]] for r in day_rows]


def bet_pnl(r, stake, mode):
    """Dollars of profit and of capital committed for one event under a trading mode."""
    p = r["ask"]/100
    if mode == "basket":
        n = math.floor(stake/(p+0.07*p*(1-p))+1e-9)
        if n < 1:
            return 0.0, 0.0
        fee = math.ceil(7*n*p*(1-p)-1e-9)/100
        return n*(r["payout"]-p)-fee, n*p+fee
    filled = r["low_in_play"] is not None and r["low_in_play"] < 80
    if mode == "dip":
        n = math.floor(stake/0.80+1e-9)
        fee = math.ceil(1.75*n*0.8*0.2-1e-9)/100 if r["maker_fee"] else 0.0
        return (n*(r["payout"]-0.80)-fee if filled else 0.0), n*0.80
    if mode == "both":
        n = math.floor(stake/(p+0.07*p*(1-p)+0.80)+1e-9)
        if n < 1:
            return 0.0, 0.0
        fee = math.ceil(7*n*p*(1-p)-1e-9)/100
        dip = n*(r["payout"]-0.80)-(math.ceil(1.75*n*0.8*0.2-1e-9)/100 if r["maker_fee"] else 0.0) if filled else 0.0
        return n*(r["payout"]-p)-fee+dip, n*p+fee+n*0.80
    raise ValueError(mode)


def simulate(rows, rule="equal", bankroll=BANKROLL, share=0.10, mode="basket"):
    """{day: (return on the bankroll, share of the bankroll committed, bets)} in date order."""
    by = defaultdict(list)
    for r in rows:
        by[r["day"]].append(r)
    result = {}
    for day in sorted(by):
        parts = [bet_pnl(r, s, mode) for r, s in zip(by[day], stakes(by[day], rule, bankroll, share))]
        result[day] = (sum(p for p, _ in parts)/bankroll, sum(c for _, c in parts)/bankroll, len(parts))
    return result


def drawdown(path):
    peak, worst = -1e18, 0.0
    for x in path:
        peak = max(peak, x)
        worst = min(worst, x-peak)
    return worst


def day_stats(result):
    days = sorted(result)
    r = np.array([result[d][0] for d in days])
    if not len(r):
        return dict(days=0)
    span = (datetime.fromisoformat(days[-1])-datetime.fromisoformat(days[0])).days+1
    weeks, months = defaultdict(float), defaultdict(float)
    grow = defaultdict(lambda: 1.0)
    for d, x in zip(days, r):
        iso = datetime.fromisoformat(d).isocalendar()
        weeks[(iso[0], iso[1])] += x
        months[d[:7]] += x
        grow[d[:7]] *= 1+x
    streak = longest = 0
    for x in r:
        streak = streak+1 if x < 0 else 0
        longest = max(longest, streak)
    pick = RNG.integers(0, len(r), (DRAWS, len(r)))
    means = r[pick].mean(1)
    compound = np.cumprod(1+r)
    return dict(days=len(r), calendar_days=span, active_share=len(r)/span, bets=int(sum(result[d][2] for d in days)),
                bets_per_day=float(np.mean([result[d][2] for d in days])), committed=float(np.mean([result[d][1] for d in days])),
                mean=float(100*r.mean()), mean_ci=[float(100*np.percentile(means, 2.5)), float(100*np.percentile(means, 97.5))],
                median=float(100*np.median(r)), sd=float(100*r.std(ddof=1)) if len(r) > 1 else 0.0, win_days=float((r > 0).mean()),
                flat_days=float((r == 0).mean()), losing_days=float((r < 0).mean()), worst_day=float(100*r.min()), best_day=float(100*r.max()),
                worst_week=float(100*min(weeks.values())), losing_weeks=float(np.mean([v < 0 for v in weeks.values()])), weeks=len(weeks),
                losing_streak=int(longest), total_simple=float(100*r.sum()), total_simple_ci=[float(100*len(r)*np.percentile(means, 2.5)),
                                                                                             float(100*len(r)*np.percentile(means, 97.5))],
                total_compound=float(100*(compound[-1]-1)), drawdown_simple=float(100*drawdown(np.cumsum(r))),
                drawdown_compound=float(100*min(0.0, min(compound/np.maximum.accumulate(compound)-1))),
                per_calendar_day=float(100*r.sum()/span), monthly_simple={m: float(100*v) for m, v in sorted(months.items())},
                monthly_compound={m: float(100*(v-1)) for m, v in sorted(grow.items())}, losing_months=float(np.mean([v < 0 for v in months.values()])),
                sharpe_daily=float(r.mean()/r.std(ddof=1)) if len(r) > 1 and r.std() > 0 else None)


def show_days(label, s, width=40):
    if not s["days"]:
        return f"  {label[:width]:{width}} no days"
    return (f'  {label[:width]:{width}} days {s["days"]:3} bets/day {s["bets_per_day"]:5.1f} | mean {s["mean"]:+6.3f}% {band(s["mean_ci"], "%", 2)} median {s["median"]:+5.2f}% '
            f'sd {s["sd"]:5.2f}% | win {100*s["win_days"]:3.0f}% lose {100*s["losing_days"]:3.0f}% | worst day {s["worst_day"]:+6.1f}% week {s["worst_week"]:+6.1f}% '
            f'streak {s["losing_streak"]} | drawdown {s["drawdown_simple"]:+6.1f}% | total {s["total_simple"]:+6.1f}% compounded {s["total_compound"]:+6.1f}%')


# ---------------------------------------------------------------- sections

def coverage(rows, counts, summary):
    say("1. COVERAGE: favourites at 90-99c (midpoint at the pricing moment), one row per event")
    say(f"   table: {len(rows)} events at 85-99c, {len(in_band(rows))} at 90-99c; sources {dict(Counter(r['source'] for r in in_band(rows)))}")
    say(f"   price kinds at 90-99c: {dict(Counter(r['price_kind'] for r in in_band(rows)))}")
    say(f"   college basketball rows: {dict(counts)}")
    months = sorted({r["month"] for r in in_band(rows)})
    say("   " + f'{"family / context":42} {"n":>5} {"price kind":>10}  ' + " ".join(m[2:] for m in months))
    table = {}
    cells = sorted({r["cell"] for r in in_band(rows)})
    for name in cells:
        part = [r for r in in_band(rows) if r["cell"] == name]
        by = Counter(r["month"] for r in part)
        kinds = Counter(r["price_kind"] for r in part).most_common(1)[0][0]
        table[name] = dict(n=len(part), price_kind=kinds, months=dict(by), first=part[0]["day"], last=part[-1]["day"],
                           archived=sum(r["archived"] for r in part))
        say(f'   {name[:42]:42} {len(part):5} {kinds:>10}  ' + " ".join(f"{by.get(m, 0):5}" for m in months))
    thin = [k for k, v in table.items() if v["n"] < 100 or len(v["months"]) < 4]
    say(f"   thin history (under 100 favourites or under four months): {', '.join(thin)}")
    say("   " + archive_note())
    late = [r for r in in_band(rows) if r["cell"] in SUPPORTED]
    first = defaultdict(lambda: 1e18)
    for r in late:
        first[r["day"]] = min(first[r["day"]], r["start"])
    order = sorted(first)
    nxt = {a: first[b] for a, b in zip(order, order[1:])}
    still = [r for r in late if r["day"] in nxt and r["settle"] > nxt[r["day"]]]
    hold = np.array([(r["settle"]-r["start"])/3600 for r in late])
    say(f"   supported cells: median hold {np.median(hold):.1f}h, 90th percentile {np.percentile(hold, 90):.1f}h; {len(still)} of {len(late)} positions "
        f"({100*len(still)/len(late):.1f}%) were still open at the next day's first purchase")
    say("   basket (b) by month: nights with at least one favourite / favourites on a median such night (last season's college nights are samples)")
    per = defaultdict(Counter)
    for r in late:
        per[r["month"]][r["day"]] += 1
    activity = {m: dict(nights=len(c), median=float(np.median(list(c.values()))), bets=sum(c.values())) for m, c in sorted(per.items())}
    say("     " + "  ".join(f'{m[2:]} {v["nights"]:2}/{v["median"]:.0f}' for m, v in activity.items()))
    summary["activity"] = activity
    summary["coverage"] = dict(cells=table, thin=thin, still_open_share=len(still)/len(late), median_hold_hours=float(np.median(hold)))


def archive_note():
    """How much of last season's college basketball is on disk (both archives are uniform random samples of the listed games)."""
    text = []
    try:
        from research import favourite_contexts
        events = favourite_contexts.events_of("KXNCAAMBGAME", archive=True)
        have = favourite_contexts.store("cbb-men")
        old = [k for k, e in events.items() if e[3]]
        text.append(f"men {sum(k in have for k in old)} of {len(old)} listed games read ({100*sum(k in have for k in old)/len(old):.0f}%)")
    except Exception as exc:                                   # the listing cache is optional here
        text.append(f"men: listing cache not readable ({type(exc).__name__})")
    path = PROC/"wcbb-backfill/coverage.json"
    if path.exists():
        c = json.loads(path.read_text())
        text.append(f'women {c.get("2025-26 read", 0)} of {c.get("2025-26 listed", 0)} ({100*c.get("2025-26 read", 0)/max(1, c.get("2025-26 listed", 1)):.0f}%; wcbb-backfill, still downloading)')
    return ("last season's college basketball is a random sample of its games, so its nights hold fewer favourites than really played: " + "; ".join(text) +
            ". This season is complete to mid-November. NBA has one season on Kalshi. First-round qualifying runs January to November 2026.")


def families(rows, summary):
    say("\n3a. EACH FAMILY AND CONTEXT ON ITS OWN, 90-99c (return = profit at the ask after the taker fee / cost; Wilson intervals; 'days' = bootstrap of days)")
    out = {}
    names = sorted({r["cell"] for r in in_band(rows)}, key=lambda k: (k not in SUPPORTED, k))
    for name in names:
        part = [r for r in in_band(rows) if r["cell"] == name]
        a, b = halves(part)
        out[name] = dict(all=cell(part), first=cell(a), second=cell(b))
        say(line(("* " if name in SUPPORTED else "  ")+name, out[name]["all"]))
        if len(part) >= 2*MIN_CELL:
            say(f'      halves: {a[0]["day"]}..{a[-1]["day"]} n {len(a)} return {out[name]["first"]["ret"]:+5.2f}% net {out[name]["first"]["net"]:+5.2f}c {band(out[name]["first"]["net_wilson"])}'
                f' | {b[0]["day"]}..{b[-1]["day"]} n {len(b)} return {out[name]["second"]["ret"]:+5.2f}% net {out[name]["second"]["net"]:+5.2f}c {band(out[name]["second"]["net_wilson"])}')
    say("   * = prior support. First-round qualifying by tour, and the same favourites at the warm-up quote (hours earlier):")
    q1 = [r for r in in_band(rows) if r["cell"] == SUPPORTED[0]]
    for tour in sorted({r["tour"] for r in q1}):
        out["Q1 "+tour] = cell([r for r in q1 if r["tour"] == tour])
        say(line("    Q1 "+tour, out["Q1 "+tour]))
    warm = [dict(r, mid=r["warm_mid"], ask=r["warm_ask"]) for r in rows if r["cell"] == SUPPORTED[0] and r.get("warm_ask") is not None and 90 <= r["warm_mid"] < 99]
    out["Q1 at the warm-up quote"] = cell(warm)
    say(line("    Q1, all tours, bought at the warm-up ask", out["Q1 at the warm-up quote"]))
    say("   seasons of college basketball and NBA (last season chose nothing here; shown as the cleanest earlier/later split):")
    for name in ("cbb-men", "cbb-women", "nba: regular"):
        part = [r for r in in_band(rows) if r["cell"] == name]
        for label, sub in (("to June 2026", [r for r in part if r["day"] < "2026-07-01"]), ("from October 2026", [r for r in part if r["day"] >= "2026-07-01"])):
            out[f"{name} {label}"] = cell(sub)
            say(line(f"    {name} {label}", out[f"{name} {label}"]))
    summary["families"] = out
    return out


def baskets(rows, summary):
    say("\n3b. THREE BASKETS, per bet at the ask after the fee")
    band90 = in_band(rows)
    split = sorted(r["start"] for r in band90)[len(band90)//2]
    split_day = day_of(split)
    early, late = [r for r in band90 if r["start"] < split], [r for r in band90 if r["start"] >= split]
    chosen, table = [], {}
    for name in sorted({r["cell"] for r in early}):
        c = cell([r for r in early if r["cell"] == name])
        table[name] = c
        if c["n"] >= MIN_CELL and c["net"] > 0:
            chosen.append(name)
    strict = [k for k in chosen if table[k]["net_wilson"][0] > 0]
    say(f"   split at {split_day}: {len(early)} favourites before, {len(late)} after")
    say(f"   (c) cells with {MIN_CELL}+ favourites and a positive net return in the earlier half: {', '.join(chosen)}")
    say(f"       cells whose Wilson interval was also above zero in the earlier half: {', '.join(strict) or 'none'}")
    members = {"(a) every family": lambda r: True, "(b) prior support": lambda r: r["cell"] in SUPPORTED,
               "(c) chosen on the earlier half": lambda r: r["cell"] in chosen,
               "(c') earlier-half interval above zero": lambda r: r["cell"] in strict,
               "everything outside (b)": lambda r: r["cell"] not in SUPPORTED}
    out = dict(split=split_day, chosen=chosen, strict=strict, earlier_cells=table)
    for name, keep in members.items():
        a, b, c = cell([r for r in early if keep(r)]), cell([r for r in late if keep(r)]), cell([r for r in band90 if keep(r)])
        out[name] = dict(earlier=a, later=b, all=c)
        say(f"   {name}")
        say(line("     earlier half (the choice was made here)" if name.startswith("(c") else "     earlier half", a))
        say(line("     LATER HALF (out of sample for (c))" if name.startswith("(c") else "     later half", b))
        say(line("     both", c))
    say("   (b) was assembled from studies that saw BOTH halves, so its later half is not out of sample; only (c) and (c') are.")
    say("\n3c. PRICE SUB-BANDS (both halves pooled)")
    for name in ("(a) every family", "(b) prior support"):
        for label, lo, hi in SUBBANDS:
            c = cell([r for r in in_band(rows, lo, hi) if members[name](r)])
            out[f"{name} {label}"] = c
            say(line(f"   {name[:18]} {label}", c))
    say("   (b) by sub-band and half (the sub-bands were looked at after the fact; the halves show whether the pattern repeats):")
    for label, lo, hi in SUBBANDS:
        for half, test in (("earlier", lambda r: r["start"] < split), ("later", lambda r: r["start"] >= split)):
            c = cell([r for r in in_band(rows, lo, hi) if r["cell"] in SUPPORTED and test(r)])
            out[f"(b) {label} {half}"] = c
            say(line(f"     {label} {half} half", c))
    for name in SUPPORTED:
        for label, lo, hi in SUBBANDS:
            c = cell([r for r in in_band(rows, lo, hi) if r["cell"] == name])
            out[f"{name} {label}"] = c
            say(line(f"   {name[:26]} {label}", c))
    summary["baskets"] = out
    return members, chosen, split


def fee_rounding(rows, summary):
    say("\n2a. THE TAKER FEE ROUNDED UP PER ORDER, basket (b), 90-99c (whole-cent ask = the price rounded up)")
    part = [r for r in in_band(rows) if r["cell"] in SUPPORTED]
    out = {}
    for n in (1, 3, 10, 30, 100, 1000):
        ask = np.array([math.ceil(r["ask"]-1e-9) for r in part], float)
        pay = np.array([r["payout"] for r in part])
        fee = np.ceil(7*n*ask/100*(1-ask/100)-1e-9)/n
        exact = 7*ask/100*(1-ask/100)
        net = 100*pay-ask-fee
        out[n] = dict(fee=float(fee.mean()), exact=float(exact.mean()), net=float(net.mean()), ret=float(100*(net/(ask+fee)).mean()))
        say(f"   {n:5} contracts per order: fee {fee.mean():.3f}c a contract (unrounded {exact.mean():.3f}c) -> return {out[n]['ret']:+5.2f}% net {net.mean():+5.2f}c")
    cont = cell(part)
    say(f"   (the tables elsewhere use the unrounded price and fee: return {cont['ret']:+5.2f}% net {cont['net']:+5.2f}c; rounding the ask up to a whole cent costs "
        f"{cont['net']-out[1000]['net']:.2f}c)")
    summary["fee_rounding"] = out


def simulations(rows, members, summary, split):
    say(f"\n2b. THE BASKET DAY BY DAY (bankroll ${BANKROLL:,.0f}, whole contracts, fee rounded up per order; returns are % of the bankroll per ACTIVE day)")
    out = {}
    rules = (("equal split", dict(rule="equal")), ("fixed 10% a game, capped at the bankroll", dict(rule="fixed", share=0.10)),
             ("fixed 5% a game, capped", dict(rule="fixed", share=0.05)), ("equal across families", dict(rule="family")))
    for name in ("(a) every family", "(b) prior support", "(c) chosen on the earlier half"):
        part = [r for r in in_band(rows) if members[name](r)]
        say(f"  {name}: {len(part)} bets")
        for label, kw in rules:
            s = day_stats(simulate(part, **kw))
            out[f"{name} | {label}"] = s
            say(show_days("  "+label, s))
    say("  out of sample and full coverage (fixed 10% a game unless said):")
    late = [r for r in in_band(rows) if members["(c) chosen on the earlier half"](r) and r["start"] >= split]
    for label, sub, kw in (("(c) on the later half, equal split", late, dict(rule="equal")), ("(c) on the later half", late, dict(rule="fixed")),
                           ("(b) on the later half", [r for r in in_band(rows) if r["cell"] in SUPPORTED and r["start"] >= split], dict(rule="fixed")),
                           ("(b) in the full-coverage window, equal split", window_of(rows)[0], dict(rule="equal")),
                           ("(b) in the full-coverage window", window_of(rows)[0], dict(rule="fixed"))):
        s = day_stats(simulate(sub, **kw))
        out["check | "+label] = s
        say(show_days("  "+label, s, 46))
    say(f"  full-coverage window = {window_of(rows)[1]}..{window_of(rows)[2]}: the only nights on disk with every college game; earlier nights are samples, so their")
    say("  baskets are smaller than a real night's (more single-game days, fewer losing days, bigger swings) and the totals above are NOT a forecast.")
    part = [r for r in in_band(rows) if members["(b) prior support"](r)]
    for bank in (100.0, 10000.0):
        s = day_stats(simulate(part, rule="equal", bankroll=bank))
        out[f"(b) | equal split | bankroll {bank:.0f}"] = s
        say(show_days(f"  (b) equal split, bankroll ${bank:,.0f}", s))
    main = out["(b) prior support | equal split"]
    say(f'  (b) equal split: active on {main["days"]} of {main["calendar_days"]} calendar days ({100*main["active_share"]:.0f}%); {100*main["committed"]:.0f}% of the bankroll committed on an active day; '
        f'{main["per_calendar_day"]:+.3f}% per calendar day; losing weeks {100*main["losing_weeks"]:.0f}% of {main["weeks"]}; losing months {100*main["losing_months"]:.0f}%')
    for key in ("(b) prior support | equal split", "(b) prior support | fixed 10% a game, capped at the bankroll"):
        s = out[key]
        say(f"  monthly return, {key}: (not compounded / compounded)")
        say("    " + "  ".join(f'{m[2:]} {s["monthly_simple"][m]:+5.1f}/{s["monthly_compound"][m]:+5.1f}%' for m in s["monthly_simple"]))
        say(f'    growth of the bankroll over {s["days"]} active days: {s["total_simple"]:+.1f}% {band(s["total_simple_ci"], "%")} not compounded, {s["total_compound"]:+.1f}% compounded; '
            f'deepest drawdown {s["drawdown_simple"]:+.1f}% / {s["drawdown_compound"]:+.1f}%')
    say("  by number of favourites that day, (b) equal split:")
    res = simulate(part)
    for label, lo, hi in (("1 game", 1, 1), ("2-3 games", 2, 3), ("4-9 games", 4, 9), ("10+ games", 10, 10**6)):
        sub = {d: v for d, v in res.items() if lo <= v[2] <= hi}
        s = day_stats(sub)
        out[f"(b) by games | {label}"] = s
        say(show_days("    "+label, s))
    summary["simulation"] = out
    return out


def window_of(rows):
    """The dates on which every supported family has full (not sampled) coverage."""
    part = [r for r in in_band(rows) if r["cell"] in SUPPORTED]
    begin = min(r["day"] for r in part if r["cell"] == "cbb-men" and not r["archived"])
    end = min([max(r["day"] for r in part if r["cell"] == name) for name in ("cbb-men", "cbb-women")]+[max(r["day"] for r in rows if r["family"] == "tennis")])
    return [r for r in part if begin <= r["day"] <= end], begin, end


def alternatives(rows, summary):
    say("\n4. THE SAME EVENTS, THE SAME BANKROLL: basket, 80c dip bid, both, and a maker basket (basket (b), 90-99c)")
    part = [r for r in in_band(rows) if r["cell"] in SUPPORTED]
    fills = [r for r in part if r["low_in_play"] is not None and r["low_in_play"] < 80]
    out = {}
    nets = np.array([100*r["payout"]-80-(0.28 if r["maker_fee"] else 0) for r in fills])
    lo, hi = wilson(sum(r["payout"] for r in fills), len(fills))
    out["dip per fill"] = dict(orders=len(part), fills=len(fills), fill_rate=len(fills)/len(part), won=float(np.mean([r["payout"] for r in fills])),
                               net=float(nets.mean()), net_wilson=[100*lo-80.2, 100*hi-80.2], ret=float(nets.mean()/80*100),
                               per_order=float(nets.sum()/len(part)), per_order_ret=float(100*nets.sum()/len(part)/80))
    d = out["dip per fill"]
    say(f'   dip bid: {d["orders"]} orders, {d["fills"]} fills ({100*d["fill_rate"]:.0f}%), won {100*d["won"]:.1f}% (needs 80%); per fill {d["ret"]:+.1f}% = {d["net"]:+.1f}c {band(d["net_wilson"])}; '
        f'per order (80c reserved each) {d["per_order_ret"]:+.2f}% = {d["per_order"]:+.2f}c')
    for name in SUPPORTED:
        sub = [r for r in part if r["cell"] == name]
        f = [r for r in sub if r["low_in_play"] is not None and r["low_in_play"] < 80]
        if f:
            say(f'     {name[:34]:34} orders {len(sub):4} fills {len(f):3} ({100*len(f)/len(sub):3.0f}%) won {100*np.mean([r["payout"] for r in f]):5.1f}% per fill {np.mean([100*r["payout"]-80 for r in f]):+5.1f}c '
                f'per order {sum(100*r["payout"]-80 for r in f)/len(sub):+5.2f}c | basket on the same events {cell(sub)["net"]:+5.2f}c')
    say("   day by day, equal split of the bankroll across the day's events:")
    for label, mode in (("basket (taker, at the ask)", "basket"), ("80c dip bid only", "dip"), ("basket + dip bid, one of each per event", "both")):
        s = day_stats(simulate(part, mode=mode))
        out[label] = s
        say(show_days("  "+label, s, 42))
        say(f'      per calendar day {s["per_calendar_day"]:+.3f}%; mean/sd per active day {s["sharpe_daily"]:.3f}; losing weeks {100*s["losing_weeks"]:.0f}%; losing months {100*s["losing_months"]:.0f}%')
    for label, mode in (("basket", "basket"), ("80c dip bid only", "dip"), ("basket + dip bid", "both")):
        s = day_stats(simulate(window_of(rows)[0], mode=mode))
        out["window | "+label] = s
        say(show_days(f"  full-coverage window: {label}", s, 42))
    say("   the dip bid reserves the whole bankroll and uses about a quarter of it; a bankroll split half basket, half dip (two separate pots):")
    a, b = simulate(part, mode="basket"), simulate(part, mode="dip")
    mix = {d_: ((a[d_][0]+b[d_][0])/2, (a[d_][1]+b[d_][1])/2, a[d_][2]) for d_ in a}
    out["half and half"] = day_stats(mix)
    say(show_days("  half basket, half dip", out["half and half"], 42))
    corr = float(np.corrcoef([a[d_][0] for d_ in sorted(a)], [b[d_][0] for d_ in sorted(a)])[0, 1])
    say(f"      correlation of the two daily results {corr:+.2f}")
    # Maker basket from the tennis Q5 rows.
    gap_rows = PROC/"favourite-gap/rows.json"
    if gap_rows.exists():
        q5 = [r for r in json.loads(gap_rows.read_text()) if r.get("bids") and 90 <= r["start"] < 99 and r["event"] in TENNIS_CONTEXT]
        say("   maker basket, tennis (favourite_gap.py Q5: the favourite is 90-99c at the warm-up, a bid joins the best bid there and rests until play;")
        say("   a fill needs a later pre-match print strictly below it; 'late cancel' also counts fills in the minute play was detected):")
        for label, sub in (("first-round qualifying", [r for r in q5 if TENNIS_CONTEXT[r["event"]] == "qualifying, first round"]), ("every tennis favourite", q5)):
            for name, late in (("cancelled in time", False), ("late cancel", True)):
                hit = [r for r in sub if r["bids"]["at bid"]["strict"] and (late or not r["bids"]["at bid"]["strict_in_detection_bar"])]
                miss = [r for r in sub if not r["bids"]["at bid"]["strict"]]
                if not hit:
                    continue
                level = np.array([r["bids"]["at bid"]["level"] for r in hit], float)
                fee = np.array([1.75*x/100*(1-x/100) if r["series"] in MAKER_SERIES else 0.0 for r, x in zip(hit, level)])
                nets = 100*np.array([r["payout"] for r in hit])-level-fee
                take = np.array([100*r["payout"]-r["ask0"]-fee_cents(r["ask0"]) for r in sub])
                lo, hi = wilson(sum(r["payout"] for r in hit), len(hit))
                m = dict(orders=len(sub), fills=len(hit), fill_rate=len(hit)/len(sub), level=float(level.mean()), won_filled=float(np.mean([r["payout"] for r in hit])),
                         won_never_filled=float(np.mean([r["payout"] for r in miss])) if miss else None, won_all=float(np.mean([r["payout"] for r in sub])),
                         net_per_fill=float(nets.mean()), net_wilson=[100*lo-float((level+fee).mean()), 100*hi-float((level+fee).mean())],
                         per_order=float(nets.sum()/len(sub)), taker_net=float(take.mean()))
                out[f"maker | {label} | {name}"] = m
                say(f'     {label[:24]:24} {name:17} orders {m["orders"]:4} filled {100*m["fill_rate"]:3.0f}% at {m["level"]:.1f}c | won: filled {100*m["won_filled"]:5.1f}% never filled '
                    f'{100*(m["won_never_filled"] or 0):5.1f}% all {100*m["won_all"]:5.1f}% | per fill {m["net_per_fill"]:+5.2f}c {band(m["net_wilson"])} = {100*m["net_per_fill"]/m["level"]:+5.2f}% | '
                    f'per order {m["per_order"]:+5.2f}c | taking the warm-up ask on all of them {m["taker_net"]:+5.2f}c')
        table = {r["event"]: r for r in rows}
        sub = [r for r in q5 if TENNIS_CONTEXT[r["event"]] == "qualifying, first round"]
        total, used = 0.0, 0
        for r in sub:
            b_ = r["bids"]["at bid"]
            if b_["strict"]:
                total += 100*r["payout"]-b_["level"]-(1.75*b_["level"]/100*(1-b_["level"]/100) if r["series"] in MAKER_SERIES else 0.0)
                used += 1
            elif r["event"] in table:                         # never filled: pay the last ask before play instead
                total += net_cents(table[r["event"]])
                used += 1
        out["maker | hybrid"] = dict(orders=len(sub), traded=used, per_order=total/len(sub))
        say(f"     first-round qualifying, HYBRID (rest the bid, late cancel; if never filled pay the last ask before play): traded {used} of {len(sub)}, {total/len(sub):+.2f}c per order")
        m = out.get("maker | every tennis favourite | late cancel")
        if m:
            penalty = (1-m["won_filled"])/(1-m["won_all"])
            base = cell([r for r in part if r["family"] != "tennis"])
            lose = min(1.0, (1-base["won"])*penalty)
            level = base["mid"]-base["spread"]/2 if "spread" in base else base["mid"]-0.6
            fee = 1.75*level/100*(1-level/100)*np.mean([r["maker_fee"] for r in part if r["family"] != "tennis"])
            model = dict(fill_rate=m["fill_rate"], loss_multiplier=penalty, assumed_level=level, net_per_fill=100*(1-lose)-level-fee,
                         per_order=m["fill_rate"]*(100*(1-lose)-level-fee), taker_net=base["net"])
            out["maker | basketball, modelled"] = model
            say(f'     basketball in (b), MODELLED (no pre-start bars on disk): tennis\'s fill rate {100*m["fill_rate"]:.0f}% and its losses x{penalty:.2f} among fills, bid {level:.1f}c: '
                f'per fill {model["net_per_fill"]:+.2f}c, per order {model["per_order"]:+.2f}c, against {base["net"]:+.2f}c a bet for the taker. An assumption, not a measurement.')
    summary["alternatives"] = out


def recorder_sizes():
    """Displayed size at the favourite's ask from the live recorder: last record before the start, favourite at 90-99c."""
    folder, best = ROOT/"data/raw/live-games", {}
    if not folder.exists():
        return []
    for path in sorted(folder.glob("*.jsonl")):
        with path.open() as handle:
            for text in handle:
                try:
                    x = json.loads(text)
                    start = datetime.fromisoformat(x["start"].replace("Z", "+00:00")).timestamp()
                    q = x["quote"]
                    bid, ask = 100*float(q["yes_bid_dollars"]), 100*float(q["yes_ask_dollars"])
                    status = str((x.get("state") or {}).get("status", "")).lower()
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
                tennis = "MATCH" in x["event"].split("-")[0] and x.get("type", "").startswith("tennis")
                before = x["ts"] < start and start-x["ts"] <= 1800 if not tennis else status in ("", "scheduled", "not_started", "notstarted", "created", "pre")
                if not before or not 0 < bid <= ask < 100 or ask-bid > 4:
                    continue
                mid = (bid+ask)/2
                if mid >= 50:
                    rec = dict(mid=mid, ask=ask, size=float(q.get("yes_ask_size_fp") or 0))
                else:
                    rec = dict(mid=100-mid, ask=100-bid, size=float(q.get("yes_bid_size_fp") or 0))
                rec.update(event=x["event"], type=x.get("type"), ts=x["ts"], series=x["event"].split("-")[0])
                best[x["event"]] = rec
    return [r for r in best.values() if 90 <= r["mid"] < 99]


def capacity(rows, summary):
    say("\n5. CAPACITY")
    out = {}
    part = [r for r in in_band(rows) if r["cell"] in SUPPORTED]
    say("   contracts traded before the start (one-minute bars; both sides of every trade, so an upper bound on what a buyer could have taken):")
    for name in SUPPORTED:
        v = np.array([r["pre_volume"] for r in part if r["cell"] == name and r["pre_volume"] is not None])
        if not len(v):
            say(f"     {name[:34]:34} no pre-start volume in the rows on disk")
            continue
        kind = next(r["volume_kind"] for r in part if r["cell"] == name and r["pre_volume"] is not None)
        q = np.percentile(v, [10, 25, 50, 75])
        out["volume | "+name] = dict(n=len(v), p10=float(q[0]), p25=float(q[1]), median=float(q[2]), p75=float(q[3]), none=float((v < 1).mean()), kind=kind)
        say(f"     {name[:34]:34} n {len(v):4} {kind}: 10th pct {q[0]:7.0f}, quartile {q[1]:7.0f}, median {q[2]:7.0f}, upper quartile {q[3]:7.0f}; none traded {100*(v < 1).mean():.0f}%")
    sizes = recorder_sizes()
    say(f"   displayed size at the favourite's ask, live recorder (data/raw/live-games; last listing quote before the start; {len(sizes)} favourites at 90-99c):")
    groups = defaultdict(list)
    for r in sizes:
        groups[r["series"]].append(r)
    for series, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        v = np.array([r["size"] for r in g])
        dollars = np.array([r["size"]*r["ask"]/100 for r in g])
        out["displayed | "+series] = dict(n=len(g), p25=float(np.percentile(v, 25)), median=float(np.median(v)), p75=float(np.percentile(v, 75)),
                                          median_dollars=float(np.median(dollars)), share_100=float((v >= 100).mean()), share_1000=float((v >= 1000).mean()))
        say(f"     {series[:24]:24} n {len(g):3} contracts at the ask: quartile {np.percentile(v, 25):7.0f} median {np.median(v):7.0f} upper {np.percentile(v, 75):7.0f} "
            f"= ${np.median(dollars):,.0f}; 100+ shown {100*(v >= 100).mean():3.0f}%, 1,000+ {100*(v >= 1000).mean():3.0f}%")
    say("   return against order size, basket (b): the first contracts pay the ask; the rest is assumed to cost 1c more (depth behind the ask is not on disk)")
    base = cell(part)
    for name in SUPPORTED:
        v = np.array([r["pre_volume"] for r in part if r["cell"] == name and r["pre_volume"] is not None])
        sub = [r for r in part if r["cell"] == name]
        if not len(v) or not sub:
            continue
        c = cell(sub)
        text = []
        for size in (10, 30, 100, 300, 1000, 3000):
            ok = float((v*0.10 >= size).mean())                       # at most a tenth of what traded before the start
            text.append(f"{size}: {100*ok:3.0f}% of games, net {c['net']-(1-ok):+5.2f}c")
            out[f"size | {name} | {size}"] = dict(share_of_games=ok, net=c["net"]-(1-ok))
        say(f"     {name[:26]:26} " + " | ".join(text))
    window, begin, end = window_of(rows)
    days = defaultdict(list)
    for r in window:
        days[r["day"]].append(r)
    say(f"   dollars a night the basket can hold, full-coverage window {begin}..{end} ({len(days)} nights), taking at most a tenth of pre-start volume per game:")
    med = {name: out["volume | "+name]["median"] for name in SUPPORTED if "volume | "+name in out}
    nightly = []
    for d, rs in sorted(days.items()):
        nightly.append(sum(0.10*(r["pre_volume"] if r["pre_volume"] is not None else med.get(r["cell"], 0))*r["ask"]/100 for r in rs))
    nightly = np.array(nightly)
    per_family = {}
    for name in SUPPORTED:
        vals = [sum(0.10*(r["pre_volume"] or 0)*r["ask"]/100 for r in rs if r["cell"] == name and r["pre_volume"] is not None) for rs in days.values()]
        games = [sum(r["cell"] == name for r in rs) for rs in days.values()]
        per_family[name] = dict(median_dollars=float(np.median(vals)), games=float(np.mean(games)))
        say(f"     {name[:34]:34} games a night {np.mean(games):5.1f}; median ${np.median(vals):>9,.0f} a night, lower quartile ${np.percentile(vals, 25):>9,.0f}")
    out["nightly"] = dict(begin=begin, end=end, nights=len(days), median=float(np.median(nightly)), p25=float(np.percentile(nightly, 25)), p10=float(np.percentile(nightly, 10)),
                          per_family=per_family)
    say(f"     all four: median ${np.median(nightly):,.0f}, lower quartile ${np.percentile(nightly, 25):,.0f}, 10th percentile ${np.percentile(nightly, 10):,.0f} a night")
    say("   equal split and size (a bet that does not fit inside a tenth of its game's pre-start volume is ASSUMED to cost 1c more; depth is not on disk):")
    for bank in (300, 1000, 3000, 10000, 30000, 100000):
        fits, weight = [], []
        for rs in days.values():
            stake = bank/len(rs)
            ok = [stake/(r["ask"]/100) <= 0.10*(r["pre_volume"] if r["pre_volume"] is not None else med.get(r["cell"], 0)) for r in rs]
            fits.append(np.mean(ok))
        share = float(np.mean(fits))
        out[f"bankroll | {bank}"] = dict(share_of_bets_within_capacity=share, net=base["net"]-(1-share))
        say(f"     bankroll ${bank:>7,}: {100*share:3.0f}% of bets fit inside a tenth of pre-start volume -> net about {base['net']-(1-share):+5.2f}c a contract ({base['net']:+.2f}c with no size)")
    summary["capacity"] = out


def shrink(cells):
    """Normal-normal shrinkage of cell net edges towards their precision-weighted mean (DerSimonian-Laird tau)."""
    names = [k for k, c in cells.items() if c["n"] >= 10]
    y, se = np.array([cells[k]["net"] for k in names]), np.array([cells[k]["se"] for k in names])
    w = 1/se**2
    mean = float((w*y).sum()/w.sum())
    q = float((w*(y-mean)**2).sum())
    tau2 = max(0.0, (q-(len(y)-1))/(w.sum()-(w**2).sum()/w.sum()))
    w2 = 1/(se**2+tau2)
    mean2 = float((w2*y).sum()/w2.sum())
    out = {k: dict(raw=float(v), se=float(s), shrunk=float(mean2+tau2/(tau2+s*s)*(v-mean2)) if tau2 else mean2) for k, v, s in zip(names, y, se)}
    return out, mean2, math.sqrt(tau2), float(1/math.sqrt(w2.sum()))


def forward(rows, fams, summary):
    say("\n6. HONEST FORWARD EXPECTATION")
    cells = {k: v["all"] for k, v in fams.items() if isinstance(v, dict) and "all" in v}
    shrunk, mean, tau, mean_se = shrink(cells)
    say(f"   every 90-99c cell: precision-weighted net after the fee {mean:+.2f}c (standard error {mean_se:.2f}c), spread between cells tau {tau:.2f}c")
    for name in SUPPORTED:
        s = shrunk[name]
        say(f"     {name[:34]:34} raw {s['raw']:+5.2f}c (se {s['se']:.2f}) -> shrunk {s['shrunk']:+5.2f}c")
    window, begin, end = window_of(rows)
    mix = Counter(r["cell"] for r in window)
    edge = sum(shrunk[k]["shrunk"]*n for k, n in mix.items())/sum(mix.values())
    raw = sum(shrunk[k]["raw"]*n for k, n in mix.items())/sum(mix.values())
    cost = float(np.mean([r["ask"]+fee_cents(r["ask"]) for r in window]))
    later = summary["baskets"]["(c) chosen on the earlier half"]["later"]
    say(f"   basket (b) at the mix of games in {begin}..{end}: raw {raw:+.2f}c = {100*raw/cost:+.2f}% a bet; shrunk {edge:+.2f}c = {100*edge/cost:+.2f}% a bet")
    say(f"   the out-of-sample check: cells chosen on the earlier half earned {later['net']:+.2f}c = {later['ret']:+.2f}% a bet afterwards {band(later['net_days'])} (n {later['n']})")
    days = defaultdict(list)
    for r in window:
        days[r["day"]].append(r["ask"])
    comps = [np.array(v) for v in days.values()]
    year = [r for r in in_band(rows) if r["cell"] in SUPPORTED]
    span = defaultdict(list)
    for r in year:
        span[r["day"]].append(r["ask"])
    say(f"   nights in the window: {len(comps)}, favourites a night median {np.median([len(c) for c in comps]):.0f} (range {min(len(c) for c in comps)}-{max(len(c) for c in comps)}); "
        f"outside the college season the only member is first-round qualifying: median {np.median([len(v) for d, v in span.items() if '2026-05' <= d[:7] <= '2026-09']):.0f} a night on the days it has any")
    out = dict(pooled_mean=mean, tau=tau, shrunk=shrunk, mix=dict(mix), edge_shrunk=edge, edge_raw=raw, cost=cost, window=[begin, end])
    drag = float(np.mean([r["ask"]-r["mid"]+fee_cents(r["ask"]) for r in window]))
    say(f"   if the midpoint were fair, the taker would lose the half-spread and the fee: {-drag:+.2f}c = {-100*drag/cost:+.2f}% a bet")
    scenarios = (("raw historical edge", raw), ("shrunk edge (central)", edge), ("half the shrunk edge", edge/2), ("zero after costs", 0.0),
                 ("midpoint is fair", -drag))

    def run(comps, edge_c, rule, nights=40000):
        res = np.empty(nights)
        pick = RNG.integers(0, len(comps), nights)
        for i, j in enumerate(pick):
            ask = comps[j]
            n = len(ask)
            stake = 1/n if rule == "equal" else min(0.10, 1/n)
            costs = ask+7*ask/100*(1-ask/100)
            win = RNG.random(n) < np.minimum(0.9995, (costs+edge_c)/100)
            res[i] = (stake*(100*win-costs)/costs).sum()
        return res

    for label, comps_, tag in ((f"college season, nights like {begin}..{end}", comps, "season"),
                               ("January to March 2026 as sampled (57% of men's games, few women's: real nights are bigger)",
                                [np.array(v) for d, v in span.items() if "2026-01" <= d[:7] <= "2026-03"], "conference"),
                               ("May to September, first-round qualifying only", [np.array(v) for d, v in span.items() if "2026-05" <= d[:7] <= "2026-09"], "summer")):
        if not comps_:
            continue
        say(f"   {label} ({len(comps_)} nights resampled, 40,000 simulated):")
        for rule in ("equal", "fixed"):
            for name, e in scenarios:
                r = run(comps_, e, rule)
                weeks, months = r[:39998].reshape(-1, 7).sum(1), r[:39990].reshape(-1, 30).sum(1)
                need = ((1.645+0.8416)*r.std()/r.mean())**2 if r.mean() > 0 else float("inf")
                o = dict(mean=float(100*r.mean()), median=float(100*np.median(r)), sd=float(100*r.std()), losing_day=float((r < 0).mean()), losing_week=float((weeks < 0).mean()),
                         losing_month=float((months < 0).mean()), month_mean=float(100*months.mean()), month_p05=float(100*np.percentile(months, 5)),
                         month_p01=float(100*np.percentile(months, 1)), day_p01=float(100*np.percentile(r, 1)), worst_day=float(100*r.min()), nights_for_power=float(need))
                out[f"{tag} | {rule} | {name}"] = o
                say(f'     {"equal split" if rule == "equal" else "10% a game ":11} {name:22} day {o["mean"]:+6.3f}% (median {o["median"]:+5.2f}%, sd {o["sd"]:4.2f}%) month {o["month_mean"]:+5.1f}% | '
                    f'losing day {100*o["losing_day"]:3.0f}% week {100*o["losing_week"]:3.0f}% month {100*o["losing_month"]:3.0f}% | bad month (1 in 20) {o["month_p05"]:+5.1f}%, (1 in 100) {o["month_p01"]:+5.1f}% '
                    f'| 1-in-100 day {o["day_p01"]:+5.1f}% | nights for 80% power {o["nights_for_power"]:,.0f}')
    sd_bet = float(np.std([net_cents(r) for r in year]))
    out["sd_per_bet"] = sd_bet
    say(f"   in bets: standard deviation {sd_bet:.1f}c a bet, so 80% power (5% one-sided) needs " +
        ", ".join(f"{((1.645+0.8416)*sd_bet/e)**2:,.0f} bets for {e:.1f}c" for e in (0.5, 1.0, 1.5, 2.0)))
    summary["forward"] = out


def polymarket_extra(summary):
    path = PROC/"nba-backfill/rows.json"
    if not path.exists():
        return
    rows = [r for r in json.loads(path.read_text()) if not r.get("kalshi_era") and r.get("phase") == "regular" and r.get("at_start") is not None
            and r.get("payout") is not None]
    say("\nEXTRA (kept apart: another venue, midpoints only, no ask): NBA regular season on Polymarket before Kalshi listed the NBA")
    out = {}
    for label, lo, hi in (("90-99c", 90, 99), ("85-90c control", 85, 90)):
        part = [r for r in rows if lo <= r["at_start"] < hi]
        if not part:
            continue
        mid, pay = np.array([r["at_start"] for r in part]), np.array([r["payout"] for r in part])
        ask = mid+0.5
        net = 100*pay-ask-7*ask/100*(1-ask/100)
        lo_, hi_ = wilson(pay.sum(), len(part))
        out[label] = dict(n=len(part), mid=float(mid.mean()), won=float(pay.mean()), gap=float(100*pay.mean()-mid.mean()), net=float(net.mean()),
                          gap_wilson=[100*lo_-float(mid.mean()), 100*hi_-float(mid.mean())], first=min(r["day"] for r in part), last=max(r["day"] for r in part))
        say(f'   midpoint at the start {label}: n {len(part)} ({out[label]["first"]}..{out[label]["last"]}) price {mid.mean():.1f} won {100*pay.mean():.1f}% gap {out[label]["gap"]:+.2f}c '
            f'{band(out[label]["gap_wilson"])}; with a 1c spread and Kalshi\'s fee {net.mean():+.2f}c')
    summary["polymarket_nba"] = out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, counts = load()
    (OUT/"table.json").write_text(json.dumps(rows))
    summary = {}
    say("NIGHTLY BASKET OF PRE-EVENT FAVOURITES AT 90-99c   (research/favourite_basket.py; simulation on historical bars, nothing traded)\n")
    coverage(rows, counts, summary)
    fee_rounding(rows, summary)
    fams = families(rows, summary)
    members, chosen, split = baskets(rows, summary)
    simulations(rows, members, summary, split)
    alternatives(rows, summary)
    capacity(rows, summary)
    forward(rows, fams, summary)
    polymarket_extra(summary)
    (OUT/"summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (OUT/"report.txt").write_text(out_buffer.getvalue())


if __name__ == "__main__":
    main()
