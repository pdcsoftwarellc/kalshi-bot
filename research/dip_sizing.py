"""How should an account size the favourite-dip bids: per family, per bet and per night?

The rule being sized (nothing re-derived): a 90-99c pre-game favourite gets a
resting bid at 80c once the game is live, held to settlement; a fill needs a
print strictly below the bid. research/dip_capacity.py measured the contracts
printing through each bid; this script reuses its orders, fills and accounting
(read from its cached data/processed/dip-capacity/rows.json, rebuilt through
its own loaders if the file is missing) and answers five sizing questions:

  1. one order size per family (men's / women's college basketball, tennis
     first-round qualifying) for a $5,000, $10,000 and $25,000 account, chosen
     on the first chronological half of the fills and reported on the second;
  2. the risk of the chosen sizes on the actual sequence of fills by night;
  3. fixed contracts against per-bet and per-night caps as a share of the
     account, by resampling whole nights over a 150-day season;
  4. whether lower rungs (75, 70, 65, 60c) pay once their cash is charged;
  5. the sizing table for $10,000.

Inputs that are ASSUMPTIONS, not measurements
  * SHARE = 25%: a bid of Q contracts is filled for min(Q, 25% of the contracts
    printing strictly through it) - the capacity study's convention.
  * Fills per 30 days: college basketball as in November 2026 (to the 18th),
    tennis as in May-October 2026 (dip_capacity section 4b).
  * Every bid is funded with cash from the start of play to settlement, and the
    busiest moment in the sample (2 November 2026) is the busiest there will be.
  * Men's college basketball pays a maker fee of 1.75 x p x (1-p) cents a
    contract (0.28c at 80c); the other two families pay none.
  * In the season simulation a fill happens in full at the first print through
    the bid, and one night's bids overlap completely (their peaks are added).

    .venv/bin/python research/dip_sizing.py

No network, no orders. Output: data/processed/dip-sizing/.
"""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
import io
import itertools
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import dip_capacity as dc

OUT = dc.ROOT / "data/processed/dip-sizing"
KEYS = ("men", "women", "tennis")
NAMES = dict(dc.GROUPS)
SHARE = dc.SHARE
MAKER = dict(men=1.75, women=0.0, tennis=0.0)             # cents x p x (1-p) per contract
ACCOUNTS = (5000, 10000, 25000)
GRID = (0, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000)
LADDERS = ((80,), (80, 75), (80, 70), (80, 65), (80, 60), (80, 70, 60), (80, 75, 70, 65, 60))
LOWER = ((70,), (60,))          # one bid lower down INSTEAD of 80c, shown for reference
SEASON, DRAWS = 150, 4000
rng = np.random.default_rng(11)
buffer = io.StringIO()


def say(text=""):
    print(text, flush=True)
    buffer.write(text+"\n")


def load():
    file = dc.OUT / "rows.json"
    rows = json.loads(file.read_text()) if file.exists() else [dc.measure(o) for o in dc.orders()]
    for o in rows:
        o["levels"] = {int(k): v for k, v in o["levels"].items()}
        o["month"] = o["day"][:7]
    return sorted(rows, key=lambda o: (o["settle"], o["event"]))


def fee(key, level=80):
    return MAKER[key]*level/100*(1-level/100)


def game_dollars(o, size, ladder=(80,), share=SHARE, fees=True, banded=False):
    """(dollars paid, dollars earned) for bids of `size` contracts at each rung of `ladder` on one game. Each contract
    printed is counted once and goes to our HIGHEST bid above it that still has room (price priority), so the top rung
    fills exactly as it does alone: min(size, share x everything printed under it). banded=True is the capacity
    study's convention instead: a rung only gets the prints between itself and the next rung down."""
    room = [float(size)]*len(ladder)
    for i, level in enumerate(ladder):
        volume = share*(o["levels"][level]["through"]-(o["levels"][ladder[i+1]]["through"] if i+1 < len(ladder) else 0))
        for j in ([i] if banded else range(i+1)):
            take = min(room[j], volume)
            room[j] -= take
            volume -= take
    paid = earned = 0.0
    for level, left in zip(ladder, room):
        got = size-left
        paid += got*level/100
        earned += got*(100*o["payout"]-level-(fee(o["group"], level) if fees else 0))/100
    return paid, earned


def halves(part):
    part = sorted(part, key=lambda o: (o["settle"], o["event"]))
    return part[:len(part)//2], part[len(part)//2:]


def frontier(rows):
    """Counts (men, women, tennis) of orders resting at once, over the whole sample, undominated ones only."""
    marks = sorted([(o["start"], 1, o["group"]) for o in rows]+[(o["settle"], -1, o["group"]) for o in rows], key=lambda m: (m[0], m[1]))
    now, seen = Counter(), set()
    for _, step, key in marks:
        now[key] += step
        seen.add(tuple(now[k] for k in KEYS))
    keep = [t for t in seen if not any(u != t and all(a >= b for a, b in zip(u, t)) for u in seen)]
    return np.array(sorted(keep), float)


def reserve(sizes, triples, weights=(.8, .8, .8)):
    """Cash needed at the busiest moment: sizes may be arrays (broadcast)."""
    return np.max([sum(t[i]*sizes[i]*weights[i] for i in range(3)) for t in triples], axis=0)


def frequency(rows):
    freq = {}
    for key in KEYS:
        part = [o for o in rows if o["group"] == key]
        use = [m for m in sorted({o["month"] for o in part}) if "2026-05" <= m <= "2026-10"] if key == "tennis" else ["2026-11"]
        scale = 1/len(use) if key == "tennis" else 30/dc.days_in(part, "2026-11")
        sub = [o for o in part if o["month"] in use]
        freq[key] = dict(orders=len(sub)*scale, fills=len(dc.fills(sub, 80))*scale, peak=max(dc.peak([o for o in sub if o["month"] == m])[0] for m in use))
    return freq


def per_fill(part, size, ladder=(80,), fees=True):
    return np.array([game_dollars(o, size, ladder, fees=fees)[1] for o in part])


def monthly(parts, sizes, freq, fees=True):
    return {key: freq[key]["fills"]*per_fill(parts[key], size, fees=fees).mean() for key, size in zip(KEYS, sizes)}


def clustered(parts, freq, plans, draws=DRAWS):
    """Game-clustered bootstrap of dollars a month for several plans (lists of sizes) on the SAME resampled games."""
    out = np.zeros((len(plans), draws))
    for i, key in enumerate(KEYS):
        pick = rng.integers(0, len(parts[key]), (draws, len(parts[key])))
        for j, plan in enumerate(plans):
            out[j] += freq[key]["fills"]*per_fill(parts[key], plan[i])[pick].mean(axis=1)
    return out


def span(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return f"[{lo:+8,.0f},{hi:+8,.0f}]"


cache = {}


def best(parts, freq, account, triples, ladders=((80,),)*3, grid=GRID, cap=None, keep=12):
    """Grid search: sizes maximising dollars a month with the busiest moment funded and, if cap is given, no bid's
    cash above that share of the account. Returns (sizes, dollars, reserve) sorted best first."""
    table = []
    for i, key in enumerate(KEYS):
        name = (id(parts[key]), ladders[i])
        if name not in cache:
            cache[name] = np.array([freq[key]["fills"]*per_fill(parts[key], q, ladders[i]).mean() for q in grid])
        table.append(cache[name])
    a, b, c = np.meshgrid(np.arange(len(grid)), np.arange(len(grid)), np.arange(len(grid)), indexing="ij")
    sizes = [np.array(grid, float)[x] for x in (a, b, c)]
    need = reserve(sizes, triples, [sum(l)/100 for l in ladders])
    total = table[0][a]+table[1][b]+table[2][c]
    total = np.where(need <= account, total, -np.inf)
    if cap is not None:
        for s, ladder in zip(sizes, ladders):
            total = np.where(s*sum(ladder)/100 <= cap*account+1e-9, total, -np.inf)
    order = np.argsort(-total, axis=None)[:keep]
    return [(tuple(int(s.flat[i]) for s in sizes), float(total.flat[i]), float(need.flat[i])) for i in order if np.isfinite(total.flat[i])]


# ---------------------------------------------------------------- 1. sizes by family

def part_one(rows, hits, freq, triples):
    say("=== 1. ORDER SIZE BY FAMILY (80c bid; fill = min(Q, 25% of contracts through); men's maker fee charged)")
    top = int(triples.sum(axis=1).max())
    say(f"  Orders resting at once, (men, women, tennis), undominated moments of the whole sample: {[tuple(int(v) for v in t) for t in triples]}; most at once {top}")
    say(f"  Fills per 30 days assumed: "+", ".join(f"{key} {freq[key]['fills']:.0f}" for key in KEYS))
    one = 10000/(.8*top)
    for fees in (False, True):
        m = monthly(hits, (one,)*3, freq, fees)
        say(f"  One size for all, $10,000, Q {one:.0f}, {'with' if fees else 'without'} the men's fee: "+" ".join(f"{k} ${v:+,.0f}" for k, v in m.items())+f" = ${sum(m.values()):+,.0f} a month"
            + ("   (the capacity study's baseline)" if not fees else ""))
    say("\n  1a. Return on each extra slice of size, % of the money the slice puts in (whole sample | first half | second half)")
    first = {key: halves(hits[key])[0] for key in KEYS}
    second = {key: halves(hits[key])[1] for key in KEYS}
    for key in KEYS:
        say(f"    {NAMES[key]}")
        for lo, hi in zip((0, 50, 100, 200, 300, 500, 1000, 2000), (50, 100, 200, 300, 500, 1000, 2000, 3000)):
            text = f"      {lo:5,}-{hi:5,} contracts:"
            for part in (hits[key], first[key], second[key]):
                paid = sum(game_dollars(o, hi)[0]-game_dollars(o, lo)[0] for o in part)
                earned = sum(game_dollars(o, hi)[1]-game_dollars(o, lo)[1] for o in part)
                text += f" {100*earned/max(paid, 1e-9):+6.1f}%"
            say(text)
    say("\n  1b. Sizes chosen on the FIRST half of each family's fills (round sizes only), valued on the second half and on everything.")
    say("      'best' = most dollars a month on the first half; 'robust' = within 2% of that with the least money put in (drops slices that earn next to nothing).")
    say("      'bid cap' = no single bid's cash above that share of the account. Hindsight rows show what the other half or the whole sample would have picked.")
    chosen = {}
    for account in ACCOUNTS:
        one = account/(.8*top)
        say(f"    account ${account:,}: one size for all would be Q {one:.0f}")
        def show(label, plan):
            text = f"      {label:34} men {plan[0]:5,} women {plan[1]:5,} tennis {plan[2]:5,} reserve ${float(reserve(plan, triples)):7,.0f}"
            for name, parts in (("1st", first), ("2nd", second), ("all", hits)):
                value = sum(monthly(parts, plan, freq).values())
                base = sum(monthly(parts, (one,)*3, freq).values())
                text += f" | {name} ${value:+7,.0f} = {100*value/account:+5.1f}% (one size ${base:+7,.0f})"
            say(text)
        for cap in (None, .05, .03):
            ranked = best(first, freq, account, triples, cap=cap, keep=4000)
            near = [r for r in ranked if r[1] >= .98*ranked[0][1]]
            robust = min(near, key=lambda r: sum(freq[k]["fills"]*q for k, q in zip(KEYS, r[0])))
            name = "no bid cap" if cap is None else f"bid cap {100*cap:.0f}%"
            show(f"{name}: best on first half", ranked[0][0])
            show(f"{name}: robust", robust[0])
            chosen[(account, cap)] = robust[0]
        show("hindsight, second half, no cap", best(second, freq, account, triples)[0][0])
        show("hindsight, everything, no cap", best(hits, freq, account, triples)[0][0])
        show("hindsight, everything, cap 5%", best(hits, freq, account, triples, cap=.05)[0][0])
    return chosen, first, second


def part_one_final(hits, first, second, freq, triples, plans):
    say(f"\n  1c. THE SIZES CARRIED FORWARD: the robust first-half choice with no bid above {100*BID_CAP:.0f}% of the account (why a cap: sections 2 and 3)")
    top = int(triples.sum(axis=1).max())
    out = {}
    for account, plan in plans.items():
        one = account/(.8*top)
        say(f"    account ${account:,}: men {plan[0]:,} women {plan[1]:,} tennis {plan[2]:,} contracts; dollars per bid ${.8*plan[0]:,.0f} / ${.8*plan[1]:,.0f} / ${.8*plan[2]:,.0f} "
            f"= {80*plan[0]/account:.1f}% / {80*plan[1]/account:.1f}% / {80*plan[2]/account:.1f}% of the account; reserve at the busiest moment ${float(reserve(plan, triples)):,.0f}")
        for name, parts in (("first half (chosen here)", first), ("second half (out of sample)", second), ("whole sample", hits)):
            d = clustered(parts, freq, [plan, (one,)*3])
            m = monthly(parts, plan, freq)
            value = sum(m.values())
            say(f"      {name:28} ${value:+7,.0f} a month = {100*value/account:+5.1f}% {span(d[0])} = [{100*np.percentile(d[0], 2.5)/account:+5.1f}%,{100*np.percentile(d[0], 97.5)/account:+5.1f}%] "
                f"| men ${m['men']:+6,.0f} women ${m['women']:+6,.0f} tennis ${m['tennis']:+6,.0f} | gain over one size ${np.mean(d[0]-d[1]):+6,.0f} {span(d[0]-d[1])}, "
                f"better in {100*np.mean(d[0] > d[1]):.0f}% of draws; P(month's expectation < 0) {100*np.mean(d[0] < 0):.0f}%")
            if name == "whole sample":
                out[account] = dict(sizes=plan, month=value, low=float(np.percentile(d[0], 2.5)), high=float(np.percentile(d[0], 97.5)))
    return out


# ---------------------------------------------------------------- 2. risk of the chosen sizes

def drawdown(values):
    total = np.cumsum(values)
    return float(np.max(np.maximum.accumulate(np.maximum(total, 0))-total)) if len(total) else 0.0


def longest(flags):
    best_run = run = 0
    for f in flags:
        run = run+1 if f else 0
        best_run = max(best_run, run)
    return best_run


def part_two(hits, plan, account, brief=False):
    if brief:
        say(f"  2c. For comparison, the sizes with NO bid cap (men {plan[0]}, women {plan[1]}, tennis {plan[2]}) on the same sequence")
    else:
        say(f"\n=== 2. RISK OF THE CHOSEN SIZES ON THE ACTUAL SEQUENCE (account ${account:,}; men {plan[0]}, women {plan[1]}, tennis {plan[2]} contracts; no compounding)")
        say("    The history is as sampled: men's 2025-26 is a 57% sample of games and both basketball seasons were quieter than November 2026, so dollars per night here understate today's.")
    size = dict(zip(KEYS, plan))
    every = sorted((o for key in KEYS for o in hits[key]), key=lambda o: (o["settle"], o["event"]))
    for o in every:
        o["pnl"] = game_dollars(o, size[o["group"]])[1]
        o["lost"] = o["payout"] < .5
    def daily(part):
        days = defaultdict(float)
        for o in part:
            days[o["day"]] += o["pnl"]
        return days
    def weekly(days):
        weeks = defaultdict(float)
        for d, v in days.items():
            y, w, _ = date.fromisoformat(d).isocalendar()
            weeks[(y, w)] += v
        return weeks
    spans = (("whole sample", lambda o: True), ("November 2026 only (today's frequency, 18 days)", lambda o: o["month"] == "2026-11"))
    for label, test in spans:
        part = [o for o in every if test(o)]
        days = daily(part)
        d = np.array([days[k] for k in sorted(days)])
        w = np.array(list(weekly(days).values()))
        losses = Counter(o["day"] for o in part if o["lost"])
        count = Counter(o["day"] for o in part)
        say(f"  {label}: fills {len(part)}, losses {sum(o['lost'] for o in part)}, nights with a fill {len(d)}, weeks {len(w)}, total ${d.sum():+,.0f} = {100*d.sum()/account:+.1f}% of the account")
        for name, v in (("night", d), ("week", w)):
            q = np.percentile(v, [5, 25, 50, 75, 95])
            say(f"    per {name:5}: mean ${v.mean():+7,.0f} sd ${v.std():6,.0f} | 5/25/50/75/95th ${q[0]:+7,.0f} ${q[1]:+6,.0f} ${q[2]:+6,.0f} ${q[3]:+6,.0f} ${q[4]:+6,.0f} | "
                f"worst ${v.min():+7,.0f} = {100*v.min()/account:+5.1f}% | best ${v.max():+6,.0f} | losing {100*np.mean(v < 0):.0f}%")
        worst = sorted(days, key=days.get)[:3]
        say("    worst nights: "+"; ".join(f"{k} ${days[k]:+,.0f} ({losses[k]} lost of {count[k]})" for k in worst))
        dd = drawdown(d)
        say(f"    maximum drawdown ${dd:,.0f} = {100*dd/account:.1f}% of the account | longest run of losing fills {longest(o['lost'] for o in part)} | "
            f"longest run of losing nights {longest(v < 0 for v in d)} | longest run of losing weeks {longest(v < 0 for v in w)}")
        say(f"    nights with 2+ losses: {sum(v >= 2 for v in losses.values())} of {len(d)} = {100*sum(v >= 2 for v in losses.values())/len(d):.1f}% "
            f"(of the {sum(v >= 2 for v in count.values())} nights with 2+ fills: {100*sum(v >= 2 for v in losses.values())/max(sum(v >= 2 for v in count.values()), 1):.1f}%); "
            f"most losses in one night {max(losses.values(), default=0)}; most fills in one night {max(count.values())}")
    if brief:
        return
    say("  2b. Are same-night outcomes correlated? Outcomes (with their size) shuffled among each family's fills, nights kept; 5,000 shuffles")
    days = sorted({o["day"] for o in every})
    index = {d: i for i, d in enumerate(days)}
    where = np.array([index[o["day"]] for o in every])
    def stats(lost, pnl):
        per = np.bincount(where, weights=lost, minlength=len(days))
        money = np.bincount(where, weights=pnl, minlength=len(days))
        return np.mean(per >= 2), per.var(), money.min(), money.std(), drawdown(money)
    lost = np.array([o["lost"] for o in every], float)
    pnl = np.array([o["pnl"] for o in every])
    actual = stats(lost, pnl)
    groups = [np.array([i for i, o in enumerate(every) if o["group"] == key]) for key in KEYS]
    sims = []
    for _ in range(5000):
        order = np.arange(len(every))
        for g in groups:
            order[g] = rng.permutation(g)
        sims.append(stats(lost[order], pnl[order]))
    sims = np.array(sims)
    for i, (name, unit, worse) in enumerate((("share of nights with 2+ losses", "%", 1), ("variance of losses per night", "", 1), ("worst night", "$", -1),
                                             ("sd of nightly dollars", "$", 1), ("maximum drawdown", "$", 1))):
        scale = 100 if unit == "%" else 1
        p = np.mean(worse*sims[:, i] >= worse*actual[i])
        say(f"    {name:32} actual {scale*actual[i]:9,.2f} | shuffled mean {scale*sims[:, i].mean():9,.2f}, 2.5-97.5% {scale*np.percentile(sims[:, i], 2.5):9,.2f} to {scale*np.percentile(sims[:, i], 97.5):9,.2f} | "
            f"share of shuffles at least as bad {100*p:.0f}%")
    for key in KEYS:
        part = hits[key]
        per = Counter(o["day"] for o in part if o["lost"])
        n = Counter(o["day"] for o in part)
        rate = np.mean([o["lost"] for o in part])
        expect = sum(1-(1-rate)**k-k*rate*(1-rate)**(k-1) for k in n.values())
        say(f"    {NAMES[key]:32} loss rate {100*rate:.1f}%; nights with 2+ losses {sum(v >= 2 for v in per.values())} against {expect:.1f} expected if independent")


# ---------------------------------------------------------------- 3. sizing rules

def night_pool(rows, hits, freq, size=30000):
    """Synthetic nights at today's frequency. Each family's history is cut into eras; a quiet era's nights are merged
    (Poisson number of whole nights) until its fills per day match the assumed frequency. Whole nights stay together."""
    eras = {}
    for key in KEYS:
        part = [o for o in rows if o["group"] == key]
        cuts = [("old", lambda o: o["month"] < "2026-11"), ("new", lambda o: o["month"] == "2026-11")] if key != "tennis" else [("all", lambda o: True)]
        target = freq[key]["fills"]/30
        eras[key] = []
        for name, test in cuts:
            sub = [o for o in part if test(o)]
            first, last = date.fromisoformat(min(o["day"] for o in sub)), date.fromisoformat(max(o["day"] for o in sub))
            days = [(first+timedelta(days=i)).isoformat() for i in range((last-first).days+1)]
            got = [o for o in hits[key] if test(o)]
            rate = len(got)/len(days)
            merge = 1.0 if key == "tennis" or name == "new" else target/rate
            nights = []
            for d in days:
                orders = [o for o in sub if o["day"] == d]
                fills = [o for o in got if o["day"] == d]
                nights.append((dc.peak(orders)[0] if orders else 0, fills))
            eras[key].append(dict(name=name, merge=merge, nights=nights, weight=len(days)/merge, rate=rate))
    pool = []
    for _ in range(size):
        peaks, fills = [0, 0, 0], []
        for i, key in enumerate(KEYS):
            weights = np.array([e["weight"] for e in eras[key]])
            era = eras[key][rng.choice(len(weights), p=weights/weights.sum())]
            for _ in range(1 if era["merge"] == 1 else rng.poisson(era["merge"])):
                top, got = era["nights"][rng.integers(len(era["nights"]))]
                peaks[i] += top
                fills += got
        fills = sorted(fills, key=lambda o: (datetime.fromtimestamp(o["levels"][80]["t0"], dc.EASTERN)-timedelta(hours=5)).time())
        pool.append((peaks, fills))
    width = max(len(f) for _, f in pool)
    arrays = dict(peaks=np.array([p for p, _ in pool], float), family=np.zeros((size, width), int), cap=np.zeros((size, width)), net=np.zeros((size, width)),
                  count=np.array([len(f) for _, f in pool]), lost=np.array([sum(o["payout"] < .5 for o in f) for _, f in pool]))
    for n, (_, fills) in enumerate(pool):
        for j, o in enumerate(fills):
            arrays["family"][n, j] = KEYS.index(o["group"])
            arrays["cap"][n, j] = SHARE*o["levels"][80]["through"]
            arrays["net"][n, j] = (100*o["payout"]-80-fee(o["group"]))/100
    return arrays, eras


def season(pool, picks, start, fixed=None, bet=None, night=None, ceiling=(3000, 300, 3000)):
    """Run every resampled season at once. fixed = contracts by family; bet = share of the current account per bid;
    night = share of the current account that may be filled in one night (later fills are refused)."""
    sims = picks.shape[0]
    account = np.full(sims, float(start))
    high, deepest, scaled, used = account.copy(), np.zeros(sims), 0.0, 0.0
    rows = np.arange(sims)[:, None]
    for t in range(picks.shape[1]):
        n = picks[:, t]
        if fixed is not None:
            sizes = np.tile(np.array(fixed, float), (sims, 1))
        else:
            sizes = np.minimum(bet*np.maximum(account, 0)[:, None]/.8, np.array(ceiling, float)[None, :])
        need = .8*(sizes*pool["peaks"][n]).sum(axis=1)
        scale = np.minimum(1, np.maximum(account, 0)/np.maximum(need, 1e-9))
        scaled += np.mean(scale < 1)
        sizes = sizes*scale[:, None]
        got = np.minimum(sizes[rows, pool["family"][n]], pool["cap"][n])
        if night is not None:
            cost = .8*got
            before = np.cumsum(cost, axis=1)-cost
            got = np.clip(night*np.maximum(account, 0)[:, None]-before, 0, cost)/.8
        used += np.mean(.8*got.sum(axis=1)/np.maximum(account, 1e-9))
        account = account+(got*pool["net"][n]).sum(axis=1)
        high = np.maximum(high, account)
        deepest = np.maximum(deepest, (high-account)/high)
    return account, deepest, scaled/picks.shape[1], used/picks.shape[1]


def kelly_fraction(p, win, loss):
    return (p*win-(1-p)*loss)/(win*loss)


def night_kelly(counts, p, win=.25, loss=1.0):
    """Share of the account per bid that maximises expected log growth when a night has `counts` simultaneous bets."""
    tally = Counter(int(c) for c in counts if c)
    grid = np.linspace(0, .99/(max(tally)*loss), 400)
    growth = np.zeros_like(grid)
    for k, weight in tally.items():
        for lost in range(k+1):
            growth += weight*comb(k, lost)*(1-p)**lost*p**(k-lost)*np.log1p(grid*(win*(k-lost)-loss*lost))
    return float(grid[np.argmax(growth)])


def part_three(rows, hits, freq, plans, account, ceiling):
    say(f"\n=== 3. SIZING RULES over a {SEASON}-day season, starting at ${account:,}: whole nights resampled, {DRAWS:,} seasons")
    pool, eras = night_pool(rows, hits, freq)
    for key in KEYS:
        say(f"  {NAMES[key]:32} "+" | ".join(f"era {e['name']}: {len(e['nights'])} days, {30*e['rate']:.0f} fills per 30 days, nights merged per season night {e['merge']:.1f}" for e in eras[key]))
    made = [30*np.mean((pool["family"] == i) & (pool["cap"] > 0)) * pool["family"].shape[1] for i in range(3)]
    say(f"  the simulated nights: fills per 30 days men {made[0]:.0f} women {made[1]:.0f} tennis {made[2]:.0f} (assumed {freq['men']['fills']:.0f} / {freq['women']['fills']:.0f} / {freq['tennis']['fills']:.0f}); "
        f"fills per night mean {pool['count'].mean():.1f}, 95th {np.percentile(pool['count'], 95):.0f}, most {pool['count'].max()}; nights with 2+ losses {100*np.mean(pool['lost'] >= 2):.1f}%, most losses {pool['lost'].max()}")
    total = pool["peaks"].sum(axis=1)
    say(f"  bids resting at once per night (peaks added across families and merged nights - an overstatement): median {np.median(total):.0f}, 95th {np.percentile(total, 95):.0f}, most {total.max():.0f}")
    say(f"  per-bet rules: contracts = share x account / 80c, never more than {ceiling[0]:,} men / {ceiling[1]:,} women / {ceiling[2]:,} tennis (where the extra slice stops paying, section 1a),")
    say("  cut pro rata on a night whose bids together need more cash than the account holds; each fill still limited to 25% of the contracts through (the measured decay); men's fee charged.")
    say("  night cap: bids are cancelled once the night's FILLED dollars reach the cap (fills taken in clock order).")
    picks = rng.integers(0, len(pool["count"]), (DRAWS, SEASON))
    plan = plans[BID_CAP]
    rules = [("(a) fixed contracts "+"/".join(str(p) for p in q), dict(fixed=q)) for q in dict.fromkeys(plans.values())]
    rules += [(f"(b) each bid {100*f:.0f}% of account", dict(bet=f)) for f in (.01, .02, .03, .05)]
    rules += [(f"(c) bid {100*f:.0f}%, night cap {100*c:.0f}%", dict(bet=f, night=c)) for f in (.02, .03, .05) for c in (.10, .20, .30)]
    rules += [(f"(a') fixed contracts, night cap {100*c:.0f}%", dict(fixed=plan, night=c)) for c in (.10, .20)]
    say("  'a month' = median gain over the season / 5, % of the starting account (compounding rules grow, so their later months are larger in dollars).")
    say(f"  {'rule':36} {'median end':>11} {'5th':>9} {'95th':>9} {'a month':>12} {'P(end<start)':>12} {'P(25% drawdown)':>15} {'median max dd':>13} {'95th max dd':>11} {'nights cut for cash':>19} {'filled/night':>12}")
    table = {}
    for name, args in rules:
        end, deep, cut, used = season(pool, picks, account, ceiling=ceiling, **args)
        q = np.percentile(end, [5, 50, 95])
        monthly_rate = 100*(q[1]/account-1)*30/SEASON
        say(f"  {name:36} ${q[1]:10,.0f} ${q[0]:8,.0f} ${q[2]:8,.0f} {monthly_rate:+11.1f}% {100*np.mean(end < account):11.1f}% {100*np.mean(deep >= .25):14.1f}% {100*np.median(deep):12.1f}% {100*np.percentile(deep, 95):10.1f}% "
            f"{100*cut:18.1f}% {100*used:11.1f}%")
        table[name] = dict(median=float(q[1]), p5=float(q[0]), p95=float(q[2]), below=float(np.mean(end < account)), dd25=float(np.mean(deep >= .25)), dd_median=float(np.median(deep)), dd95=float(np.percentile(deep, 95)))

    say("\n  3b. What Kelly says. One bid at 80c wins 20c (less the fee) or loses 80c: the growth-optimal share of the account is p - (1-p) x loss/win.")
    say("      Break-even win rate 80.0% (80.3% with the men's fee). Intervals: game-clustered bootstrap, 2.5-97.5%.")
    everything = [o for key in KEYS for o in hits[key]]
    for label, part, key in [(NAMES[k], hits[k], k) for k in KEYS]+[("all three together", everything, None)]:
        wins = np.array([o["payout"] for o in part])
        cost = fee(key) if key else np.mean([fee(o["group"]) for o in part])
        win, loss = (20-cost)/80, (80+cost)/80
        boots = wins[rng.integers(0, len(wins), (DRAWS, len(wins)))].mean(axis=1)
        k = kelly_fraction(boots, win, loss)
        text = f"    {label:32} fills {len(part):3} win rate {100*wins.mean():.1f}% [{100*np.percentile(boots, 2.5):.1f},{100*np.percentile(boots, 97.5):.1f}] | full Kelly {100*kelly_fraction(wins.mean(), win, loss):5.1f}% of the account per bid "
        text += f"[{100*np.percentile(k, 2.5):+6.1f}%,{100*np.percentile(k, 97.5):+5.1f}%], P(no edge at all) {100*np.mean(k <= 0):.1f}%"
        if key:
            size = plan[KEYS.index(key)]
            got = np.array([min(size, SHARE*o["levels"][80]["through"]) for o in part])
            weighted = (got*wins).sum()/got.sum()
            pick = rng.integers(0, len(wins), (DRAWS, len(wins)))
            kw = kelly_fraction((got*wins)[pick].sum(axis=1)/got[pick].sum(axis=1), win, loss)
            text += (f" | at {size} contracts the contract-weighted win rate is {100*weighted:.1f}%: Kelly {100*kelly_fraction(weighted, win, loss):.1f}% "
                     f"[{100*np.percentile(kw, 2.5):+.1f}%,{100*np.percentile(kw, 97.5):+.1f}%], P(no edge) {100*np.mean(kw <= 0):.0f}%")
        say(text)
    wins = np.array([o["payout"] for o in everything])
    boots = wins[rng.integers(0, len(wins), (DRAWS, len(wins)))].mean(axis=1)
    got = np.array([min(plan[KEYS.index(o["group"])], SHARE*o["levels"][80]["through"]) for o in everything])
    rates = (("bottom of the interval", np.percentile(boots, 2.5)), ("contract-weighted, chosen sizes", (got*wins).sum()/got.sum()), ("point estimate", wins.mean()), ("top of the interval", np.percentile(boots, 97.5)))
    shares = (.01, .02, .03, .05, .08, .12, .20, .30)
    say("      Several bids are live in one night and cash limits the bid to account / (bids resting at once). Expected log growth per NIGHT, in % of the account,")
    say("      for a bid of f% of the account on the simulated nights (fills per night as simulated, outcomes independent, no size decay): Kelly is where the row peaks.")
    say(f"      {'win rate':44}"+"".join(f"{'f '+format(100*f, '.0f')+'%':>8}" for f in shares)+"   peak at")
    tally = Counter((int(k), int(max(n, k, 1))) for k, n in zip(pool["count"], pool["peaks"].sum(axis=1)))
    nightly = {}
    for label, p in rates:
        grid = np.array(sorted(set(shares) | set(np.linspace(.005, .5, 100))))
        growth = np.zeros_like(grid)
        for (k, n), weight in tally.items():
            stake = np.minimum(grid, .99/n)
            for lost in range(k+1):
                growth += weight*comb(k, lost)*(1-p)**lost*p**(k-lost)*np.log1p(stake*(.25*(k-lost)-lost))
        growth /= sum(tally.values())
        nightly[label] = float(grid[np.argmax(growth)])
        say(f"      {format(100*p, '.1f')+'% ('+label+')':44}"+"".join(f"{100*growth[list(grid).index(f)]:+8.3f}" for f in shares)+f"   f {100*nightly[label]:.0f}%")
    say("      One bid at a time (tennis is close to this: 1-2 fills a night), growth per BET in % of the account:")
    for label, p in rates:
        say(f"      {format(100*p, '.1f')+'% ('+label+')':44}"+"".join(f"{100*(p*np.log1p(.25*f)+(1-p)*np.log1p(-f)):+8.3f}" for f in shares)+f"   f {100*max(kelly_fraction(p, .25, 1), 0):.0f}%")
    stake = [.8*q/account for q in plan]
    say(f"      The chosen sizes put {100*stake[0]:.1f}% / {100*stake[1]:.1f}% / {100*stake[2]:.1f}% of the account on a men's / women's / tennis bid.")
    return table, pool


# ---------------------------------------------------------------- 4. ladder

def part_four(rows, hits, freq, triples):
    say("\n=== 4. THE LADDER: Q contracts at each rung, cash reserved = Q x (sum of the rungs) on every resting order")
    say("    Each contract printed goes to our highest bid above it with room left, so the 80c rung fills exactly as it does alone and lower rungs only get what is left.")
    say("    Dollars a month at the assumed frequency, men's fee charged; the account may leave cash idle (Q is the best size at or under what the account funds).")
    say("    Whole sample, so every ladder figure is in-sample; 'halves' gives the gain on each chronological half.")
    for key in KEYS:
        text = f"    {NAMES[key]:32} per rung alone, small size (10 contracts), cents per contract and % on the rung:"
        for level in (80, 75, 70, 65, 60):
            got = [o for o in hits[key] if o["levels"][level]["filled"]]
            nets = np.array([100*o["payout"]-level-fee(key, level) for o in got])
            lo, hi = np.percentile(nets[rng.integers(0, len(nets), (2000, len(nets)))].mean(axis=1), [2.5, 97.5])
            text += f" {level}c n {len(got):3} {nets.mean():+5.1f}c [{lo:+5.1f},{hi:+5.1f}] = {100*nets.mean()/level:+5.1f}% |"
        say(text)
    say("  4a. Each family alone, reserve sized for that family's own busiest moment")
    summary = {}
    for key in KEYS:
        part, top, fills = hits[key], freq[key]["peak"], freq[key]["fills"]
        a, b = halves(part)
        say(f"  {NAMES[key]} (most resting at once {top})")
        def plan(account, ladder, games=part):
            limit = account/(top*sum(ladder)/100)
            sizes = [q for q in np.geomspace(10, 200000, 120) if q < limit]+[limit]
            values = [fills*np.mean([game_dollars(o, q, ladder)[1] for o in games]) for q in sizes]
            i = int(np.argmax(values))
            return sizes[i], values[i]
        for account in (10000, 25000, 50000, 100000, 250000, 1000000):
            base_q, base = plan(account, (80,))
            text = f"    account ${account:9,}: 80c alone Q {base_q:7,.0f} ${base:+8,.0f} = {100*base/account:+5.1f}%"
            options = []
            for ladder in LADDERS[1:]+LOWER:
                q, value = plan(account, ladder)
                if ladder not in LOWER:
                    options.append((value, q, ladder))
                text += f" | {'+'.join(str(l) for l in ladder)}{' only' if ladder in LOWER else ''} Q {q:6,.0f} ${value:+8,.0f}"
            value, q, ladder = max(options)
            pick = rng.integers(0, len(part), (2000, len(part)))
            one = np.array([game_dollars(o, base_q)[1] for o in part])
            two = np.array([game_dollars(o, q, ladder)[1] for o in part])
            gap = fills*(two[pick].mean(axis=1)-one[pick].mean(axis=1))
            parts = " / ".join(f"{fills*(np.mean([game_dollars(o, q, ladder)[1] for o in h])-np.mean([game_dollars(o, base_q)[1] for o in h])):+,.0f}" for h in (a, b))
            worst = min(game_dollars(o, q, ladder)[1] for o in part), min(game_dollars(o, base_q)[1] for o in part)
            parts += f"; worst single game ${worst[0]:,.0f} against ${worst[1]:,.0f}"
            say(text)
            say(f"{'':26}best ladder {'+'.join(str(l) for l in ladder)}: ${value-base:+8,.0f} a month against 80c alone {span(gap)}, better in {100*np.mean(gap > 0):.0f}% of draws; halves {parts}")
        cross = None
        for account in np.geomspace(1000, 3000000, 90):
            base = plan(account, (80,))[1]
            value = max(plan(account, ladder)[1] for ladder in LADDERS[1:])
            if value > base*1.02 and value > base+1:
                cross = cross or account
            else:
                cross = None
        summary[key] = cross
        say(f"    a ladder first beats 80c alone by 2% (and stays ahead) at an account of about {'$'+format(cross, ',.0f') if cross else 'never, up to $3,000,000'}")
    say("  4b. All three families in one account: best round sizes with and without a ladder (whole sample - in-sample, so the ladder's edge is flattered)")
    for account in (10000, 25000, 50000, 100000):
        plain = best(hits, freq, account, triples)[0]
        top_value, top_plan, top_ladders = -np.inf, None, None
        for ladders in itertools.product(LADDERS, repeat=3):
            found = best(hits, freq, account, triples, ladders)
            if found and found[0][1] > top_value:
                top_value, top_plan, top_ladders = found[0][1], found[0][0], ladders
        d = []
        for i, key in enumerate(KEYS):
            pick = rng.integers(0, len(hits[key]), (2000, len(hits[key])))
            d.append(freq[key]["fills"]*(per_fill(hits[key], top_plan[i], top_ladders[i])[pick].mean(axis=1)-per_fill(hits[key], plain[0][i])[pick].mean(axis=1)))
        gap = sum(d)
        say(f"    account ${account:7,}: 80c only {plain[0]} ${plain[1]:+8,.0f} = {100*plain[1]/account:+5.1f}% | best with ladders "
            + ", ".join(f"{key} {top_plan[i]}x{'+'.join(str(l) for l in top_ladders[i])}" for i, key in enumerate(KEYS))
            + f" ${top_value:+8,.0f} = {100*top_value/account:+5.1f}% | gain ${top_value-plain[1]:+7,.0f} {span(gap)}, better in {100*np.mean(gap > 0):.0f}% of draws")
    return summary


def part_four_c(hits, freq, plan, account):
    full = LADDERS[-1]
    say(f"  4c. The clean test at ${account:,}: nothing chosen. The carried sizes at 80c only (men {plan[0]}, women {plan[1]}, tennis {plan[2]}) against the SAME cash per order")
    say(f"      spread over all five rungs (sizes x 80/{sum(full)}), and against half the cash left at 80c with the other half on the four lower rungs.")
    every = sorted((o for key in KEYS for o in hits[key]), key=lambda o: (o["settle"], o["event"]))
    options = (("80c only", lambda q: [(q, (80,))]), ("five rungs, equal contracts", lambda q: [(q*80/sum(full), full)]),
               ("half at 80c + four lower rungs", lambda q: [(q/2, (80,)), (q*40/sum(full[1:]), full[1:])]))
    base = None
    for label, build in options:
        def value(o):
            return sum(game_dollars(o, q, ladder)[1] for q, ladder in build(plan[KEYS.index(o["group"])]))
        pnl = {key: np.array([value(o) for o in hits[key]]) for key in KEYS}
        base = base or pnl
        month = {key: freq[key]["fills"]*pnl[key].mean() for key in KEYS}
        local = np.random.default_rng(5)
        gap = sum(freq[key]["fills"]*(pnl[key]-base[key])[local.integers(0, len(pnl[key]), (DRAWS, len(pnl[key])))].mean(axis=1) for key in KEYS)
        half = [sum(freq[key]["fills"]*(pnl[key]-base[key])[[o in set(map(id, h)) for o in map(id, hits[key])]].mean() for key in KEYS for h in [halves(hits[key])[i]]) for i in (0, 1)]
        days = defaultdict(float)
        for o in every:
            days[o["day"]] += value(o)
        d = np.array([days[k] for k in sorted(days)])
        say(f"      {label:32} ${sum(month.values()):+7,.0f} a month = {100*sum(month.values())/account:+5.1f}% (men ${month['men']:+6,.0f} women ${month['women']:+5,.0f} tennis ${month['tennis']:+5,.0f}) | "
            f"gain over 80c only ${np.mean(gap):+6,.0f} {span(gap)}, better in {100*np.mean(gap > 0):3.0f}% of draws; halves {half[0]:+,.0f} / {half[1]:+,.0f} | "
            f"on the actual sequence: worst night ${d.min():+,.0f}, maximum drawdown ${drawdown(d):,.0f}, sd of a night ${d.std():,.0f}, worst single game ${min(value(o) for o in every):+,.0f}")


def part_five(rows, hits, freq, plan, account, sized, table):
    say(f"\n=== 5. BOTTOM LINE for ${account:,}")
    say(f"  contracts per 80c order: men {plan[0]}, women {plan[1]}, tennis {plan[2]} = ${.8*plan[0]:,.0f} / ${.8*plan[1]:,.0f} / ${.8*plan[2]:,.0f} a bid = {80*plan[0]/account:.1f}% / {80*plan[1]/account:.1f}% / {80*plan[2]/account:.1f}% of the account")
    r = sized[account]
    say(f"  expected at the assumed frequency and 25% share: ${r['month']:+,.0f} a month = {100*r['month']/account:+.1f}% [{100*r['low']/account:+.1f}%,{100*r['high']/account:+.1f}%] (game-clustered, whole sample)")
    say("  Sensitivity of that figure to the two assumptions that are not measured:")
    for share in (.05, .10, .25, .50):
        value = sum(freq[key]["fills"]*np.mean([game_dollars(o, q, share=share)[1] for o in hits[key]]) for key, q in zip(KEYS, plan))
        say(f"    our share of the contracts through {100*share:3.0f}%: ${value:+7,.0f} a month = {100*value/account:+5.1f}%")
    old = dict(men=sum(len([o for o in hits["men"] if o["month"] == m]) for m in ("2025-12", "2026-01", "2026-02"))/3/.57,
               women=sum(len([o for o in hits["women"] if o["month"] == m]) for m in ("2026-01", "2026-02"))/2/.956, tennis=freq["tennis"]["fills"])
    value = {key: old[key]*per_fill(hits[key], q).mean() for key, q in zip(KEYS, plan)}
    say(f"    fills per month as in LAST season's conference months (men {old['men']:.0f}, women {old['women']:.0f}, tennis {old['tennis']:.0f}) instead of November 2026's: "
        f"${sum(value.values()):+,.0f} a month = {100*sum(value.values())/account:+.1f}% (men ${value['men']:+,.0f} women ${value['women']:+,.0f} tennis ${value['tennis']:+,.0f})")
    name = "(a) fixed contracts "+"/".join(str(p) for p in plan)
    t = table[name]
    say(f"  a 150-day season at these fixed sizes: median end ${t['median']:,.0f}, 5th-95th ${t['p5']:,.0f} to ${t['p95']:,.0f}; deepest drawdown median {100*t['dd_median']:.1f}%, one season in twenty {100*t['dd95']:.1f}% or worse; "
        f"P(25% drawdown) {100*t['dd25']:.1f}%; P(ending below the start) {100*t['below']:.1f}%")


def main():
    rows = load()
    hits = {key: [o for o in rows if o["group"] == key and o["levels"][80]["filled"]] for key in KEYS}
    freq, triples = frequency(rows), frontier(rows)
    say("FAVOURITE-DIP SIZING. Strict print-through fills; every interval resamples whole games or whole nights.")
    say("Assumptions (not measured): we get 25% of the contracts printing through the bid; fills per month as in November 2026 (basketball) and May-October 2026 (tennis);")
    say("every bid is funded in cash from the start of play to settlement; the busiest moment seen is the busiest there will be; men's maker fee 0.28c a contract.\n")
    chosen, first, second = part_one(rows, hits, freq, triples)
    plans = {account: chosen[(account, BID_CAP)] for account in ACCOUNTS}
    sized = part_one_final(hits, first, second, freq, triples, plans)
    plan = plans[10000]
    part_two(hits, plan, 10000)
    part_two(hits, chosen[(10000, None)], 10000, brief=True)
    table, pool = part_three(rows, hits, freq, {cap: chosen[(10000, cap)] for cap in (BID_CAP, None, .03)}, 10000, CEILING)
    ladder = part_four(rows, hits, freq, triples)
    part_four_c(hits, freq, plan, 10000)
    part_five(rows, hits, freq, plan, 10000, sized, table)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"report.txt").write_text(buffer.getvalue())
    (OUT/"summary.json").write_text(json.dumps(dict(sizes={str(k): v for k, v in sized.items()}, rules=table, ladder_pays_from=ladder,
                                                    frequency=freq, resting_at_once=triples.tolist()), indent=1))


BID_CAP = .05                   # largest share of the account in one bid for the sizes carried forward
CEILING = (2000, 200, 2000)     # contracts beyond which an extra slice earns under about 3% on the first half (section 1a)

if __name__ == "__main__":
    main()
