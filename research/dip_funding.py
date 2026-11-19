"""Just-in-time funding of the favourite-dip bid: what a fixed account earns if the 80c bid is placed at a trigger instead of resting all game.

The traded rule rests an 80c bid on every 90-99c favourite from the start of
play to settlement; about a quarter fill, all reserve Q x 80c.
research/dip_capacity.py found the limit for a $10,000 account is the PEAK
number of bids resting at once (39 in November 2026), which fixes one order
size of 321 contracts. This script replays the same orders, prints and
accounting (imported from dip_capacity, nothing re-derived, no network) with
the bid placed only when the favourite first TRADES at or below a trigger T.

Timing conventions (every one chosen against the just-in-time bid unless said)
  * Prints: dip_capacity.prints - the trade record for every minute that
    straddles 85/80/75/70/65/60c, one-minute bars elsewhere. A bar's time is
    known only to the minute.
  * Trigger: the first print at or below T. FILLS assume it happened at the END
    of its bar (latest) and the bid is live DELAY seconds later (20 and 60).
    FUNDING assumes it happened at the START of its bar (earliest), no delay.
    The 'optimistic' columns flip the fill convention (trigger at the bar's
    start, a through-print at the bar's end).
  * Strict fill (the headline): the bid is live at or before the FIRST print
    strictly below 80c of the game. 'Late' fills: the bid arrives after that
    print and a later print goes through; these happen to a naive
    implementation whether wanted or not, so they are reported separately and
    as a sensitivity.
  * Cancel: once every print for 10 minutes has been above T+5c the unfilled
    bid is cancelled (clock starts at the first print above T+5c after the last
    print at or below it; minutes without a trade carry no information). It is
    placed again at the next print at or below T. A bid that has had a print
    through it is never cancelled, and is funded to settlement.
  * Queue: a late-placed bid fills only after X further contracts print
    strictly through 80c while it is live; it then gets min(Q, 25% x (through
    - X)), the capacity study's sizing.
  * Account (dip_capacity section 5): Q = account / (80c x most bids funded at
    once in November 2026, all three sports); fills a month at the current
    frequency; dollars per fill averaged over EVERY baseline fill in the
    sample (a missed fill earns zero); intervals resample games.

    .venv/bin/python research/dip_funding.py

Output: data/processed/dip-funding/report.txt and summary.json.
"""
from collections import defaultdict
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT
from research import dip_capacity as dc

OUT = ROOT / "data/processed/dip-funding"
TRIGGERS = (90, 88, 86, 84, 82)
DELAYS = (20, 60)
QUEUES = (0, 500, 2000, 10000)
ACCOUNTS = (1000, 5000, 10000, 25000)
BID, RECOVER, HOLD, DRAWS = 80, 5, 600, 4000
GROUPS = dc.GROUPS
RNG = np.random.default_rng(23)
buffer = io.StringIO()


def say(text=""):
    print(text, flush=True)
    buffer.write(text+"\n")


# ---------------------------------------------------------------- one order, one plan

def replay(o, trigger, delay, cancel, optimistic=False):
    """Replay one order. Returns funded intervals, whether the first print-through was caught (strict), the contracts
    printed through 80c while the bid was live (and able to fill), and how many times it was placed."""
    live = hit = strict = False
    fund_from = fill_from = recovering = None
    spans, through, placed, first_seen = [], 0.0, 0, False
    for when, lo, hi, size, seller in o["stream"]:
        early = when-60 if seller is None else when                # a bar's prints lie in (when-60, when]
        if live and cancel and not hit and recovering is not None and when > recovering+HOLD:
            spans.append((fund_from, recovering+HOLD))
            live, recovering = False, None
        if lo <= trigger+RECOVER:
            recovering = None
        elif recovering is None:
            recovering = when
        below = hi < BID
        if live and below and (when if optimistic else early) >= fill_from:
            if not first_seen:
                strict = True
            hit = True
            through += size
        if below:
            first_seen = True
        if not live and lo <= trigger:
            live, placed, recovering = True, placed+1, None
            fund_from = max(o["start"], early)
            fill_from = (early if optimistic else when)+delay
    if live:
        if cancel and not hit and recovering is not None and recovering+HOLD < o["settle"]:
            spans.append((fund_from, recovering+HOLD))
        else:
            spans.append((fund_from, max(o["settle"], fund_from)))
    return dict(spans=spans, strict=strict, through=through, placed=placed)


def got(o, plan, size, queue=0, late=False):
    """Contracts filled for a bid of `size` under a plan (None = resting from the start)."""
    if plan is None:
        return min(size, dc.SHARE*max(0.0, o["levels"][BID]["through"]-queue))
    r = o["plans"][plan]
    if not (r["strict"] or late):
        return 0.0
    return min(size, dc.SHARE*max(0.0, r["through"]-queue))


def kept(o, plan, queue=0, late=False):
    return got(o, plan, 1e-6, queue, late) > 0


def net(o):
    return 100*o["payout"]-BID


def line(part):
    if not part:
        return "n   0"
    nets = [net(o) for o in part]
    return f"n {len(part):3} win {100*np.mean([o['payout'] for o in part]):5.1f}% net {np.mean(nets):+5.1f}c {dc.band(dc.boot(nets))} = {100*np.mean(nets)/BID:+5.1f}%"


def short(part):
    return f"{len(part):3} {np.mean([net(o) for o in part]):+5.1f}c" if part else "  0    n/a"


def most(spans):
    """Largest number of funded bids at once, when, and bid-seconds."""
    marks = sorted([(a, 1) for a, b in spans]+[(b, -1) for a, b in spans])
    best = now = 0
    at = None
    for when, step in marks:
        now += step
        if now > best:
            best, at = now, when
    return best, at, sum(b-a for a, b in spans)


def spans_of(part, plan):
    return [(o["start"], o["settle"]) for o in part] if plan is None else [s for o in part for s in o["plans"][plan]["spans"]]


def daily_peaks(part, plan):
    by_day = defaultdict(list)
    for o in part:
        by_day[o["day"]] += [(o["start"], o["settle"])] if plan is None else o["plans"][plan]["spans"]
    return {day: most(spans)[0] for day, spans in by_day.items()}


def frequency(rows):
    """Fills per 30 days at the current frequency, exactly as dip_capacity section 4b."""
    freq = {}
    for key, _ in GROUPS:
        part = [o for o in rows if o["group"] == key]
        if key == "tennis":
            use = [m for m in sorted({o["month"] for o in part}) if "2026-05" <= m <= "2026-10"]
            scale = 1/len(use)
        else:
            use, scale = ["2026-11"], 30/dc.days_in(part, "2026-11")
        sub = [o for o in part if o["month"] in use]
        freq[key] = dict(fills=len(dc.fills(sub, BID))*scale, orders=len(sub)*scale)
    return freq


def monthly(hits, freq, picks, size, plan, queue=0, late=False, subset=None):
    """Dollars a month, the bootstrap draws, per-sport dollars and money in. `subset` = 0/1 for a chronological half of the fills."""
    total, draws, parts, money = 0.0, np.zeros(DRAWS), {}, 0.0
    for key, _ in GROUPS:
        part = hits[key]
        contracts = np.array([got(o, plan, size, queue, late) for o in part])
        dollars = contracts*np.array([net(o) for o in part])/100
        if subset is not None:
            half = len(part)//2
            pick = slice(0, half) if subset == 0 else slice(half, None)
            parts[key] = freq[key]["fills"]*dollars[pick].mean()
            total += parts[key]
            continue
        parts[key] = freq[key]["fills"]*dollars.mean()
        total += parts[key]
        money += freq[key]["fills"]*.8*contracts.mean()
        draws += freq[key]["fills"]*dollars[picks[key]].mean(axis=1)
    return total, draws, parts, money


def interval(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return f"[{lo:+7,.0f},{hi:+7,.0f}]"


# ---------------------------------------------------------------- report

def report():
    rows = [dc.measure(o) for o in dc.orders()]
    for o in rows:
        o["stream"] = dc.prints(o)[0]
        o["plans"] = {(t, d, c, opt): replay(o, t, d, c, opt) for t in TRIGGERS for d in DELAYS for c in (False, True) for opt in (False, True)}
    OUT.mkdir(parents=True, exist_ok=True)
    by_group = {key: [o for o in rows if o["group"] == key] for key, _ in GROUPS}
    hits = {key: sorted(dc.fills(by_group[key], BID), key=lambda o: (o["settle"], o["event"])) for key, _ in GROUPS}
    summary = {}

    say("=== 0. BASELINE (80c bid resting from the start, strict fills) - must match dip_capacity")
    for key, name in GROUPS:
        a, b = dc.halves(by_group[key])
        say(f"  {name:32} orders {len(by_group[key]):4} fills {line(hits[key])} | halves {short(dc.fills(a, BID))} / {short(dc.fills(b, BID))}")
    say("  Fills with NO WARNING on the tape: the game's first in-play print (or its whole first traded minute) is already at or below the level (pre-game price was 90-99c).")
    say("  The resting bid is credited with these; a print-triggered bid cannot be in front of them. Quotes are not on record, so whether the ask gave warning is unknown.")
    for key, name in GROUPS:
        text = f"    {name:32}"
        for level in (88, 79):
            part = [o for o in hits[key] if o["stream"][0][2] <= level]
            half = len(hits[key])//2
            text += f" | first print <= {level}c: {len(part):3} of {len(hits[key])} ({sum(o in part for o in hits[key][:half])} / {sum(o in part for o in hits[key][half:])} by half) net {short(part)[4:]}"
        say(text)

    say("\n=== 1. THE TRIGGER RULE: bid 80c placed when the favourite first trades at or below T")
    say("    kept = bid live at or before the game's first print under 80c (conservative timing; 'opt' = optimistic timing bound)")
    say("    Where the trigger falls inside a cached trade window (11 minutes before every men's and tennis fill; every minute under 85c) it is timed to the second, else to the minute.")
    say("    halves = chronological halves of the baseline FILLS (section 0 halves the orders, as dip_capacity does).")
    say("    late = bid arrives after that print and a later print still goes through it. missed = no fill at all. Returns are per fill on 80c.")
    for key, name in GROUPS:
        base = hits[key]
        say(f"  {name}: baseline fills {line(base)}")
        for cancel in (False, True):
            say(f"    {'WITH cancel above T+5c for 10 min' if cancel else 'NO cancel'}")
            for t in TRIGGERS:
                for d in DELAYS:
                    plan = (t, d, cancel, False)
                    keep = [o for o in base if o["plans"][plan]["strict"]]
                    opt = sum(o["plans"][(t, d, cancel, True)]["strict"] for o in base)
                    late = [o for o in base if not o["plans"][plan]["strict"] and o["plans"][plan]["through"] > 0]
                    miss = [o for o in base if not o["plans"][plan]["strict"] and o["plans"][plan]["through"] <= 0]
                    half = len(base)//2
                    ka, kb = [o for o in base[:half] if o["plans"][plan]["strict"]], [o for o in base[half:] if o["plans"][plan]["strict"]]
                    say(f"      T {t}c lead {d:2}s: kept {len(keep):3} = {100*len(keep)/len(base):5.1f}% (opt {100*opt/len(base):5.1f}%) {line(keep)[6:]} "
                        f"| halves kept {100*len(ka)/half:3.0f}% {short(ka)[4:]} / {100*len(kb)/(len(base)-half):3.0f}% {short(kb)[4:]} | late {short(late)} | missed {short(miss)}")
                    summary[f"{key}|T{t}|lead{d}|{'cancel' if cancel else 'hold'}"] = dict(
                        baseline=len(base), kept=len(keep), kept_optimistic=opt, late=len(late), missed=len(miss),
                        net_kept=float(np.mean([net(o) for o in keep])) if keep else None)

    say("\n  1b. TIME EACH BID IS FUNDED (hours from placement to cancel or settlement; baseline = start to settlement), every order")
    for key, name in GROUPS:
        part = by_group[key]
        hours = np.array([(o["settle"]-o["start"])/3600 for o in part])
        say(f"  {name}: baseline {len(part)} bids, hours each median {np.median(hours):.2f} mean {hours.mean():.2f}, total bid-hours {hours.sum():,.0f}")
        for cancel in (False, True):
            for t in TRIGGERS:
                plan = (t, 60, cancel, False)
                each = np.array([sum(b-a for a, b in o["plans"][plan]["spans"])/3600 for o in part])
                on = each[[o["plans"][plan]["placed"] > 0 for o in part]]
                times = np.mean([o["plans"][plan]["placed"] for o in part if o["plans"][plan]["placed"]])
                say(f"    T {t}c {'cancel' if cancel else 'hold  '}: bids ever placed {len(on):3} = {100*len(on)/len(part):4.1f}% of orders, placements per placed bid {times:.2f}, "
                    f"hours funded per placed bid median {np.median(on):.2f} mean {on.mean():.2f} | bid-hours {each.sum():6,.0f} = {100*each.sum()/hours.sum():4.1f}% of baseline")

    say("\n=== 2. QUEUE COST: the late bid fills only after X more contracts print strictly through 80c (strict timing, 60s lead, no cancel)")
    say("    fills as % of the baseline's fills at X = 0; return per fill on 80c. 'resting' = the same handicap applied to the bid resting from the start, for scale.")
    for key, name in GROUPS:
        base = hits[key]
        say(f"  {name} ({len(base)} baseline fills)")
        for label, plan in [("resting", None)]+[(f"T {t}c", (t, 60, False, False)) for t in TRIGGERS]:
            text = f"    {label:8}"
            for x in QUEUES:
                keep = [o for o in base if kept(o, plan, x)]
                half = len(base)//2
                ha, hb = [o for o in base[:half] if kept(o, plan, x)], [o for o in base[half:] if kept(o, plan, x)]
                nets = [net(o) for o in keep]
                text += f" | X {x:6,}: {len(keep):3} = {100*len(keep)/len(base):5.1f}% net {np.mean(nets):+5.1f}c {dc.band(dc.boot(nets))} = {100*np.mean(nets)/BID:+5.1f}% (halves {short(ha)[4:]}/{short(hb)[4:]})"
                summary[f"{key}|{label}|X{x}"] = dict(fills=len(keep), net=float(np.mean(nets)))
            say(text)
    say("  Depth actually resting at 80c before the first print-through (research/jit_bids.py section 3, seller-initiated prints AT 80c before t0):")
    say("    tennis 25/50/75/90% = 0 / 0 / 109 / 752 contracts; men's college basketball 0 / 6 / 392 / 1,293. Women's not measured.")

    say("\n=== 3. BIDS FUNDED AT ONCE (60s lead; funding from the earliest moment the trigger can have printed)")
    now = [o for o in rows if o["month"] == "2026-11"]
    days = sorted({o["day"] for o in now})
    hours_nov = 24*len(days)
    say(f"  November 2026 ({len(days)} days with orders, {len(now)} orders, all three sports). Dollars at Q = 321 contracts ($256.80 a bid).")
    peaks = {}
    for cancel in (None, False, True):
        for t in ((None,) if cancel is None else TRIGGERS):
            plan = None if cancel is None else (t, 60, cancel, False)
            top, at, seconds = most(spans_of(now, plan))
            each = {key: most(spans_of([o for o in now if o["group"] == key], plan))[0] for key, _ in GROUPS}
            tennis_all = max(most(spans_of([o for o in by_group["tennis"] if o["month"] == m], plan))[0] for m in ("2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"))
            daily = daily_peaks(now, plan)
            values = np.array([daily.get(d, 0) for d in days])
            first, second = values[:len(days)//2].max(), values[len(days)//2:].max()
            average = seconds/3600/hours_nov
            label = "resting all game (baseline)" if cancel is None else f"T {t}c {'cancel' if cancel else 'hold  '}"
            peaks[plan] = top
            say(f"    {label:28} most at once {top:3} (men {each['men']:2}, women {each['women']:2}, tennis {each['tennis']:2}; tennis May-Oct {tennis_all:2}) "
                f"| daily peak median {np.median(values):4.1f} 90th pct {np.percentile(values, 90):4.1f} | peak in first / second half of the month {first:2}/{second:2} "
                f"| average funded {average:5.2f} bids = ${256.8*average:6,.0f}, peak ${256.8*top:6,.0f}")
            summary[f"peak|{label.strip()}"] = dict(peak=top, average=average, by_sport=each, halves=[int(first), int(second)])
    say("  Whole sample, most funded at once per sport (the earlier seasons are thinner samples, so November is the binding month):")
    for cancel in (None, False, True):
        for t in ((None,) if cancel is None else (88,)):
            plan = None if cancel is None else (t, 60, cancel, False)
            say(f"    {'resting' if cancel is None else 'T 88c '+('cancel' if cancel else 'hold')}: "+", ".join(f"{key} {most(spans_of(by_group[key], plan))[0]}" for key, _ in GROUPS))
    say("  Most funded at once, all three sports together, by month (T 88c hold / T 88c cancel / resting):")
    say("    "+" | ".join(f"{m} {most(spans_of([o for o in rows if o['month'] == m], (88, 60, False, False)))[0]}/{most(spans_of([o for o in rows if o['month'] == m], (88, 60, True, False)))[0]}/"
                          f"{most(spans_of([o for o in rows if o['month'] == m], None))[0]}" for m in sorted({o["month"] for o in rows})))

    say("\n=== 4. THE ACCOUNT: Q = account / (80c x most bids funded at once in November 2026), dollars a month at the current frequency")
    freq = frequency(rows)
    picks = {key: RNG.integers(0, len(hits[key]), (DRAWS, len(hits[key]))) for key, _ in GROUPS}
    say("  fills per 30 days (as dip_capacity 4b): "+", ".join(f"{key} {freq[key]['fills']:.0f}" for key, _ in GROUPS))

    def row(account, label, plan, queue=0, late=False, tops=None):
        tops = tops or peaks[(plan[0], 60, plan[2], False)] if plan else peaks[None]
        size = account/(.8*tops)
        total, draws, parts, money = monthly(hits, freq, picks, size, plan, queue, late)
        base_total, base_draws, _, _ = monthly(hits, freq, picks, account/(.8*peaks[None]), None)
        h1 = monthly(hits, freq, picks, size, plan, queue, late, subset=0)[0]
        h2 = monthly(hits, freq, picks, size, plan, queue, late, subset=1)[0]
        b1 = monthly(hits, freq, picks, account/(.8*peaks[None]), None, subset=0)[0]
        b2 = monthly(hits, freq, picks, account/(.8*peaks[None]), None, subset=1)[0]
        say(f"    {label:40} peak {tops:4.1f} Q {size:6,.0f} | men ${parts['men']:+6,.0f} women ${parts['women']:+6,.0f} tennis ${parts['tennis']:+5,.0f} "
            f"| all ${total:+7,.0f} {interval(draws)} = {100*total/account:+5.1f}% a month | on money in {100*total/max(money, 1e-9):+5.1f}% "
            f"| vs resting ${total-base_total:+6,.0f} {interval(draws-base_draws)} | halves ${h1:+6,.0f} / ${h2:+6,.0f} (resting ${b1:+6,.0f} / ${b2:+6,.0f})")
        return dict(peak=tops, Q=size, dollars=total, percent=100*total/account, interval=[float(v) for v in np.percentile(draws, [2.5, 97.5])],
                    gain=total-base_total, gain_interval=[float(v) for v in np.percentile(draws-base_draws, [2.5, 97.5])], halves=[h1, h2], resting_halves=[b1, b2])

    for account in ACCOUNTS:
        say(f"  account ${account:,}")
        summary[f"account{account}|resting"] = row(account, "resting all game (baseline)", None)
        for cancel in (False, True):
            for t in TRIGGERS:
                name = f"T {t}c {'cancel' if cancel else 'hold'}, 60s lead, strict"
                summary[f"account{account}|{name}"] = row(account, name, (t, 60, cancel, False))
        if account == 10000:
            say("    sensitivities at T 88c, no cancel:")
            summary["account10000|T88 20s"] = row(account, "  20s lead", (88, 20, False, False))
            summary["account10000|T88 late"] = row(account, "  60s lead, late fills also taken", (88, 60, False, False), late=True)
            for x in QUEUES[1:]:
                summary[f"account10000|T88 X{x}"] = row(account, f"  60s lead, queue X = {x:,}", (88, 60, False, False), queue=x)
            summary["account10000|T88 X500 late"] = row(account, "  60s lead, late fills taken, X = 500", (88, 60, False, False), queue=500, late=True)
            for margin in (1.25, 1.5, 2.0):
                summary[f"account10000|T88 margin{margin}"] = row(account, f"  reserve for {margin:.2f}x the observed peak", (88, 60, False, False), tops=margin*peaks[(88, 60, False, False)])
            say("    sensitivities at T 88c WITH cancel:")
            summary["account10000|T88c X500"] = row(account, "  60s lead, queue X = 500", (88, 60, True, False), queue=500)
            for margin in (1.25, 1.5, 2.0):
                summary[f"account10000|T88c margin{margin}"] = row(account, f"  reserve for {margin:.2f}x the observed peak", (88, 60, True, False), tops=margin*peaks[(88, 60, True, False)])
    say("\n=== 5. RISK AT THE LARGER ORDER SIZE ($10,000 account; realised dollars per day over the whole sample, as sampled; strict fills, 60s lead)")
    for label, plan in (("resting all game", None), ("T 88c hold", (88, 60, False, False)), ("T 88c cancel", (88, 60, True, False)), ("T 88c hold, reserve 2x the peak", (88, 60, False, False))):
        size = 10000/(.8*peaks[plan]*(2 if "2x" in label else 1))
        daily, biggest = defaultdict(float), 0.0
        for o in rows:
            if o["levels"][BID]["filled"]:
                c = got(o, plan, size)
                daily[o["day"]] += c*net(o)/100
                biggest = max(biggest, .8*c)
        values = sorted(v for v in daily.values())
        run = low = 0.0
        for day in sorted(daily):
            run = min(0.0, run+daily[day])
            low = min(low, run)
        say(f"    {label:32} Q {size:6,.0f} | one bid = ${.8*size:6,.0f} = {100*.8*size/10000:4.1f}% of the account | worst five days {[round(v) for v in values[:5]]} | "
            f"losing days {100*np.mean([v < 0 for v in values]):.0f}% | deepest run of losses ${low:,.0f}")
        summary[f"risk|{label}"] = dict(Q=size, worst=values[:5], drawdown=low)
    (OUT/"summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (OUT/"report.txt").write_text(buffer.getvalue())


if __name__ == "__main__":
    report()
