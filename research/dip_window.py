"""Does limiting WHEN the resting 80c dip bid may fill preserve its return at large order size?

The rule sized by research/dip_capacity.py: a 90-99c favourite gets a resting
bid at 80c once play is live, held to settlement, filled only by prints
strictly below 80c. Its return falls with order size because the contracts
printing through 80c pile up in the games the favourite loses. Here the bid
accepts fills only for a window W after the first print through 80c and then
cancels what is unfilled:

    contracts = min(Q, share x contracts printed under 80c inside the window)

Orders, in-play prints and frequencies are dip_capacity's (imported, cached
data only, no network). One-minute bars that sit wholly under 80c are spread
evenly over their minute; a first print that is a bar starts the clock at the
start of that minute.

Variants (one table each)
  a  window measured from the start of play (only early-game dips fill)
  b  cancel the remainder at the first print above 85c after the first fill
  c  cancel the remainder at the first print below 70c (keep what is filled).
     The triggering trade itself still fills us (it traded through the bid);
     a triggering one-minute bar does not (lower bound).

    .venv/bin/python research/dip_window.py

Output: data/processed/dip-window/report.txt and cells.json.
"""
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import dip_capacity as dc

OUT = dc.ROOT / "data/processed/dip-window"
LEVEL = 80
WINDOWS = (1, 2, 5, 10, 20, 60, None)             # minutes after the first print; None = no limit
FROM_START = (5, 10, 20, 30, 60, 90, None)        # minutes after the start of play
SIZES = (10, 100, 300, 1000, 3000, 10000)
SHARES = (.25, .10)
RNG = np.random.default_rng(11)
DRAWS = 4000
buffer = io.StringIO()


def say(text=""):
    print(text, flush=True)
    buffer.write(text+"\n")


# ---------------------------------------------------------------- per-game volume under 80c

def prepare(o):
    """Prints under 80c as arrays (start, end, contracts): a trade has start == end, a bar covers its minute."""
    stream, _ = dc.prints(o)
    below = [p for p in stream if p[2] < LEVEL]
    assert len(below) == len([p for p in stream if p[1] < LEVEL]), "an unread straddling minute"
    o["lo"] = np.array([p[0]-60 if p[4] is None else p[0] for p in below])
    o["hi"] = np.array([p[0] for p in below])
    o["vol"] = np.array([p[3] for p in below])
    o["t0"] = float(o["lo"][0])
    after = [p for p in stream if p[0] > o["t0"]]
    up = [p for p in after if p[2] > 85]
    o["t_up"] = (up[0][0]-60 if up[0][4] is None else up[0][0]) if up else None
    down = [p for p in stream if p[1] < 70]
    o["t_down"], o["down_bar"] = (None, False)
    if down:
        bar = down[0][4] is None
        o["t_down"], o["down_bar"] = (down[0][0]-60 if bar else down[0][0]), bar
    return o


def volume(o, a, b, closed=True):
    """Contracts printed under 80c in [a, b] (or [a, b) when not closed); bars pro rata."""
    lo, hi, vol = o["lo"], o["hi"], o["vol"]
    trade = hi == lo
    inside = (hi >= a) & ((hi <= b) if closed else (hi < b))
    out = vol[trade & inside].sum()
    part = np.clip((np.minimum(hi, b)-np.maximum(lo, a))/60, 0, 1)
    return float(out+(vol[~trade]*part[~trade]).sum())


def available(o, rule, w):
    """Contracts under 80c that the bid is live for."""
    end = o["settle"]+10**7
    if rule == "window":
        return volume(o, o["t0"], end if w is None else o["t0"]+60*w)
    if rule == "start":
        return volume(o, o["start"], end if w is None else o["start"]+60*w)
    if rule == "up":
        return volume(o, o["t0"], end) if o["t_up"] is None else volume(o, o["t0"], o["t_up"], closed=False)
    if rule == "down":
        if o["t_down"] is None:
            return volume(o, o["t0"], end)
        return volume(o, o["t0"], o["t_down"], closed=not o["down_bar"])
    raise ValueError(rule)


# ---------------------------------------------------------------- statistics

def cell(avail, net, size, share, pick=None):
    """Contracts and dollars per game for a bid of `size`."""
    got = np.minimum(size, share*avail)
    if pick is not None:
        got, net = got[pick], net[pick]
    return got, got*net/100


def ratio(got, dollars):
    return 100*dollars.sum()/max(.8*got.sum(), 1e-9)


def boot(got, dollars, base=None):
    """Clustered by game (one order per game): 95% intervals of the % return, of dollars per game, and of dollars per game minus `base`."""
    pick = RNG.integers(0, len(got), (DRAWS, len(got)))
    g, d = got[pick].sum(axis=1), dollars[pick].sum(axis=1)
    out = dict(ret=np.percentile(100*d/np.maximum(.8*g, 1e-9), [2.5, 97.5]), usd=np.percentile(d/len(got), [2.5, 97.5]))
    if base is not None:
        out["diff"] = np.percentile((dollars-base)[pick].mean(axis=1), [2.5, 97.5])
    return out


def label(w):
    return "no limit" if w is None else f"{w:3} min "


def table(part, freq, rule, windows, share, store, key):
    net = np.array([100*o["payout"]-LEVEL for o in part])
    half = np.arange(len(part)) < len(part)//2
    for size in SIZES:
        base = None
        rows = []
        for w in reversed(windows):                      # baseline first, printed last
            avail = np.array([available(o, rule, w) for o in part])
            got, dollars = cell(avail, net, size, share)
            if w is None and rule in ("window", "start"):
                base = dollars
            b = boot(got, dollars, base)
            a1, a2 = cell(avail, net, size, share, half), cell(avail, net, size, share, ~half)
            n = int((got > 0).sum())
            win = 100*np.mean(net[got > 0] > 0) if n else float("nan")
            text = (f"    Q {size:6,} {label(w) if rule in ('window', 'start') else 'cancel  '}: fills {n:3} win {win:5.1f}% contracts {got.mean():7,.0f} full {100*np.mean(got >= size):3.0f}% | "
                    f"return {ratio(got, dollars):+6.1f}% ({.8*ratio(got, dollars):+5.1f}c) [{b['ret'][0]:+6.1f},{b['ret'][1]:+6.1f}]% | "
                    f"per 30 days in ${freq*.8*got.mean():9,.0f} earned ${freq*dollars.mean():+8,.0f} [{freq*b['usd'][0]:+8,.0f},{freq*b['usd'][1]:+8,.0f}] | "
                    f"halves {ratio(*a1):+6.1f}% / {ratio(*a2):+6.1f}%")
            if "diff" in b:
                text += f" | vs no limit ${freq*(dollars-base).mean():+7,.0f} [{freq*b['diff'][0]:+7,.0f},{freq*b['diff'][1]:+7,.0f}]"
            if n < 30:
                text += "  (few fills)"
            rows.append(text)
            store.append(dict(group=key, rule=rule, share=share, size=size, window=w, fills=n, contracts=float(got.mean()), ret=float(ratio(got, dollars)),
                              ret_ci=[float(v) for v in b["ret"]], usd30=float(freq*dollars.mean()), usd30_ci=[float(freq*v) for v in b["usd"]],
                              halves=[float(ratio(*a1)), float(ratio(*a2))]))
        for text in reversed(rows):
            say(text)
    return


def equal_money(part, share):
    """The control: the no-limit bid at the smaller size that puts in the SAME money as the windowed bid of Q."""
    net = np.array([100*o["payout"]-LEVEL for o in part])
    whole = np.array([available(o, "window", None) for o in part])
    for size in SIZES[1:]:
        text = f"    Q {size:6,}:"
        for w in WINDOWS[:-1]:
            avail = np.array([available(o, "window", w) for o in part])
            got, dollars = cell(avail, net, size, share)
            lo, hi = 0.0, float(size)
            for _ in range(60):
                mid = (lo+hi)/2
                lo, hi = (mid, hi) if np.minimum(mid, share*whole).sum() < got.sum() else (lo, mid)
            g2, d2 = cell(whole, net, hi, share)
            pick = RNG.integers(0, len(part), (DRAWS, len(part)))
            diff = np.percentile(100*dollars[pick].sum(axis=1)/(.8*got[pick].sum(axis=1))-100*d2[pick].sum(axis=1)/(.8*g2[pick].sum(axis=1)), [2.5, 97.5])
            text += f" | {w:2}m {ratio(got, dollars):+5.1f}% vs Q*={hi:5,.0f} {ratio(g2, d2):+5.1f}% (diff [{diff[0]:+5.1f},{diff[1]:+5.1f}])"
        say(text)


def out_of_half(part, share):
    net = np.array([100*o["payout"]-LEVEL for o in part])
    half = np.arange(len(part)) < len(part)//2
    avail = {w: np.array([available(o, "window", w) for o in part]) for w in WINDOWS}
    for size in SIZES:
        text = f"    Q {size:6,}:"
        for name, score in (("best $", lambda g, d: d.mean()), ("best %", ratio)):
            best = max(WINDOWS[:-1], key=lambda w: score(*cell(avail[w], net, size, share, half)))
            first, base1 = cell(avail[best], net, size, share, half), cell(avail[None], net, size, share, half)
            second, base2 = cell(avail[best], net, size, share, ~half), cell(avail[None], net, size, share, ~half)
            pick = RNG.integers(0, len(second[0]), (DRAWS, len(second[0])))
            usd = np.percentile((second[1]-base2[1])[pick].mean(axis=1), [2.5, 97.5])
            pct = np.percentile(100*second[1][pick].sum(axis=1)/(.8*second[0][pick].sum(axis=1))-100*base2[1][pick].sum(axis=1)/(.8*base2[0][pick].sum(axis=1)), [2.5, 97.5])
            text += (f" | {name} on half 1: W {label(best).strip():>6} ({ratio(*first):+5.1f}% vs {ratio(*base1):+5.1f}%) -> half 2: {ratio(*second):+5.1f}% vs no limit {ratio(*base2):+5.1f}% "
                     f"(diff [{pct[0]:+5.1f},{pct[1]:+5.1f}]pt), $/fill {second[1].mean():+7.2f} vs {base2[1].mean():+7.2f} (diff [{usd[0]:+7.2f},{usd[1]:+7.2f}])")
        say(text)


# ---------------------------------------------------------------- report

def main():
    rows = [dc.measure(o) for o in dc.orders()]
    OUT.mkdir(parents=True, exist_ok=True)
    groups, freq, peaks = {}, {}, {}
    for key, name in dc.GROUPS:
        part = [o for o in rows if o["group"] == key]
        if key == "tennis":
            use = [m for m in sorted({o["month"] for o in part}) if "2026-05" <= m <= "2026-10"]
            scale = 1/len(use)
        else:
            use, scale = ["2026-11"], 30/dc.days_in(part, "2026-11")
        sub = [o for o in part if o["month"] in use]
        freq[key] = len(dc.fills(sub, LEVEL))*scale
        peaks[key] = max(dc.peak([o for o in sub if o["month"] == m])[0] for m in use)
        groups[key] = [prepare(o) for o in sorted(dc.fills(part, LEVEL), key=lambda o: (o["settle"], o["event"]))]
    store = []

    say("=== 0. THE SAMPLE (80c bid, strict fills; same orders, prints and frequency as dip_capacity.py)")
    for key, name in dc.GROUPS:
        part = groups[key]
        t0 = np.array([(o["t0"]-o["start"])/60 for o in part])
        whole = np.array([available(o, "window", None) for o in part])
        lost = np.array([o["payout"] < .5 for o in part])
        say(f"  {name:32} fills {len(part):3} lost {lost.sum():2} | fills per 30 days {freq[key]:.0f} | minutes from start to the first print under 80c: quartiles {np.percentile(t0, 25):.0f} / {np.median(t0):.0f} / {np.percentile(t0, 75):.0f}")
        for w in WINDOWS:
            v = np.array([available(o, "window", w) for o in part])
            rank = np.argsort(np.argsort(v)).astype(float)
            rho = np.corrcoef(rank, 1-lost)[0, 1]
            say(f"      contracts under 80c within {label(w)}: median won {np.median(v[~lost]):9,.0f} lost {np.median(v[lost]):9,.0f} | mean won {v[~lost].mean():9,.0f} lost {v[lost].mean():9,.0f} "
                f"| lost games' share of all contracts {100*v[lost].sum()/v.sum():3.0f}% (they are {100*lost.mean():.0f}% of games) | rank correlation with winning {rho:+.2f}")
        up = np.array([o["t_up"] is not None for o in part])
        down = np.array([o["t_down"] is not None for o in part])
        say(f"      printed above 85c after the first print: {up.sum()} games ({lost[up].sum()} lost) | printed under 70c: {down.sum()} games ({lost[down].sum()} lost; trigger was a bar in {sum(o['down_bar'] for o in part)})")
        assert abs(whole.sum()-sum(o["levels"][LEVEL]["through"] for o in part)) < 1e-6*max(whole.sum(), 1)

    for share in SHARES:
        say(f"\n=== 1. WINDOW AFTER THE FIRST PRINT UNDER 80c; contracts = min(Q, {100*share:.0f}% of the contracts under 80c inside the window)")
        say("    return = dollars earned / money put in; intervals = bootstrap by game, 95%; halves = chronological halves of the fills; 'vs no limit' = dollars per 30 days against the same Q with no window (paired)")
        for key, name in dc.GROUPS:
            say(f"  {name} ({len(groups[key])} fills, {freq[key]:.0f} per 30 days)")
            table(groups[key], freq[key], "window", WINDOWS, share, store, key)

    for share in SHARES:
        say(f"\n=== 2a. WINDOW FROM THE START OF PLAY (only prints under 80c in the first W minutes of the game fill), share {100*share:.0f}%")
        for key, name in dc.GROUPS:
            say(f"  {name}")
            table(groups[key], freq[key], "start", FROM_START, share, store, key)
    for rule, title in (("up", "2b. CANCEL THE REMAINDER AT THE FIRST PRINT ABOVE 85c AFTER THE FIRST FILL"), ("down", "2c. CANCEL THE REMAINDER AT THE FIRST PRINT UNDER 70c (keep what is filled)")):
        for share in SHARES:
            say(f"\n=== {title}, share {100*share:.0f}%   (compare with the 'no limit' rows of section 1)")
            for key, name in dc.GROUPS:
                say(f"  {name}")
                table(groups[key], freq[key], rule, (None,), share, store, key)

    say("\n=== 3. REAL OR ARTIFACT?")
    say("  3a. Is the window just a smaller order? Windowed bid of Q against the NO-LIMIT bid of the smaller size Q* that puts in the same money (diff = 95% interval of the difference in % return, paired)")
    for share in SHARES:
        for key, name in dc.GROUPS:
            say(f"  {name}, share {100*share:.0f}%")
            equal_money(groups[key], share)
    say("\n  3b. Window chosen on the first chronological half of the fills, reported on the second half (by dollars per fill and by % return)")
    for share in SHARES:
        for key, name in dc.GROUPS:
            say(f"  {name}, share {100*share:.0f}% (second half: {len(groups[key])-len(groups[key])//2} fills, {sum(o['payout'] < .5 for o in groups[key][len(groups[key])//2:])} lost)")
            out_of_half(groups[key], share)

    say("\n=== 4. A $10,000 ACCOUNT, all three sports, Q = account / (80c x most orders resting at once), as dip_capacity.py section 5")
    now = [o for o in rows if o["month"] == "2026-11"]
    top, day = dc.peak(now)
    size = 10000/(.8*top)
    say(f"  most orders resting at once in November 2026: {top} (on {day}) -> Q = {size:.0f} contracts. The window does not lower the reserve: unfilled bids rest before the dip.")
    rules = [("window", w) for w in WINDOWS]+[("start", w) for w in FROM_START[:-1]]+[("up", None), ("down", None)]
    names = {"window": "window after first print", "start": "window from game start", "up": "cancel above 85c", "down": "cancel under 70c"}
    for account, q in ((10000, size), (25000, 25000/(.8*top)), (100000, 100000/(.8*top))):
        for share in SHARES+(.05, .50):
            say(f"  account ${account:,}, Q {q:,.0f}, share {100*share:.0f}%")
            picks = {key: RNG.integers(0, len(groups[key]), (DRAWS, len(groups[key]))) for key in groups}
            base = {}
            for rule, w in [("window", None)]+rules:
                total = money = 0.0
                draws, diff = np.zeros(DRAWS), np.zeros(DRAWS)
                text = ""
                for key, _ in dc.GROUPS:
                    part = groups[key]
                    net = np.array([100*o["payout"]-LEVEL for o in part])
                    got, dollars = cell(np.array([available(o, rule, w) for o in part]), net, q, share)
                    if (rule, w) == ("window", None):
                        base[key] = dollars
                    total += freq[key]*dollars.mean()
                    money += freq[key]*.8*got.mean()
                    draws += freq[key]*dollars[picks[key]].mean(axis=1)
                    diff += freq[key]*(dollars-base[key])[picks[key]].mean(axis=1)
                    text += f" | {key} ${freq[key]*dollars.mean():+6,.0f} ({ratio(got, dollars):+5.1f}%)"
                if (rule, w) == ("window", None) and text and "shown" in base:
                    continue
                base["shown"] = True
                lo, hi = np.percentile(draws, [2.5, 97.5])
                dl, dh = np.percentile(diff, [2.5, 97.5])
                say(f"    {names[rule]:24} {label(w) if rule in ('window', 'start') else '        '}: money in ${money:8,.0f} earned ${total:+7,.0f} a month [{lo:+7,.0f},{hi:+7,.0f}] = {100*total/max(money, 1e-9):+5.1f}% on money in, "
                    f"{100*total/account:+5.1f}% of the account{text} | vs no limit ${total-sum(freq[k]*base[k].mean() for k in groups):+6,.0f} [{dl:+6,.0f},{dh:+6,.0f}]")
    (OUT/"cells.json").write_text(json.dumps(store))
    (OUT/"report.txt").write_text(buffer.getvalue())


if __name__ == "__main__":
    main()
