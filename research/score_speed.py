"""Is Kalshi's own live score fast enough to buy the scoring team before the price moves?

research/late_state.py (test 4e) found the winner-market price takes about 5 s
(NBA) to 30 s (college) to absorb a late made basket in a close game. This
script asks whether the score Kalshi itself serves at /live_data/batch arrives
inside that window, using

* the 15 s recorder (research/live_games.py, data/raw/live-games, read only):
  every score change, the poll that first showed it (T), the poll before
  (T-prev), and the stamps Kalshi puts on the state (`score_last_updated_ts`,
  `last_updated_ts`);
* a short 2 s poll of live games (`live`), which measures how long after
  `score_last_updated_ts` a score becomes visible to a public reader;
* the public trade tape (/markets/trades) around those moments.

Public unauthenticated reads only. Nothing is traded. paper/ is not touched.

Commands
    tape     trade tape and settlement for recorder games with a late close score change (1 read / 5 s)
    live     watch live basketball slowly (1 read / 30 s); poll at 2 s only while a game is late and close,
             for at most --minutes of fast polling in total and at most 12 events
    livetape trade tape for the games the live poll watched
    report   everything, no network

Definitions
* Late and close: last 3:00 of regulation (NBA 4th quarter, college 2nd half)
  or overtime, and the lead before the change was 6 or less.
* Scoring team's price: every trade on either market of the event, restated as
  the scoring team's YES price. A taker purchase of the scoring team is a YES
  taker on its market or a NO taker on the opponent's.
* Fill proxy: the first taker purchase of the scoring team printed at or after
  the decision time plus the delay, within 20 s. Fee 7*p*(1-p) cents.
* Mark: last trade 60 s after the fill, and settlement (100 or 0).
* Return: net cents / (price paid + fee). Intervals: 2,000 bootstrap draws of
  whole games. Halves: games in start order, split at the median game.

Outputs under data/processed/score-speed/ (raw reads in its raw/ folder).
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import random
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, timestamp

REC = ROOT / "data/raw/live-games"
OUT = ROOT / "data/processed/score-speed"
RAW = OUT / "raw"
FLAG = RAW / "fast.flag"                      # exists while the 2 s poll is running; the tape download waits
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}
SERIES = ("KXNBAGAME", "KXNCAAMBGAME", "KXNCAAWBGAME")
NAMES = {"KXNBAGAME": "NBA", "KXNCAAMBGAME": "Men's college", "KXNCAAWBGAME": "Women's college"}
PACE = 6.5
last_call = [0.0]


def series_of(event):
    return event.split("-")[0]


def secs(clock):
    try:
        m, s = str(clock).split(":")
        return int(m)*60+float(s)
    except ValueError:
        return None


def regulation(event, state):
    if state.get("period_type") == "quarter":
        return 4
    return 2


def is_late(event, state):
    period, left = state.get("period") or 0, secs(state.get("period_remaining_time"))
    reg = regulation(event, state)
    if period > reg:
        return True
    return period == reg and left is not None and left <= 180


def get(client, path, params, pace=PACE):
    """One paced public read; backs off 60-300 s on 429. Returns the response or None."""
    for attempt in range(5):
        while FLAG.exists() and pace >= PACE and time.time()-FLAG.stat().st_mtime < 30:
            time.sleep(5)
        wait = last_call[0]+pace-time.time()
        if wait > 0:
            time.sleep(wait)
        last_call[0] = time.time()
        try:
            r = client.get(config.KALSHI_PROD_BASE+path, params=params)
        except httpx.RequestError:
            time.sleep(10)
            continue
        if r.status_code == 429:
            print(f"429 on {path}; waiting {60*(attempt+1)} s", flush=True)
            time.sleep(min(60*(attempt+1), 300))
            continue
        if r.status_code >= 500:
            time.sleep(10)
            continue
        return r
    return None


# ------------------------------------------------------------------ recorder

def recorder_rows():
    rows = defaultdict(list)
    for f in sorted(REC.glob("*.jsonl")):
        with f.open() as fh:
            for line in fh:
                if not any(s in line[:90] for s in SERIES):
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r["event"].startswith(SERIES):
                    rows[r["event"]].append(r)
    return rows


def sides(event, tickers):
    """(away ticker, home ticker): the event code is AWAY then HOME. One ticker is enough."""
    code = event.split("-")[-1][7:]
    found = set()
    for t in tickers:
        team = t.split("-")[-1]
        if code.startswith(team) and len(team) < len(code):
            found.add((team, code[len(team):]))
        if code.endswith(team) and len(team) < len(code):
            found.add((code[:-len(team)], team))
    if len(found) != 1:
        return None
    away, home = found.pop()
    return f"{event}-{away}", f"{event}-{home}"


def changes_of(event, rs, tickers=None):
    """Score changes in one event's rows (each row: ts, state, optional prev poll time)."""
    out, prev = [], None
    for r in rs:
        s = r["state"]
        score = (s.get("away_points"), s.get("home_points"))
        if None in score:
            continue
        if prev and score != prev["score"] and s.get("status") == "inprogress":
            da, dh = score[0]-prev["score"][0], score[1]-prev["score"][1]
            if da >= 0 and dh >= 0 and da+dh <= 8:
                side = "away" if dh == 0 else "home" if da == 0 else "both"
                lead_before = prev["score"][0]-prev["score"][1]
                out.append(dict(event=event, T=r["ts"], prevT=r.get("prev_poll", prev["ts"]), side=side, pts=da+dh,
                                before=list(prev["score"]), after=list(score), lead=abs(lead_before),
                                scorer_margin=None if side == "both" else (lead_before if side == "away" else -lead_before),
                                period=s.get("period"), left=secs(s.get("period_remaining_time")),
                                late=is_late(event, s), score_ts=s.get("score_last_updated_ts"),
                                upd_ts=s.get("last_updated_ts"), evt_ts=s.get("last_event_created_ts"),
                                provider=s.get("provider"), play=s.get("last_play"), sent=r.get("sent")))
        prev = dict(score=score, ts=r["ts"])
    return out


def cents(q, key):
    try:
        return 100*float(q[key])
    except (KeyError, TypeError, ValueError):
        return None


def quote_series(rs, pair):
    """[(time the quote was read, away ask, away ask size, home ask, home ask size)] from the recorder's rows."""
    out = []
    for r in rs:
        book = {r["ticker"]: r["quote"]}
        if r.get("other"):
            book[r["other"]["ticker"]] = r["other"]["quote"]
        row = [r.get("quote_ts") or r["ts"]]
        for own, opp in (pair, pair[::-1]):
            offers = []
            if own in book and cents(book[own], "yes_ask_dollars"):
                offers.append((cents(book[own], "yes_ask_dollars"), cents(book[own], "yes_ask_size_fp")/100))
            if opp in book and cents(book[opp], "yes_bid_dollars"):
                offers.append((100-cents(book[opp], "yes_bid_dollars"), cents(book[opp], "yes_bid_size_fp")/100))
            row += list(min(offers)) if offers else [None, None]
        out.append(row)
    return out


def recorder_changes():
    rows = recorder_rows()
    games, changes = {}, []
    for event, rs in rows.items():
        first = rs[0]
        pair = sides(event, [first["ticker"]]+([first["other"]["ticker"]] if first.get("other") else []))
        if not pair:
            continue
        games[event] = dict(event=event, away=pair[0], home=pair[1], start=first.get("start"), quotes=quote_series(rs, pair))
        changes += changes_of(event, rs)
    return games, changes


# ------------------------------------------------------------------ tape

def tape_file(ticker):
    return RAW / "trades" / f"{ticker}.json"


def fetch_tape(client, ticker, start, stop, pages=40):
    rows, cursor, truncated = [], "", True
    for _ in range(pages):
        params = dict(ticker=ticker, min_ts=int(start), max_ts=int(stop), limit=1000)
        if cursor:
            params["cursor"] = cursor
        r = get(client, "/markets/trades", params)
        if r is None or r.status_code != 200:
            return None
        data = r.json()
        for t in data.get("trades", []):
            if not t.get("is_block_trade"):
                rows.append([timestamp(t["created_time"]), t.get("taker_side"),
                             round(100*float(t["yes_price_dollars"]), 2), float(t.get("count_fp", 0))])
        cursor = data.get("cursor")
        if not cursor or not data.get("trades"):
            truncated = False
            break
    rows.sort()
    if truncated and rows:                     # pages run newest first: keep what is complete
        start = rows[0][0]
    return dict(start=start, stop=stop, truncated=truncated, rows=rows)


def fetch_result(client, event):
    r = get(client, "/markets", dict(event_ticker=event, limit=10))
    if r is None or r.status_code != 200:
        return None
    return {m["ticker"]: m.get("result") for m in r.json().get("markets", [])}


def download(wanted):
    """wanted: event -> (away ticker, home ticker, start, stop)."""
    (RAW / "trades").mkdir(parents=True, exist_ok=True)
    results_file = RAW / "results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}
    with httpx.Client(timeout=40, headers=HEADERS) as client:
        for n, (event, (away, home, start, stop)) in enumerate(sorted(wanted.items())):
            for ticker in (away, home):
                f = tape_file(ticker)
                if f.exists():
                    old = json.loads(f.read_text())
                    if (old["start"] <= start or old.get("truncated")) and old["stop"] >= stop:
                        continue
                got = fetch_tape(client, ticker, start, stop)
                if got is None:
                    print("failed", ticker, flush=True)
                    continue
                f.write_text(json.dumps(got))
                print(f"  {ticker} {len(got['rows'])} trades{' TRUNCATED' if got['truncated'] else ''}", flush=True)
            if not (results.get(event) and all(results[event].values())):
                res = fetch_result(client, event)
                if res:
                    results[event] = res
                    results_file.write_text(json.dumps(results))
            print(f"{n+1}/{len(wanted)} {event}", flush=True)


def tape_cmd():
    games, changes = recorder_changes()
    wanted = {}
    for c in changes:
        if c["late"] and c["lead"] <= 6:
            g = games[c["event"]]
            lo, hi = wanted.get(c["event"], (None, None, 9e18, 0))[2:]
            wanted[c["event"]] = (g["away"], g["home"], min(lo, c["prevT"]-180), max(hi, c["T"]+300))
    print(len(wanted), "games with a late close score change")
    download(wanted)


# ------------------------------------------------------------------ live 2 s poll

def milestones(client, since):
    found, cursor = {}, ""
    for _ in range(8):
        r = get(client, "/milestones", dict(limit=200, category="Sports", minimum_start_date=since, cursor=cursor))
        if r is None or r.status_code != 200:
            break
        page = r.json()
        for m in page.get("milestones", []):
            if m.get("type") != "basketball_game":
                continue
            for event in m.get("primary_event_tickers") or []:
                if event.startswith(SERIES):
                    found[m["id"]] = (event, m.get("start_date"))
        cursor = page.get("cursor")
        if not cursor or not page.get("milestones"):
            break
    return found


def urgency(event, s):
    """Smaller is more urgent; None when the game is not in progress."""
    if s.get("status") != "inprogress":
        return None
    reg, period, left = regulation(event, s), s.get("period") or 0, secs(s.get("period_remaining_time")) or 0
    lead = abs((s.get("away_points") or 0)-(s.get("home_points") or 0))
    remaining = left+max(0, reg-period)*(720 if reg == 4 else 1200) if period <= reg else 0
    return remaining+60*lead, (period >= reg and (period > reg or left <= 300) and lead <= 9)


def live_cmd(minutes, hours, max_events):
    RAW.mkdir(parents=True, exist_ok=True)
    out = (RAW / "live-polls.jsonl").open("a")
    beat = (RAW / "live-beat.jsonl").open("a")
    stop, fast_left, known, refreshed = time.time()+hours*3600, minutes*60, {}, 0
    chosen, last, reselect, fast = [], {}, 0, False
    with httpx.Client(timeout=10, headers=HEADERS) as client:
        while time.time() < stop and fast_left > 0:
            now = time.time()
            if now-refreshed > 900 and not fast:
                since = datetime.fromtimestamp(now-5*3600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                found = milestones(client, since)
                if found:
                    known = found
                refreshed = time.time()
            broad = (not fast) or time.time()-reselect > 60
            if broad:
                started = sorted((m for m, (e, start) in known.items() if start and timestamp(start) < time.time()+600),
                                 key=lambda m: known[m][1], reverse=True)
                batches = [started[i:i+40] for i in range(0, min(len(started), 120), 40)]
            else:
                batches = [chosen]
            if not batches or not batches[0]:
                time.sleep(60)
                continue
            rows, sent, got, server = [], None, None, None
            for ids in batches:
                if fast:
                    FLAG.write_text(str(time.time()))
                began = time.time()
                r = get(client, "/live_data/batch", dict(milestone_ids=ids), pace=2.0 if fast else 30.0/len(batches)+2)
                if r is None or r.status_code != 200:
                    continue
                sent, got = last_call[0], time.time()
                if fast:
                    fast_left -= min(got-began, 6)
                try:
                    server = parsedate_to_datetime(r.headers["date"]).timestamp()
                except (KeyError, ValueError, TypeError):
                    server = None
                for row in r.json().get("live_datas", []):
                    rows.append((row, sent, got, server))
                beat.write(json.dumps(dict(ts=got, sent=sent, server=server, fast=fast, n=len(ids)))+"\n")
            beat.flush()
            if got is None:
                continue
            states = {}
            for row, sent, got, server in rows:
                event = known.get(row["milestone_id"], (None,))[0]
                if event:
                    states[event] = (row["milestone_id"], row["details"], sent, got, server)
            if broad:
                ranked = sorted((u, e) for e, v in states.items() if (u := urgency(e, v[1])))
                chosen = [states[e][0] for u, e in ranked[:max_events]]
                was, fast = fast, any(u[1] for u, e in ranked)
                reselect = time.time()
                if fast != was:
                    print(f"{datetime.now(timezone.utc):%H:%M:%S} fast={fast} events={[e for u, e in ranked[:max_events]]} "
                          f"fast seconds left {fast_left:.0f}", flush=True)
                if not fast and FLAG.exists():
                    FLAG.unlink()
            wrote = 0
            for event, (mid, s, sent, got, server) in states.items():
                keep = {k: s.get(k) for k in ("away_points", "home_points", "period", "period_remaining_time", "period_type",
                                               "status", "provider", "score_last_updated_ts", "last_updated_ts",
                                               "last_event_created_ts", "last_play", "possession", "winner")}
                key = json.dumps(keep, sort_keys=True)
                prev_poll = last.get(("poll", event))
                last[("poll", event)] = got
                if last.get(event) != key:
                    last[event] = key
                    out.write(json.dumps(dict(ts=got, sent=sent, server=server, prev_poll=prev_poll, fast=fast and not broad,
                                              event=event, state=keep), separators=(",", ":"))+"\n")
                    wrote += 1
            out.flush()
    if FLAG.exists():
        FLAG.unlink()
    print("live poll finished; fast seconds unused", round(fast_left), flush=True)


def live_rows():
    rows = defaultdict(list)
    f = RAW / "live-polls.jsonl"
    if f.exists():
        for line in f.open():
            r = json.loads(line)
            rows[r["event"]].append(r)
    return rows


def live_changes():
    """Score changes whose previous poll of the same event was at most 4 s earlier."""
    out = []
    for event, rs in live_rows().items():
        for c in changes_of(event, rs):
            if c["prevT"] and c["T"]-c["prevT"] <= 4:
                out.append(c)
    return out


def livetape_cmd():
    wanted = {}
    for c in live_changes():
        if c["late"] and c["lead"] <= 6:
            e = c["event"]
            code = e.split("-")[-1][7:]
            lo, hi = wanted.get(e, (None, None, 9e18, 0))[2:]
            wanted[e] = (None, None, min(lo, c["prevT"]-180), max(hi, c["T"]+300))
    with httpx.Client(timeout=40, headers=HEADERS) as client:
        for e in list(wanted):
            res = fetch_result(client, e)
            pair = sides(e, sorted(res)) if res and len(res) == 2 else None
            if not pair:
                print("no markets", e)
                del wanted[e]
                continue
            wanted[e] = (pair[0], pair[1])+wanted[e][2:]
    (RAW / "live-games.json").write_text(json.dumps({e: w[:2] for e, w in wanted.items()}))
    print(len(wanted), "live games with a late close score change")
    download(wanted)


# ------------------------------------------------------------------ analysis

def fee(p):
    return 7*(p/100)*(1-p/100)


class Tape:
    """Both markets of an event restated as one team's YES price."""

    def __init__(self, own, opp):
        rows = [(t, p, n, side == "yes") for t, side, p, n in own["rows"]]
        rows += [(t, 100-p, n, side == "no") for t, side, p, n in opp["rows"]]
        rows.sort()
        self.rows, self.times = rows, [r[0] for r in rows]
        self.buys = [r for r in rows if r[3]]
        self.buy_times = [r[0] for r in self.buys]
        self.start, self.stop = max(own["start"], opp["start"]), min(own["stop"], opp["stop"])

    def last(self, t):
        i = bisect_right(self.times, t)
        return self.rows[i-1][1] if i else None

    def buy(self, t, within=20):
        i = bisect_left(self.buy_times, t)
        if i < len(self.buys) and self.buys[i][0] <= t+within:
            return self.buys[i]
        return None

    def depth(self, t, price, span=5):
        i = bisect_left(self.buy_times, t)
        n = 0
        while i < len(self.buys) and self.buys[i][0] <= t+span:
            if self.buys[i][1] <= price+1:
                n += self.buys[i][2]
            i += 1
        return n


def load_tape(ticker):
    f = tape_file(ticker)
    return json.loads(f.read_text()) if f.exists() else None


def boot(values, games, costs=None, draws=2000, seed=7):
    """Game-clustered interval for the mean (or, with costs, for total net / total cost in %)."""
    costs = costs or [1.0]*len(values)
    by = defaultdict(list)
    for v, k, g in zip(values, costs, games):
        by[g].append((v, k))
    groups = list(by.values())
    if len(groups) < 3:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        pick = [groups[rng.randrange(len(groups))] for _ in groups]
        means.append(sum(v for g in pick for v, k in g)/sum(k for g in pick for v, k in g))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stat(net, games, costs=None):
    """Net cents per contract; with costs, return on money staked (total net / total cost) and cents."""
    if not net:
        return "n=0"
    if costs is None:
        lo, hi = boot(net, games)
        return f"{np.mean(net):+.2f}c [{lo:+.2f},{hi:+.2f}] (n={len(net)}, games={len(set(games))})"
    lo, hi = boot(net, games, costs)
    return (f"{100*sum(net)/sum(costs):+.2f}% [{100*lo:+.2f},{100*hi:+.2f}] = {np.mean(net):+.2f}c a contract "
            f"(n={len(net)}, games={len(set(games))})")


def quantiles(v, qs=(5, 25, 50, 75, 95)):
    return " ".join(f"p{q} {np.percentile(v, q):+.1f}" for q in qs) if len(v) else "none"


def attach(changes, pairs, results):
    """Add the tape, the scoring team's ticker and its settlement to each late close single-team change."""
    out, tapes = [], {}
    for c in changes:
        if not (c["late"] and c["lead"] <= 6 and c["side"] in ("away", "home")) or c["event"] not in pairs:
            continue
        away, home = pairs[c["event"]]
        own, opp = (away, home) if c["side"] == "away" else (home, away)
        if (own, opp) not in tapes:
            a, b = load_tape(own), load_tape(opp)
            tapes[(own, opp)] = Tape(a, b) if a and b else None
        tp = tapes[(own, opp)]
        if tp is None or not tp.rows:
            continue
        res = (results.get(c["event"]) or {}).get(own)
        out.append(dict(c, tape=tp, won=None if res not in ("yes", "no") else res == "yes"))
    return out


def path(say, cs, anchor, label):
    offs = (-30, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20, 30, 45, 60, 120)
    cols = defaultdict(list)
    for c in cs:
        t0 = anchor(c)
        if t0 is None:
            continue
        base = c["tape"].last(t0-60)
        if base is None:
            continue
        for o in offs:
            p = c["tape"].last(t0+o)
            if p is not None:
                cols[o].append(p-base)
    if not cols:
        return
    say(f"      vs {label} (n={len(cols[0])}): "+" ".join(f"{o:+d}s {np.mean(cols[o]):+.2f}" for o in offs))
    total = np.mean(cols[120]) if cols[120] else float("nan")
    if total and total == total:
        say(f"        share of the +120 s move already in the price: at {label} {100*np.mean(cols[0])/total:.0f}%, "
            f"+2 s {100*np.mean(cols[2])/total:.0f}%, +5 s {100*np.mean(cols[5])/total:.0f}%, +15 s {100*np.mean(cols[15])/total:.0f}%")


def simulate(say, cs, anchor, label, delays=(1, 3, 5), order=None):
    for d in delays:
        m60, mset, paid, depth, waits = [], [], [], [], []
        for c in cs:
            t0 = anchor(c)
            if t0 is None:
                continue
            fill = c["tape"].buy(t0+d)
            if fill is None or not 1 <= fill[1] <= 99:
                continue
            t, p = fill[0], fill[1]
            cost = p+fee(p)
            mark = c["tape"].last(t+60)
            if t+60 <= c["tape"].stop+1 and mark is not None:
                m60.append((mark-cost, cost, c["event"]))
            if c["won"] is not None:
                mset.append(((100 if c["won"] else 0)-cost, cost, c["event"]))
            paid.append(p)
            waits.append(t-(t0+d))
            depth.append(c["tape"].depth(t, p))
        tried = sum(1 for c in cs if anchor(c) is not None)
        if not paid:
            say(f"      {label}+{d}s: no fills (tried {tried})")
            continue

        def line(rows):
            return stat([r[0] for r in rows], [r[2] for r in rows], [r[1] for r in rows])
        say(f"      {label}+{d}s: filled {len(paid)}/{tried}, median price {np.median(paid):.0f}c, median wait for a print {np.median(waits):.1f}s, "
            f"contracts bought by takers within 5 s at up to 1c worse: median {np.median(depth):.0f}, p25 {np.percentile(depth, 25):.0f}")
        say(f"        marked 60 s later: {line(m60)}")
        say(f"        held to settlement: {line(mset)}")
        if order and len(order) >= 6:
            half = {g: i < len(order)/2 for i, g in enumerate(order)}
            for name, flag in (("first half of games", True), ("second half of games", False)):
                say(f"          {name}: 60 s {line([r for r in m60 if half.get(r[2]) == flag])}; "
                    f"settlement {line([r for r in mset if half.get(r[2]) == flag])}")


def quoted(say, cs, games, label):
    """Buy at the ask the recorder read in the first pass AFTER the one that showed the score (a real quote)."""
    m60, mset, paid, size, when = [], [], [], [], []
    for c in cs:
        qs = games[c["event"]]["quotes"]
        i = bisect_right([q[0] for q in qs], c["T"]+16)
        if not i:
            continue
        q = qs[i-1]
        p, n = (q[1], q[2]) if c["side"] == "away" else (q[3], q[4])
        if p is None or not 1 <= p <= 99:
            continue
        cost, t = p+fee(p), c["T"]+16
        mark = c["tape"].last(t+60)
        if mark is not None and t+60 <= c["tape"].stop+1:
            m60.append((mark-cost, cost, c["event"]))
        if c["won"] is not None:
            mset.append(((100 if c["won"] else 0)-cost, cost, c["event"]))
        paid.append(p)
        size.append(n or 0)
    if not paid:
        say(f"      {label}: no quotes")
        return
    say(f"      {label}: n={len(paid)}, median ask {np.median(paid):.0f}c, contracts at that ask median {np.median(size):.0f} (p25 {np.percentile(size, 25):.0f})")
    say(f"        last trade 60 s after that: {stat([r[0] for r in m60], [r[2] for r in m60], [r[1] for r in m60])}")
    say(f"        held to settlement: {stat([r[0] for r in mset], [r[2] for r in mset], [r[1] for r in mset])}")


def move_timing(say, cs, visible_lo, visible_hi, label):
    """When did the price make half of its move, against when the score was visible?"""
    early, late, n_all = [], [], 0
    for c in cs:
        lo, hi = visible_lo(c), visible_hi(c)
        if lo is None or hi is None:
            continue
        tp = c["tape"]
        ref = min(lo, c["score_ts"] or lo)-10
        pre, post = tp.last(ref), tp.last(hi+45)
        if pre is None or post is None:
            continue
        n_all += 1
        if post-pre < 3:
            continue
        i = bisect_right(tp.times, ref)
        half = next((r[0] for r in tp.rows[i:] if r[1] >= pre+0.5*(post-pre)), None)
        if half is None:
            continue
        early.append(half-lo)                  # seconds the half-move came after the earliest possible visibility
        late.append(half-hi)                   # ... after the latest possible visibility
    if not early:
        say(f"      {label}: no measurable moves")
        return
    early, late = np.array(early), np.array(late)
    say(f"      {label}: {len(early)} of {n_all} changes moved the price 3c or more in the scoring team's favour")
    say(f"        price half-moved AFTER the score was certainly visible: {100*np.mean(late > 0):.0f}%"
        f" (by 1 s or more {100*np.mean(late >= 1):.0f}%, 3 s or more {100*np.mean(late >= 3):.0f}%, 5 s or more {100*np.mean(late >= 5):.0f}%)")
    say(f"        price half-moved after the earliest moment it could have been visible: {100*np.mean(early > 0):.0f}%"
        f" (by 3 s or more {100*np.mean(early >= 3):.0f}%, 5 s or more {100*np.mean(early >= 5):.0f}%)")
    say(f"        half-move minus latest visibility, s: {quantiles(late)}")
    say(f"        half-move minus earliest visibility, s: {quantiles(early)}")


def report_cmd():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    results = json.loads((RAW / "results.json").read_text()) if (RAW / "results.json").exists() else {}
    games, changes = recorder_changes()
    say("SCORE SPEED: is Kalshi's own live score early enough to buy the scoring team before the price moves?")
    say(f"written {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    say()
    say("1. Recorder (15 s polls of /live_data/batch)")
    span = [c["T"] for c in changes]
    days = (max(span)-min(span))/86400 if span else 0
    say(f"  {len(games)} basketball events, {len(changes)} score changes over {days:.2f} days "
        f"({datetime.fromtimestamp(min(span), timezone.utc):%m-%d %H:%M} to {datetime.fromtimestamp(max(span), timezone.utc):%m-%d %H:%M} UTC)")
    say("  " + ", ".join(f"{NAMES[s]} {sum(1 for g in games if series_of(g) == s)} games" for s in SERIES))
    late = [c for c in changes if c["late"] and c["lead"] <= 6]
    say(f"  late close score changes: {len(late)} in {len(set(c['event'] for c in late))} games; "
        + ", ".join(f"{NAMES[s]} {sum(1 for c in late if series_of(c['event']) == s)}" for s in SERIES))
    say("  seconds from Kalshi's stamps to the recorder poll that first showed the new score (all score changes):")
    for prov in sorted({c["provider"] or "?" for c in changes}):
        cs = [c for c in changes if (c["provider"] or "?") == prov]
        for field in ("score_ts", "upd_ts"):
            v = [c["T"]-c[field] for c in cs if c[field]]
            if v:
                say(f"    {prov:13s} poll - {field:8s} n={len(v):5d} {quantiles(v)}")
        v = [c["prevT"]-c["score_ts"] for c in cs if c["score_ts"] and c["T"]-c["prevT"] < 20]
        if v:
            v = np.array(v)
            say(f"    {prov:13s} stamped before the PREVIOUS poll yet not shown in it: {100*np.mean(v > 0):.0f}% "
                f"(by 2 s or more {100*np.mean(v >= 2):.0f}%, 5 s or more {100*np.mean(v >= 5):.0f}%, 10 s or more {100*np.mean(v >= 10):.0f}%)")
    say()

    # ---- live 2 s measurement
    lc = live_changes()
    lag = {}
    say("2. Live 2 s poll: how long after score_last_updated_ts is a new score visible to a public reader?")
    beats = [json.loads(l) for l in (RAW / "live-beat.jsonl").open()] if (RAW / "live-beat.jsonl").exists() else []
    fastb = [b for b in beats if b.get("fast")]
    if beats:
        skew_hi = min(b["server"]+1-b["sent"] for b in beats if b.get("server"))
        skew_lo = max(b["server"]-b["ts"] for b in beats if b.get("server"))
        skew = (skew_lo+skew_hi)/2
        gaps = np.diff([b["ts"] for b in fastb]) if len(fastb) > 2 else np.array([])
        gaps = gaps[gaps < 10]
        say(f"  polls {len(beats)} ({len(fastb)} at the fast rate, about {len(fastb)*2/60:.0f} minutes); "
            f"round trip median {1000*np.median([b['ts']-b['sent'] for b in beats]):.0f} ms; "
            f"server clock minus this machine's between {skew_lo:+.2f} and {skew_hi:+.2f} s (stamps below are corrected by {skew:+.2f})")
        if len(gaps):
            say(f"  gap between fast polls: median {np.median(gaps):.2f} s, p95 {np.percentile(gaps, 95):.2f} s")
    else:
        skew = 0.0
    say(f"  score changes caught with the previous poll at most 4 s earlier: {len(lc)} in {len(set(c['event'] for c in lc))} games")
    for prov in sorted({c["provider"] or "?" for c in lc}):
        cs = [c for c in lc if (c["provider"] or "?") == prov and c["score_ts"]]
        if not cs:
            say(f"    {prov}: no score stamp")
            continue
        first = np.array([c["T"]+skew-c["score_ts"] for c in cs])
        before = np.array([c["prevT"]+skew-c["score_ts"] for c in cs])
        mid = (first+before)/2-0.5              # the stamp is a whole second: its true time is on average 0.5 s later
        lag[prov] = (float(np.median(mid)), float(np.percentile(mid, 75)))
        say(f"    {prov:13s} n={len(cs)}: first poll showing it minus stamp {quantiles(first)}")
        say(f"    {'':13s}        last poll NOT showing it minus stamp {quantiles(before)}")
        say(f"    {'':13s}        publish delay (midpoint, stamp taken as mid-second): median {np.median(mid):.1f} s, p75 {np.percentile(mid, 75):.1f} s, p90 {np.percentile(mid, 90):.1f} s, max {mid.max():.1f} s")
        upd = np.array([c["T"]+skew-c["upd_ts"] for c in cs if c["upd_ts"]])
        say(f"    {'':13s}        first poll showing it minus last_updated_ts {quantiles(upd)}")
    say()

    # ---- price path and simulation
    pairs = {e: (g["away"], g["home"]) for e, g in games.items()}
    rec = attach(changes, pairs, results)
    order = sorted({c["event"] for c in rec}, key=lambda e: (games[e]["start"] or "", e))
    live_pairs = json.loads((RAW / "live-games.json").read_text()) if (RAW / "live-games.json").exists() else {}
    liv = attach(lc, live_pairs, results)

    def recorder_part(title, cs):
        say(f"  {title}: {len(cs)} changes, {len(set(c['event'] for c in cs))} games")
        if not cs:
            return
        say("    mean scoring-team price (last trade) minus its price 60 s before the anchor, cents:")
        path(say, cs, lambda c: c["score_ts"], "Kalshi score stamp")
        path(say, cs, lambda c: c["prevT"], "poll before (score not yet shown)")
        path(say, cs, lambda c: c["T"], "first 15 s poll showing it")
        say("    when the price moved against when the score was visible:")
        move_timing(say, cs, lambda c: c["prevT"], lambda c: c["T"], "15 s grid alone (visible somewhere in (previous poll, poll])")
        if lag:
            def lo(c):
                d = lag.get(c["provider"])
                return max(c["prevT"], c["score_ts"]) if d and c["score_ts"] else None

            def hi(c):
                d = lag.get(c["provider"])
                return min(c["T"], max(c["prevT"], c["score_ts"]+1+d[1])) if d and c["score_ts"] else None
            move_timing(say, cs, lo, hi, "grid tightened with the stamp (visible between the stamp and the stamp + 1 s + p75 publish delay)")
        say("    buy the scoring team (first taker purchase at or after the time shown), fee paid:")
        simulate(say, cs, lambda c: c["T"], "recorder's own 15 s poll", order=order)
        say("    buy at the ask the recorder itself read (a real quote, not a print; the listing it comes from may itself lag):")
        quoted(say, cs, games, "ask as of the next pass (5-15 s after the score was seen)")
        simulate(say, cs, lambda c: c["score_ts"], "Kalshi score stamp (not achievable: the stamp precedes publication)", delays=(0, 1, 3, 5), order=order)
        for name, idx in (("median", 0), ("p75", 1)):
            if lag:
                simulate(say, cs, lambda c: c["score_ts"]+0.5+lag[c["provider"]][idx] if c["score_ts"] and c["provider"] in lag else None,
                         f"stamp + {name} publish delay measured live", order=order)

    say("3. Recorder games, late close score changes with a trade tape (only one team scored between polls)")
    for s in SERIES:
        cs = [c for c in rec if series_of(c["event"]) == s]
        if not cs:
            continue
        say(f" {NAMES[s]}")
        recorder_part("all score changes", cs)
        recorder_part("baskets (2 or 3 points)", [c for c in cs if c["pts"] in (2, 3)])
        recorder_part("baskets that tied the game or changed/extended a lead of 3 or less", [c for c in cs if c["pts"] in (2, 3) and c["lead"] <= 3])
    if rec:
        per_week = len(rec)/max(days, 0.1)*7
        say(f"  frequency: {len(rec)} late close single-team score changes in {days:.2f} recorded days = about {per_week:.0f} a week "
            f"({len([c for c in rec if c['pts'] in (2, 3)])/max(days, 0.1)*7:.0f} baskets a week) in {len(order)/max(days, 0.1)*7:.0f} games a week (early-season men's college schedule; NBA barely sampled)")
    say()

    say("4. Live 2 s poll games, late close score changes with a trade tape")
    for s in SERIES:
        cs = [c for c in liv if series_of(c["event"]) == s]
        if not cs:
            continue
        say(f" {NAMES[s]}: {len(cs)} changes, {len(set(c['event'] for c in cs))} games")
        for title, sub in (("all score changes", cs), ("baskets (2 or 3 points)", [c for c in cs if c["pts"] in (2, 3)])):
            say(f"  {title}: {len(sub)}")
            if not sub:
                continue
            path(say, sub, lambda c: c["T"]+skew, "first 2 s poll showing it")
            path(say, sub, lambda c: c["prevT"]+skew, "last 2 s poll not showing it")
            path(say, sub, lambda c: c["score_ts"], "Kalshi score stamp")
            move_timing(say, sub, lambda c: c["prevT"]+skew, lambda c: c["T"]+skew, "2 s grid")
            simulate(say, sub, lambda c: c["T"]+skew, "first 2 s poll showing it", delays=(0, 1, 3, 5))
    if not liv:
        say("  none")
    (OUT / "report.txt").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("tape", "live", "livetape", "report"))
    parser.add_argument("--minutes", type=float, default=40, help="total minutes of 2 s polling")
    parser.add_argument("--hours", type=float, default=6, help="stop watching after this long")
    parser.add_argument("--events", type=int, default=12)
    args = parser.parse_args()
    if args.command == "tape":
        tape_cmd()
    elif args.command == "live":
        live_cmd(args.minutes, args.hours, min(args.events, 12))
    elif args.command == "livetape":
        livetape_cmd()
    else:
        report_cmd()


if __name__ == "__main__":
    main()
