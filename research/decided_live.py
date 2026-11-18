"""Live order-book check of markets that the score has already decided.

Reads Kalshi's own live game state (milestones and live data, as live_games.py
does) every few seconds. When a game's score has passed an "over" line, or a
period that a market covers has ended, the market's ORDER BOOK is read (never
the cached listing quote) and the displayed ask and size are appended to a
log, on each pass while an offer is displayed and every few minutes otherwise.

Public reads only; no orders. At most one request at a time, paced, with
back-off on HTTP 429.
"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, UTC

OUT = ROOT / "data/processed/decided"
GAMES = {"KXNBAGAME": "NBA", "KXNHLGAME": "NHL", "KXNCAAMBGAME": "NCAAMB"}
# suffix -> (kind, last period the market covers; 0 = whole game). "over" kinds settle on a count that only rises.
SUFFIX = {
    "NBA": {"GAME": ("winner", 0), "SPREAD": ("spread", 0), "TOTAL": ("total", 0), "TEAMTOTAL": ("team_total", 0), "1HTOTAL": ("total", 2), "1HTEAMTOTAL": ("team_total", 2),
            "1QTOTAL": ("total", 1), "1QTEAMTOTAL": ("team_total", 1), "1H": ("winner", 2), "1HSPREAD": ("spread", 2),
            "1Q": ("winner", 1), "1QSPREAD": ("spread", 1), "2Q": ("winner", -2), "2QSPREAD": ("spread", -2),
            "2QTOTAL": ("ptotal", -2), "3Q": ("winner", -3), "3QSPREAD": ("spread", -3), "3QTOTAL": ("ptotal", -3)},
    "NHL": {"GAME": ("winner", 0), "SPREAD": ("spread", 0), "TOTAL": ("total", 0), "TEAMTOTAL": ("team_total", 0), "1PTOTAL": ("total", 1), "1P": ("winner", 1),
            "1PSPREAD": ("spread", 1), "2P": ("winner", -2), "2PSPREAD": ("spread", -2), "2PTOTAL": ("ptotal", -2)},
    "NCAAMB": {"GAME": ("winner", 0), "SPREAD": ("spread", 0), "TOTAL": ("total", 0), "TEAMTOTAL": ("team_total", 0), "1HTOTAL": ("total", 1), "1HTEAMTOTAL": ("team_total", 1),
               "1H": ("winner", 1), "1HSPREAD": ("spread", 1)},
}
PACE = .45
last_call = [0.0]


def get(client, path, **params):
    for attempt in range(8):
        wait = PACE-(time.time()-last_call[0])
        if wait > 0:
            time.sleep(wait)
        last_call[0] = time.time()
        try:
            r = client.get(config.KALSHI_PROD_BASE+path, params=params)
        except httpx.RequestError:
            time.sleep(1+attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1.5+1.5*attempt)
            continue
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()
    return {}


def milestones(client):
    since = (datetime.now(UTC)-timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    found, cursor = {}, ""
    while True:
        page = get(client, "/milestones", limit=200, category="Sports", minimum_start_date=since, cursor=cursor)
        for m in page.get("milestones", []):
            for event in m.get("primary_event_tickers") or []:
                if event.split("-")[0] in GAMES:
                    found[event] = m["id"]
        cursor = page.get("cursor")
        if not cursor or not page.get("milestones"):
            return found


def listing(client, keys):
    """Open markets of every derived series for the given game keys: ticker -> description."""
    out = {}
    for league, suffixes in SUFFIX.items():
        mine = {k for lg, k in keys if lg == league}
        if not mine:
            continue
        for suffix, (kind, period) in suffixes.items():
            series, cursor = "KX"+league+suffix, ""
            while True:
                page = get(client, "/markets", series_ticker=series, status="open", limit=1000, cursor=cursor)
                for m in page.get("markets", []):
                    key = m["event_ticker"].split("-", 1)[1]
                    if key not in mine:
                        continue
                    tail = m["ticker"].rsplit("-", 1)[1]
                    team = re.match(r"[A-Z]*", tail).group(0)
                    out[m["ticker"]] = dict(league=league, key=key, series=series, kind=kind, period=period,
                                            strike=m.get("floor_strike"), team=team, title=m.get("title"))
                cursor = page.get("cursor")
                if not cursor:
                    break
    return out


def book(client, ticker):
    """Best ask and size on each side, and the size offered at 99c or better."""
    data = get(client, f"/markets/{ticker}/orderbook").get("orderbook_fp") or {}
    out = {}
    for side, other in (("yes", "no_dollars"), ("no", "yes_dollars")):
        bids = sorted(((round(100*float(p)), float(s)) for p, s in data.get(other) or []), reverse=True)
        out[side+"_ask"] = 100-bids[0][0] if bids else None
        out[side+"_size"] = bids[0][1] if bids else 0
        out[side+"_size_to_99"] = sum(s for p, s in bids if 100-p <= 99)
        out[side+"_levels"] = [[100-p, s] for p, s in bids[:4]]
    return out


FINAL = ("closed", "final", "complete", "completed", "finished")


def teams_side(m):
    teams = m["key"][7:]
    return "home" if teams.endswith(m["team"]) and m["team"] else "away" if teams.startswith(m["team"]) and m["team"] else None


def decided(m, state, ends):
    """Which side the live state has decided, if any: ('yes'|'no'|'either', reason)."""
    home, away = state.get("home_points"), state.get("away_points")
    if home is None or away is None:
        return None
    period = int(state.get("period") or 0)
    status = str(state.get("status", "")).lower()
    final = status in FINAL and home != away          # the game is over: every full-game market is decided
    if final and m["period"] == 0:
        if teams_side(m) is None and m["kind"] != "total":
            return None
        mine, other = (home, away) if teams_side(m) == "home" else (away, home)
        count = home+away if m["kind"] == "total" else mine if m["kind"] == "team_total" else mine-other
        line = 0 if m["kind"] == "winner" else m["strike"]
        if line is None:
            return None
        return ("yes" if count > line else "no", f"final {away}-{home}: {count} vs {line}")
    over = final or period > abs(m["period"]) or (period == abs(m["period"]) and status in ("halftime", "intermission", "end_of_period"))
    teams = m["key"][7:]
    if m["kind"] in ("total", "team_total") and m["strike"] is not None:
        if m["kind"] == "total":
            now = home+away
        elif teams.endswith(m["team"]):
            now = home
        elif teams.startswith(m["team"]):
            now = away
        else:
            return None
        if m["period"] == 0 or not over:
            return ("yes", f"count {now} > {m['strike']}") if now > m["strike"] else None
        end = ends.get((m["key"], m["period"]))          # score when the period ended, if we saw it
        if end:
            count = sum(end) if m["kind"] == "total" else end[0] if teams.endswith(m["team"]) else end[1]
            return ("yes" if count > m["strike"] else "no", f"period over at {count} vs {m['strike']}")
        return ("either", "period over, period score not observed")
    if m["period"] != 0 and over:
        return ("either", "period over")
    return None


def run(hours, every):
    OUT.mkdir(parents=True, exist_ok=True)
    stop = time.time()+hours*3600
    known, markets, refreshed, ends, done, empty, prev = {}, {}, 0, {}, {}, {}, {}
    seen_live, ended, read_before = set(), {}, set()
    log = OUT / f"live-{datetime.now(UTC):%Y-%m-%d}.jsonl"
    with httpx.Client(timeout=30) as client, log.open("a") as out:
        while time.time() < stop:
            began = time.time()
            try:
                if began-refreshed > 600 or not known:
                    known = milestones(client)
                ids = {v: k for k, v in known.items()}
                live = {}
                for i in range(0, len(ids), 40):
                    chunk = list(ids)[i:i+40]
                    for row in get(client, "/live_data/batch", milestone_ids=chunk).get("live_datas", []):
                        live[ids[row["milestone_id"]]] = row["details"]
                seen = time.time()
                playing = {}
                for e, s in live.items():
                    status = str(s.get("status", "")).lower()
                    if status in ("inprogress", "halftime", "intermission", "live"):
                        playing[e] = s
                        seen_live.add(e)
                    elif status in FINAL and e in seen_live:      # keep a finished game for ten minutes
                        ended.setdefault(e, time.time())
                        if time.time()-ended[e] < 600:
                            playing[e] = s
                for e, s in playing.items():           # remember the score at each period end
                    key, p = e.split("-", 1)[1], int(s.get("period") or 0)
                    was = prev.get(e)
                    if was and was[0] < p:
                        ends.setdefault((key, was[0]), (was[1], was[2]))
                    if str(s.get("status", "")).lower() in ("halftime", "intermission") and p:
                        ends[(key, p)] = (s.get("home_points"), s.get("away_points"))
                    prev[e] = (p, s.get("home_points"), s.get("away_points"))
                if began-refreshed > 600:
                    markets = listing(client, {(GAMES[e.split("-")[0]], e.split("-", 1)[1]) for e in playing})
                    refreshed = began
                    print(f"{datetime.now(UTC):%H:%M:%S} live games {len(playing)} open derived markets {len(markets)}", flush=True)
                state_of = {e.split("-", 1)[1]: s for e, s in playing.items()}
                rows = 0
                # Order of reading: lines the score has just passed (never read before), then lines within one
                # score of being passed (the book as it stands BEFORE the deciding play), then a few re-reads.
                fresh, near, again = [], [], []
                for ticker, m in markets.items():
                    s = state_of.get(m["key"])
                    if s is None:
                        continue
                    verdict = decided(m, s, ends)
                    if verdict:
                        if ticker not in read_before:
                            fresh.append((ticker, m, s, verdict))
                        elif done.get(ticker, 0) <= time.time():
                            again.append((ticker, m, s, verdict))
                    elif m["kind"] == "total" and m["period"] == 0 and m["strike"] is not None and str(s.get("status", "")).lower() == "inprogress":
                        gap = m["strike"]-(s.get("home_points") or 0)-(s.get("away_points") or 0)
                        if 0 < gap <= (1 if m["league"] == "NHL" else 3):
                            near.append((ticker, m, s, ("yes", f"near: {gap} short of the line")))
                for ticker, m, s, verdict in fresh+near[:8]+again[:10]:
                    read_before.add(ticker) if not verdict[1].startswith("near") else None
                    b = book(client, ticker)
                    if verdict[1].startswith("near"):
                        out.write(json.dumps(dict(ts=time.time(), state_seen=seen, ticker=ticker, league=m["league"], kind=m["kind"],
                                                  scope=m["period"], strike=m["strike"], side="yes", reason=verdict[1],
                                                  period=s.get("period"), clock=s.get("period_remaining_time"), status=s.get("status"),
                                                  home=s.get("home_points"), away=s.get("away_points"),
                                                  score_ts=s.get("score_last_updated_ts"), state_ts=s.get("last_updated_ts"), **b),
                                             separators=(",", ":"))+"\n")
                        continue
                    side = verdict[0]
                    asks = [b[x+"_ask"] for x in (("yes", "no") if side == "either" else (side,))]
                    # an undetermined side counts only when it is priced as the near-certain one
                    offered = any(a is not None and (50 if side == "either" else 0) < a <= 99 for a in asks)
                    row = dict(ts=time.time(), state_seen=seen, ticker=ticker, league=m["league"], kind=m["kind"], scope=m["period"],
                               strike=m["strike"], side=side, reason=verdict[1], period=s.get("period"),
                               clock=s.get("period_remaining_time"), status=s.get("status"), home=s.get("home_points"),
                               away=s.get("away_points"), score_ts=s.get("score_last_updated_ts"),
                               state_ts=s.get("last_updated_ts"), **b)
                    if not offered:
                        empty[ticker] = empty.get(ticker, 0)+1
                        if empty[ticker] >= 3:        # look once at the market itself, then stop polling it
                            info = get(client, f"/markets/{ticker}").get("market") or {}
                            row.update(market_status=info.get("status"), close_time=info.get("close_time"),
                                       result=info.get("result"), volume=info.get("volume_fp"))
                            # no offer: look again every few minutes (sooner once the game is over)
                            done[ticker] = time.time()+(45 if str(s.get("status", "")).lower() in FINAL else 150)
                            empty[ticker] = 2
                    else:
                        empty[ticker] = 0
                    out.write(json.dumps(row, separators=(",", ":"))+"\n")
                    out.flush()
                    rows += 1
                if rows:
                    print(f"{datetime.now(UTC):%H:%M:%S} books read {rows} finished {len(done)}", flush=True)
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
                print(f"{datetime.now(UTC):%H:%M:%S} pass skipped ({type(exc).__name__}: {exc})", flush=True)
            time.sleep(max(.5, every-(time.time()-began)))


def summary():
    """What the log shows: offers on decided sides, the resting bids there, and books read just before and after a line was passed."""
    from collections import Counter, defaultdict
    from statistics import median
    rows = [json.loads(line) for f in sorted(OUT.glob("live-*.jsonl")) for line in f.open()]
    near = [r for r in rows if r["reason"].startswith("near")]
    by = defaultdict(list)
    for r in rows:
        if not r["reason"].startswith("near"):
            by[r["ticker"]].append(r)
    print(f"decided markets {len(by)}, book reads {sum(map(len, by.values()))}, over {(rows[-1]['ts']-rows[0]['ts'])/60:.0f} minutes")
    groups, offers = defaultdict(lambda: dict(markets=0, reads=0, offered=0, bids=[], ages=[])), []
    for ticker, rs in by.items():
        first = rs[0]
        g = groups[first["league"], first["kind"], "game" if first["scope"] == 0 else "period", first["reason"].split()[0].rstrip(":")]
        g["markets"] += 1
        g["reads"] += len(rs)
        hit = False
        for r in rs:
            for side in (("yes", "no") if r["side"] == "either" else (r["side"],)):
                ask = r[side+"_ask"]
                if ask is not None and (50 if r["side"] == "either" else 0) < ask <= 99:
                    hit = True
                    offers.append((ticker, r["reason"], r["period"], r["clock"], side, ask, r[side+"_size"], r[side+"_size_to_99"]))
        g["offered"] += hit
        if first["side"] != "either":
            other = "no" if first["side"] == "yes" else "yes"
            if first[other+"_ask"] is not None:        # the other side's ask is a bid for the decided side
                g["bids"].append((100-first[other+"_ask"], first[other+"_size"]))
            if first.get("score_ts") and first["reason"].startswith("count"):
                g["ages"].append(first["ts"]-first["score_ts"])
    for name, g in sorted(groups.items()):
        bids = g["bids"]
        print(f"  {' '.join(name):34} markets {g['markets']:3} reads {g['reads']:4} with an offer <=99c {g['offered']:2} | "
              f"bid for the decided side in {len(bids):3}: median {median(b[0] for b in bids) if bids else 0:.0f}c x {median(b[1] for b in bids) if bids else 0:.0f}")
    for o in offers[:40]:
        print("  OFFER", o)
    before = defaultdict(list)
    for r in near:
        before[r["ticker"]].append(r)
    pairs = []
    for ticker, ns in before.items():
        if ticker in by:
            after = min(by[ticker], key=lambda r: r["ts"])
            earlier = [n for n in ns if n["ts"] < after["ts"]]
            if earlier:
                pairs.append((max(earlier, key=lambda r: r["ts"]), after))
    live = [p for p in pairs if p[0]["yes_size_to_99"] > 0]
    print(f"lines read both just before and just after the score passed them: {len(pairs)}; with an offer <=99c before: {len(live)}; "
          f"with one after: {sum(1 for _, a in pairs if a['yes_ask'] is not None and a['yes_ask'] <= 99)}")
    for b, a in live:
        print(f"  {a['ticker']} {b['ts']-a['ts']:.0f}s before: ask {b['yes_ask']}c, {b['yes_size_to_99']:.0f} offered up to 99c ({b['reason'][6:]}) | "
              f"{a['ts']-(a.get('score_ts') or a['ts']):.0f}s after the score: ask {a['yes_ask']}, bid for it {100-a['no_ask'] if a['no_ask'] else None}c x {a['no_size']:.0f} "
              f"(period {a['period']}, {a['clock']} left)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--every", type=float, default=5, help="seconds between live-state reads")
    parser.add_argument("--summary", action="store_true", help="print what the log shows and exit")
    args = parser.parse_args()
    if args.summary:
        summary()
    else:
        run(args.hours, args.every)


if __name__ == "__main__":
    main()
