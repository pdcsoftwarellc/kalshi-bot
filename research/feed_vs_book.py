"""Does Kalshi's own live game feed lead Kalshi's own order book?

`run` polls /live_data/batch about once a second for every in-progress tennis,
basketball and hockey game, keeps a rolling order-book snapshot (every ~3s) for
the few most liquid events, and when the feed shows a score event reads that
event's book at once and again at +1, +2, +5, +10 and +30 seconds. Every poll,
event and book read is appended as a JSON line under data/raw/feed-vs-book/.

`report` re-scores those logs: how far the event winner's price had already
moved before the feed showed the event, how far it moved afterwards, and a
paper trade (buy the winner at the displayed ask right after the feed event,
sell at the displayed bid later, taker fees on both legs).

Public reads only; no orders.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import heapq
import itertools
import json
from pathlib import Path
import statistics
import sys
import threading
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, UTC

OUT = ROOT / "data/raw/feed-vs-book"
TYPES = {"tennis_tournament_singles": "tennis", "tennis_tournament_doubles": "tennis",
         "basketball_game": "basketball", "hockey_match": "hockey"}
DEAD = {"closed", "ended", "complete", "completed", "cancelled", "canceled", "postponed", "finished", "final"}
IDLE = {"not_started", "scheduled", "created", "sch", "", "none"}
OFFSETS = (0, 1, 2, 5, 10, 30)
POINT_INDEX = {0: 0, 15: 1, 30: 2, 40: 3, 50: 4}


# ----------------------------------------------------------------- state and events

def clock_seconds(text):
    try:
        parts = [float(x) for x in str(text).split(":")]
    except ValueError:
        return None
    return parts[0]*60+parts[1] if len(parts) == 2 else parts[0] if len(parts) == 1 else None


def normalise(sport, d):
    """Reduce a live-data details blob to the fields that define the score."""
    status = str(d.get("status", "")).lower()
    if sport == "tennis":
        c1, c2 = d.get("competitor1_id"), d.get("competitor2_id")
        side = {c1: 1, c2: 2}
        r1, r2 = d.get("competitor1_round_scores") or [], d.get("competitor2_round_scores") or []
        return dict(status=status, p1=d.get("competitor1_current_round_score"), p2=d.get("competitor2_current_round_score"),
                    adv=side.get(d.get("advantage")) or (d.get("advantage") or None), srv=side.get(d.get("server")),
                    g1=[r.get("score") for r in r1], g2=[r.get("score") for r in r2],
                    o1=[r.get("outcome") for r in r1], win=side.get(d.get("winner")) or (d.get("winner") or None),
                    ms=d.get("match_status"))
    return dict(status=status, a=d.get("away_points"), h=d.get("home_points"), period=d.get("period"),
                ptype=d.get("period_type"), clock=clock_seconds(d.get("period_remaining_time")),
                score_ts=d.get("score_last_updated_ts"), upd_ts=d.get("last_updated_ts"), win=d.get("winner") or None)


def is_live(state):
    return state["status"] not in DEAD and state["status"] not in IDLE


def point_index(value, tiebreak):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 4 if str(value).upper() in ("A", "AD", "ADV") else None
    return value if tiebreak else POINT_INDEX.get(value)


def tennis_events(prev, cur):
    n = min(len(prev["g1"]), len(cur["g1"]), len(prev["g2"]), len(cur["g2"]))
    try:
        gains = [(i, cur["g1"][i]-prev["g1"][i], cur["g2"][i]-prev["g2"][i]) for i in range(n)]
    except TypeError:
        return []
    moved = [g for g in gains if g[1] or g[2]]
    if moved:
        if len(moved) != 1 or sorted(moved[0][1:]) != [0, 1]:
            return []  # more than one game between polls, or a correction
        i, d1, _ = moved[0]
        winner = 1 if d1 else 2
        tiebreak = prev["g1"][i] == prev["g2"][i] and prev["g1"][i] >= 6
        set_done = len(cur["g1"]) > i+1 or (cur["o1"][i] if i < len(cur["o1"]) else "ongoing") != "ongoing"
        over = cur["status"] in DEAD or bool(cur["win"])
        serve = "tiebreak" if tiebreak else "unknown" if prev["srv"] is None else "hold" if prev["srv"] == winner else "break"
        kind = "match" if over else "set" if set_done else serve if serve in ("hold", "break") else "game"
        return [dict(type=kind, win=winner, serve=serve)]
    if not prev["g1"] or len(prev["g1"]) != len(cur["g1"]):
        return []
    tiebreak = prev["g1"][-1] == prev["g2"][-1] and (prev["g1"][-1] or 0) >= 6 or "tie" in str(prev.get("ms") or "").lower()
    a1, a2, b1, b2 = (point_index(v, tiebreak) for v in (prev["p1"], prev["p2"], cur["p1"], cur["p2"]))
    if None in (a1, a2, b1, b2):
        return []
    if not tiebreak:
        a1, a2 = a1+(prev["adv"] == 1 and a1 == 3), a2+(prev["adv"] == 2 and a2 == 3)
        b1, b2 = b1+(cur["adv"] == 1 and b1 == 3), b2+(cur["adv"] == 2 and b2 == 3)
    if abs(b1-a1)+abs(b2-a2) != 1:
        return []
    winner = 1 if (b1-b2) > (a1-a2) else 2
    return [dict(type="tb_point" if tiebreak else "point", win=winner,
                 serve="unknown" if prev["srv"] is None else "server" if prev["srv"] == winner else "returner")]


def team_events(sport, prev, cur):
    try:
        da, dh = cur["a"]-prev["a"], cur["h"]-prev["h"]
    except TypeError:
        return []
    if (da > 0) == (dh > 0) or da < 0 or dh < 0:
        return []
    winner, margin = (1 if da else 2), abs(prev["a"]-prev["h"])
    if sport == "hockey":
        return [dict(type="goal", win=winner, margin=margin, pts=da or dh)]
    final = final_period(prev)
    clock = prev["clock"] if prev["clock"] is not None else 9999
    kind = "clutch_basket" if final and clock <= 180 and margin <= 6 else \
        "late_basket" if final and clock <= 360 and margin <= 10 else "basket"
    return [dict(type=kind, win=winner, margin=margin, pts=da or dh, clock=prev["clock"])]


def final_period(state):
    period = state.get("period") or 0
    return period >= (2 if state.get("ptype") == "half" else 4)


def events(sport, prev, cur):
    return tennis_events(prev, cur) if sport == "tennis" else team_events(sport, prev, cur)


# ----------------------------------------------------------------- recorder

class Recorder:
    def __init__(self, minutes, slots, roll, tag):
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / f"{tag}-{datetime.now(UTC):%Y%m%dT%H%M%S}.jsonl"
        self.out = self.path.open("a")
        self.stop = time.time()+minutes*60
        self.slots, self.roll = slots, roll
        self.lock = threading.Lock()
        self.pause_until = 0.0
        self.counts = defaultdict(int)
        self.games = {}      # milestone id -> dict(sport, event, title, first, market info)
        self.state = {}      # milestone id -> normalised state
        self.tracked = {}    # milestone id -> ticker
        self.heap, self.seq = [], itertools.count()
        self.rolling = set()  # tickers with a rolling read scheduled
        self.last_book = {}  # ticker -> (recv time, mid cents)
        self.event_ids = itertools.count(1)

    def write(self, **row):
        with self.lock:
            self.out.write(json.dumps(row, separators=(",", ":"))+"\n")

    def get(self, client, kind, path, **params):
        """One attempt. Returns (json or None, send time, receive time); a 429 pauses every reader."""
        wait = self.pause_until-time.time()
        if wait > 0:
            time.sleep(wait)
        sent = time.time()
        try:
            r = client.get(config.KALSHI_PROD_BASE+path, params=params)
        except httpx.HTTPError:
            self.counts[kind+"_error"] += 1
            return None, sent, time.time()
        got = time.time()
        if r.status_code == 429:
            self.counts[kind+"_429"] += 1
            self.pause_until = max(self.pause_until, got+3)
            return None, sent, got
        if r.status_code != 200:
            self.counts[kind+"_error"] += 1
            return None, sent, got
        self.counts[kind+"_ok"] += 1
        return r.json(), sent, got

    # ---- slow thread: which games exist, which markets they map to
    def discover(self):
        with httpx.Client(timeout=20) as client:
            refreshed = 0
            while time.time() < self.stop:
                if time.time()-refreshed > 600:
                    self.find_games(client)
                    refreshed = time.time()
                for mid, game in list(self.games.items()):
                    state = self.state.get(mid)
                    stale = time.time()-game.get("quoted", 0) > 600
                    if state and is_live(state) and stale and time.time() < self.stop:
                        self.map_markets(client, mid, game)
                        time.sleep(.6)
                self.choose()
                time.sleep(20)

    def find_games(self, client):
        since = (datetime.now(UTC)-timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        soon = (datetime.now(UTC)+timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
        cursor = ""
        while time.time() < self.stop:
            page, _, _ = self.get(client, "meta", "/milestones", limit=200, category="Sports", minimum_start_date=since, cursor=cursor)
            if page is None:
                time.sleep(1)
                continue
            for m in page.get("milestones", []):
                tickers = m.get("primary_event_tickers") or []
                if m["type"] in TYPES and tickers and (m.get("start_date") or "9") <= soon and m["id"] not in self.games:
                    d = m.get("details") or {}
                    self.games[m["id"]] = dict(sport=TYPES[m["type"]], event=d.get("main_game_event_ticker") or tickers[0], title=m.get("title"),
                                               first=d.get("first_competitor_id") or d.get("away_team_id"), new=True)
            cursor = page.get("cursor")
            if not cursor or not page.get("milestones"):
                break
            time.sleep(.4)
        # One sweep over new games to drop the finished ones before the fast loop polls them.
        fresh = [mid for mid, g in self.games.items() if g.pop("new", False)]
        for i in range(0, len(fresh), 40):
            data, _, _ = self.get(client, "meta", "/live_data/batch", milestone_ids=fresh[i:i+40])
            seen = {row["milestone_id"]: normalise(self.games[row["milestone_id"]]["sport"], row["details"])
                    for row in (data or {}).get("live_datas", [])}
            for mid in fresh[i:i+40]:
                if mid in seen and seen[mid]["status"] in DEAD:
                    self.games[mid]["dead"] = True
                elif mid not in seen and data is not None:
                    self.games[mid]["dead"] = True
            time.sleep(.5)

    def map_markets(self, client, mid, game):
        data, _, _ = self.get(client, "meta", "/markets", event_ticker=game["event"])
        if data is None:
            return
        game["quoted"] = time.time()
        markets = [m for m in data.get("markets", []) if m.get("status") == "active"]
        if len(markets) != 2:
            return
        code = game["event"].split("-")[-1]
        sides = {}
        for m in markets:
            suffix = m["ticker"].split("-")[-1]
            strike = set((m.get("custom_strike") or {}).values())
            if game["first"] and strike:
                sides[m["ticker"]] = (1 if game["first"] in strike else 2, "strike")
            elif code.startswith(suffix) != code.endswith(suffix):
                sides[m["ticker"]] = (1 if code.startswith(suffix) else 2, "suffix")
        if sorted(s for s, _ in sides.values()) != [1, 2]:
            return
        best = max(markets, key=lambda m: float(m.get("volume_fp") or 0))
        if "ticker" in game:  # keep the first choice so the rolling book series stays on one market
            best = next((m for m in markets if m["ticker"] == game["ticker"]), best)
        bid, ask = float(best.get("yes_bid_dollars") or 0)*100, float(best.get("yes_ask_dollars") or 1)*100
        game.update(ticker=best["ticker"], yes_side=sides[best["ticker"]][0], how=sides[best["ticker"]][1],
                    volume=sum(float(m.get("volume_fp") or 0) for m in markets), mid=(bid+ask)/2, spread=ask-bid,
                    yes_name=best.get("yes_sub_title"))
        self.write(k="map", t=time.time(), mid=mid, sport=game["sport"], ev=game["event"], title=game["title"], tk=game["ticker"],
                   yes_side=game["yes_side"], how=game["how"], yes_name=game["yes_name"], volume=game["volume"])

    def choose(self):
        """Pick the events whose books are kept warm: close late basketball, one hockey game, then tennis by volume."""
        late, hockey, tennis, rest = [], [], [], []
        for mid, game in self.games.items():
            state = self.state.get(mid)
            if "ticker" not in game or not state or not is_live(state):
                continue
            known = self.last_book.get(game["ticker"])
            mid_price = known[1] if known and time.time()-known[0] < 60 else game["mid"]
            open_price = 6 <= mid_price <= 94
            score = game["volume"]*(1 if open_price else .001)*(1 if game["spread"] <= 4 else .1)
            if mid in self.tracked and open_price:
                score *= 1000  # stay on a match once chosen, so its events keep a before-book
            if game["sport"] == "tennis":
                tennis.append((score, mid))
            elif game["sport"] == "hockey":
                hockey.append((score, mid))
            else:
                close = state["a"] is not None and abs(state["a"]-state["h"]) <= 10
                if final_period(state) and close and (state["clock"] or 0) <= 420:
                    late.append((score, mid))
                elif close:
                    rest.append((score*(10 if final_period(state) else 1), mid))
        picks = [mid for _, mid in sorted(late, reverse=True)[:3]]
        picks += [mid for _, mid in sorted(hockey, reverse=True)[:1]]
        picks += [mid for _, mid in sorted(tennis, reverse=True)]
        picks += [mid for _, mid in sorted(hockey, reverse=True)[1:]+sorted(rest, reverse=True)]
        chosen = {mid: self.games[mid]["ticker"] for mid in picks[:self.slots]}
        if set(chosen) != set(self.tracked):
            self.write(k="track", t=time.time(), tracked=[[mid, tk, self.games[mid]["title"]] for mid, tk in chosen.items()])
        self.tracked = chosen
        for tk in chosen.values():
            if tk not in self.rolling:
                self.rolling.add(tk)
                self.push(time.time(), 1, tk, "roll", None, None)

    # ---- book thread
    def push(self, when, priority, ticker, why, eid, offset):
        with self.lock:
            heapq.heappush(self.heap, (when, priority, next(self.seq), ticker, why, eid, offset))

    def books(self):
        last_sent = 0.0
        with httpx.Client(timeout=10) as client:
            while time.time() < self.stop+35:
                with self.lock:
                    due = self.heap[0] if self.heap and self.heap[0][0] <= time.time() else None
                    if due:
                        heapq.heappop(self.heap)
                if not due:
                    time.sleep(.02)
                    continue
                when, priority, _, ticker, why, eid, offset = due
                if why == "roll":
                    if ticker not in self.tracked.values() or time.time() > self.stop:
                        self.rolling.discard(ticker)
                        continue
                    known = self.last_book.get(ticker)
                    if known and time.time()-known[0] < self.roll*.6:   # an event read just refreshed it
                        self.push(known[0]+self.roll, 1, ticker, "roll", None, None)
                        continue
                gap = (.2 if why == "ev" else .34)-(time.time()-last_sent)
                if gap > 0:
                    time.sleep(gap)
                last_sent = time.time()
                data, sent, got = self.get(client, "book", f"/markets/{ticker}/orderbook")
                if why == "roll":
                    self.push(max(sent+self.roll, time.time()), 1, ticker, "roll", None, None)
                if data is None:
                    self.write(k="lost", t=got, tk=ticker, why=why, eid=eid, off=offset)
                    continue
                book = data.get("orderbook_fp") or {}
                yes = [[round(float(p)*100), float(s)] for p, s in (book.get("yes_dollars") or [])[-3:]][::-1]
                no = [[round(float(p)*100), float(s)] for p, s in (book.get("no_dollars") or [])[-3:]][::-1]
                bid, ask = (yes[0][0] if yes else 0), (100-no[0][0] if no else 100)
                self.last_book[ticker] = (got, (bid+ask)/2)
                self.write(k="book", t0=round(sent, 3), t1=round(got, 3), tk=ticker, why=why, eid=eid, off=offset,
                           due=round(when, 3), yb=yes, nb=no)

    # ---- fast loop: the feed
    def run(self):
        threads = [threading.Thread(target=self.discover, daemon=True), threading.Thread(target=self.books, daemon=True)]
        for t in threads:
            t.start()
        last_ok, last_print = None, time.time()
        with httpx.Client(timeout=10) as client:
            while time.time() < self.stop:
                began = time.time()
                ids = [mid for mid, g in self.games.items() if not g.get("dead")]
                ids.sort(key=lambda mid: (mid not in self.tracked, not (mid in self.state and is_live(self.state[mid]))))
                # Live and tracked games every pass; the rest (not yet started) share the remaining places in rotation.
                core = [mid for mid in ids if mid in self.tracked or (mid in self.state and is_live(self.state[mid]))][:60]
                others = [mid for mid in ids if mid not in core]
                turn = int(began) % max(1, (len(others)+9)//10)
                batch = core+others[turn*10:turn*10+10]
                if not batch:
                    time.sleep(1)
                    continue
                data, sent, got = self.get(client, "live", "/live_data/batch", milestone_ids=batch)
                if data is None:
                    self.write(k="poll", t0=round(sent, 3), t1=round(got, 3), ok=False)
                    continue
                fired = 0
                for row in data.get("live_datas", []):
                    mid = row["milestone_id"]
                    game = self.games.get(mid)
                    if not game:
                        continue
                    cur = normalise(game["sport"], row["details"])
                    prev = self.state.get(mid)
                    self.state[mid] = cur
                    if cur["status"] in DEAD:
                        game["dead"] = True
                    core_now = {k: v for k, v in cur.items() if k not in ("upd_ts", "clock")}
                    if prev is None or {k: v for k, v in prev.items() if k not in ("upd_ts", "clock")} == core_now:
                        continue
                    seen_prev = prev.get("_seen") if prev else None
                    for event in events(game["sport"], prev, cur) if mid in core else []:
                        eid = next(self.event_ids)
                        ticker = self.tracked.get(mid)
                        if ticker and event["type"] != "basket":
                            for offset in OFFSETS:
                                self.push(got+offset, 0, ticker, "ev", eid, offset)
                        fired += 1
                        self.write(k="event", id=eid, t=round(got, 3), t_send=round(sent, 3), t_prev=last_ok, mid=mid, ev=game["event"],
                                   sport=game["sport"], tk=ticker, yes_side=game.get("yes_side"), prev=prev, cur=cur, **event)
                    self.write(k="state", t=round(got, 3), t_prev=last_ok, mid=mid, sport=game["sport"], cur=cur)
                self.write(k="poll", t0=round(sent, 3), t1=round(got, 3), ok=True, n=len(batch), fired=fired)
                last_ok = round(got, 3)
                if time.time()-last_print > 60:
                    last_print = time.time()
                    live = sum(1 for s in self.state.values() if is_live(s))
                    print(f"{datetime.now(UTC):%H:%M:%S} live {live} tracked {len(self.tracked)} {dict(self.counts)}", flush=True)
                time.sleep(max(0, 1-(time.time()-began)))
        time.sleep(33)  # let the +30s reads of the last events land
        self.write(k="end", t=time.time(), counts=dict(self.counts))
        self.out.close()
        print(f"wrote {self.path}\n{dict(self.counts)}")


# ----------------------------------------------------------------- report

def fee(price):
    return 7*price/100*(1-price/100)  # cents per contract at a price in cents


def side_quote(book, yes_side, winner):
    """(bid, ask, bid size, ask size) in cents for the event winner, from a book row of the tracked market."""
    yes_bid, yes_size = book["yb"][0] if book["yb"] else (0, 0)
    no_bid, no_size = book["nb"][0] if book["nb"] else (0, 0)
    if winner == yes_side:
        return yes_bid, 100-no_bid, yes_size, no_size
    return no_bid, 100-yes_bid, no_size, yes_size


def mean(xs):
    return sum(xs)/len(xs) if xs else float("nan")


def clustered(rows, key):
    """(per-event mean, mean of per-match means, events, matches) for rows that have the key."""
    have = [r for r in rows if r.get(key) is not None]
    by = defaultdict(list)
    for r in have:
        by[r["mid"]].append(r[key])
    return mean([r[key] for r in have]), mean([mean(v) for v in by.values()]), len(have), len(by)


def score(paths, gap_limit=3.0, before_age=5.0):
    books, evs, counts, polls, lost, tracked_titles = defaultdict(list), [], defaultdict(int), [], 0, {}
    for path in paths:
        for line in path.open():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            kind = row["k"]
            if kind == "book":
                books[row["tk"]].append(row)
            elif kind == "event":
                evs.append(row)
            elif kind == "poll":
                polls.append(row)
            elif kind == "lost":
                lost += 1
            elif kind == "end":
                for name, n in row["counts"].items():
                    counts[name] += n
            elif kind == "map":
                tracked_titles[row["mid"]] = (row["sport"], row["title"])
    for rows in books.values():
        rows.sort(key=lambda r: r["t1"])

    def at(ticker, when, tolerance, mode):
        rows = books.get(ticker, [])
        if mode == "before":      # latest book received by `when`
            pick = [r for r in rows if when-tolerance <= r["t1"] <= when]
            return pick[-1] if pick else None
        if mode == "first":       # first book requested after `when`
            pick = [r for r in rows if r["t0"] >= when-.05 and r["t1"] <= when+tolerance]
            return pick[0] if pick else None
        pick = [r for r in rows if abs(r["t1"]-when) <= tolerance]
        return min(pick, key=lambda r: abs(r["t1"]-when)) if pick else None

    scored, dropped = [], defaultdict(int)
    for e in evs:
        if e["type"] in ("basket", "match"):
            dropped[f"{e['type']} (not scored)"] += 1
            continue
        if not e["tk"]:
            dropped["book not tracked"] += 1
            continue
        if e["t_prev"] is None or e["t"]-e["t_prev"] > gap_limit:
            dropped["poll gap over 3s"] += 1
            continue
        before = at(e["tk"], e["t"], before_age, "before")
        if not before:
            dropped["no before book within 5s"] += 1
            continue
        now = at(e["tk"], e["t"], 3.5, "first")
        if not now:
            dropped["no book within 3.5s after"] += 1
            continue

        def quote(book):
            return side_quote(book, e["yes_side"], e["win"]) if book else None

        def mid(book):
            q = quote(book)
            return (q[0]+q[1])/2 if q else None
        r = dict(mid=e["mid"], type=e["type"], sport=e["sport"], t=e["t"], id=e["id"], gap=e["t"]-e["t_prev"],
                 before_age=e["t"]-before["t1"], read_delay=now["t1"]-e["t"], serve=e.get("serve"))
        m0, mb = mid(now), mid(before)
        r["m0"] = m0
        for h, tol in ((-60, 4), (-30, 4), (-10, 4)):
            b = at(e["tk"], e["t"]+h, tol, "before")
            r[f"pre{-h}"] = mb-mid(b) if b else None       # move from then to the before-snapshot
        r["lead"] = m0-mb                                   # before-snapshot -> first read after the feed event
        for h, tol in ((1, .8), (2, 1), (5, 1.5), (10, 2), (30, 3), (60, 4)):
            b = at(e["tk"], e["t"]+h, tol, "near")
            r[f"post{h}"] = mid(b)-m0 if b else None
            if h == 30:
                later = b
        bid0, ask0, _, size0 = quote(now)
        r["spread"], r["size"], r["ask"] = ask0-bid0, size0, ask0
        if size0 >= 10 and 1 <= ask0 <= 99 and later:
            bid30 = quote(later)[0]
            r["net30"] = bid30-ask0-fee(ask0)-fee(bid30)
            r["gross30"] = bid30-ask0
        scored.append(r)

    # Hold to the next comparable event in the same match (game-level with game-level, point with point).
    family = lambda kind: "point" if kind in ("point", "tb_point") else "score"
    order = sorted(scored, key=lambda r: r["t"])
    raw = {e["id"]: e for e in evs}
    for i, r in enumerate(order):
        if r.get("net30") is None:
            continue
        nxt = next((x for x in order[i+1:] if x["mid"] == r["mid"] and family(x["type"]) == family(r["type"])), None)
        if not nxt or nxt["t"]-r["t"] > 1200:
            continue
        e, book = raw[nxt["id"]], at(raw[nxt["id"]]["tk"], nxt["t"], 3.5, "first")
        if book and e["tk"] == raw[r["id"]]["tk"]:
            bid = side_quote(book, e["yes_side"], raw[r["id"]]["win"])[0]
            r["net_next"] = bid-r["ask"]-fee(r["ask"])-fee(bid)
            r["hold_next"] = nxt["t"]-r["t"]

    ok = [p for p in polls if p["ok"]]
    span = (max(p["t1"] for p in polls)-min(p["t0"] for p in polls))/60 if polls else 0
    gaps = sorted(b["t1"]-a["t1"] for a, b in zip(ok, ok[1:]))
    print(f"files {len(paths)}  span {span:.1f} min  live-data polls ok {len(ok)} failed {len(polls)-len(ok)}"
          f"  median poll gap {statistics.median(gaps) if gaps else float('nan'):.2f}s  p95 {gaps[int(len(gaps)*.95)] if gaps else float('nan'):.2f}s")
    print(f"request counts {dict(counts)}  book reads lost {lost}  book reads ok {sum(len(v) for v in books.values())}")
    seen = defaultdict(set)
    for e in evs:
        seen[(e["sport"], e["type"])].add(e["mid"])
    print("\nall feed events seen (any game, before filters): " +
          ", ".join(f"{s}/{t} {sum(1 for e in evs if e['sport'] == s and e['type'] == t)} in {len(m)} games" for (s, t), m in sorted(seen.items())))
    print("dropped: "+", ".join(f"{k} {v}" for k, v in sorted(dropped.items())))
    print("matches scored: "+"; ".join(sorted({f"{tracked_titles.get(r['mid'], ('?', '?'))[1]} [{r['sport']}]" for r in scored})))

    groups = [("tennis point", lambda r: r["type"] == "point"), ("tennis tiebreak point", lambda r: r["type"] == "tb_point"),
              ("tennis hold", lambda r: r["type"] == "hold"), ("tennis break", lambda r: r["type"] == "break"),
              ("tennis set won", lambda r: r["type"] == "set"),
              ("tennis game-level (hold+break+set)", lambda r: r["type"] in ("hold", "break", "set", "game")),
              ("basketball clutch basket", lambda r: r["type"] == "clutch_basket"),
              ("basketball late basket", lambda r: r["type"] == "late_basket"),
              ("hockey goal", lambda r: r["type"] == "goal")]
    print("\nLEAD/LAG: move in the event winner's mid price, cents (per-event mean | mean of per-match means)")
    print(f"{'event':36s} {'n':>4s} {'mat':>3s} {'-60s>bef':>13s} {'-30s>bef':>13s} {'-10s>bef':>13s} {'bef>feed':>13s}"
          f" {'feed>+2s':>13s} {'feed>+5s':>13s} {'feed>+10s':>13s} {'feed>+30s':>13s} {'feed>+60s':>13s}")
    for name, test in groups:
        rows = [r for r in scored if test(r)]
        if not rows:
            continue
        cells = []
        for key in ("pre60", "pre30", "pre10", "lead", "post2", "post5", "post10", "post30", "post60"):
            a, b, _, _ = clustered(rows, key)
            cells.append(f"{a:+6.2f}|{b:+6.2f}")
        print(f"{name:36s} {len(rows):4d} {len({r['mid'] for r in rows}):3d} "+" ".join(cells))
    print("  bef = last book received before the feed poll that showed the event (at most 5s old); feed = first book read after it."
          "\n  '-30s>bef' is how far the book had already moved in the half minute before the feed showed the event.")
    print(f"  timing: median before-book age {statistics.median([r['before_age'] for r in scored]) if scored else float('nan'):.2f}s,"
          f" median delay feed->book read {statistics.median([r['read_delay'] for r in scored]) if scored else float('nan'):.2f}s,"
          f" median poll gap at events {statistics.median([r['gap'] for r in scored]) if scored else float('nan'):.2f}s")

    print("\nPATH: winner's mid minus its mid at the first read after the feed event, cents (per-event mean, n)")
    steps = (-60, -30, -20, -15, -12, -9, -6, -3, 3, 6, 10, 20, 30, 60)
    print(f"{'event':36s} "+" ".join(f"{h:+9d}s" for h in steps))
    for name, test in groups:
        rows = [r for r in scored if test(r)]
        if not rows:
            continue
        cells = []
        for h in steps:
            got = [(at(raw[r["id"]]["tk"], r["t"]+h, 1.6, "near"), r) for r in rows]
            vals = [sum(side_quote(b, raw[r["id"]]["yes_side"], raw[r["id"]]["win"])[:2])/2-r["m0"] for b, r in got if b]
            cells.append(f"{mean(vals):+6.2f}/{len(vals):<3d}")
        print(f"{name:36s} "+" ".join(cells))

    print("\nPAPER TRADE: buy winner at displayed ask right after the feed event (size >= 10), sell at displayed bid, fees 0.07*p*(1-p) per leg")
    print(f"{'event':36s} {'n':>4s} {'mat':>3s} {'gross30':>8s} {'net30':>7s} {'net30/match':>11s} {'win%':>5s} {'spread':>6s} {'med size':>8s}"
          f" | {'n':>4s} {'net to next':>11s} {'/match':>7s} {'win%':>5s} {'hold s':>6s}")
    for name, test in groups:
        rows = [r for r in scored if test(r) and r.get("net30") is not None]
        if not rows:
            continue
        net, net_m, n, mats = clustered(rows, "net30")
        nxt = [r for r in rows if r.get("net_next") is not None]
        nn, nn_m, _, _ = clustered(nxt, "net_next")
        print(f"{name:36s} {n:4d} {mats:3d} {mean([r['gross30'] for r in rows]):+8.2f} {net:+7.2f} {net_m:+11.2f}"
              f" {100*mean([r['net30'] > 0 for r in rows]):5.0f} {mean([r['spread'] for r in rows]):6.2f} {statistics.median([r['size'] for r in rows]):8.0f}"
              f" | {len(nxt):4d} {nn:+11.2f} {nn_m:+7.2f} {100*mean([r['net_next'] > 0 for r in nxt]) if nxt else float('nan'):5.0f}"
              f" {statistics.median([r['hold_next'] for r in nxt]) if nxt else float('nan'):6.0f}")
    thin = sum(1 for r in scored if r["size"] < 10)
    print(f"  events with under 10 contracts at the ask right after the feed event: {thin} of {len(scored)}")
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run")
    run.add_argument("--minutes", type=float, default=60)
    run.add_argument("--slots", type=int, default=6, help="events whose order book is kept warm")
    run.add_argument("--roll", type=float, default=3.0, help="seconds between rolling reads of each tracked book")
    run.add_argument("--tag", default="run")
    rep = sub.add_parser("report")
    rep.add_argument("files", nargs="*", type=Path, help="default: every run-*.jsonl under data/raw/feed-vs-book")
    args = parser.parse_args()
    if args.mode == "run":
        Recorder(args.minutes, args.slots, args.roll, args.tag).run()
    else:
        score(args.files or sorted(OUT.glob("run-*.jsonl")))


if __name__ == "__main__":
    main()
