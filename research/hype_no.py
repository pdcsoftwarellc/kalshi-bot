"""Take the NO side in retail "hype" markets: is YES overpriced where casual traders mostly buy YES?

Research only. Public Kalshi reads, no orders, no authentication.

The idea. In tennis match markets 93% of pre-match taker volume bought YES
(research/thin_book.py). If casual traders mostly buy YES, markets framed as an
exciting YES (a player scoring 25+, a politician saying a word, a film scoring
above 90 on Rotten Tomatoes) should have YES overpriced and NO underpriced.
Rule tested: take NO and hold to settlement.

Fixed in advance (before any result was read)

* Families: sports player props, parlay packs, "mention" markets, entertainment
  charts and scores, reality TV, recurring politics. Crypto, indices, FX, rates,
  gas and weather are left out (studied in research/far_strikes.py and others).
  Only markets settled yes or no with close_time from 2026-09-18 to 2026-11-18
  (the recent tier, where the batch candlestick endpoint works).
* Reference times: 24h, 6h and 1h before the event clock T0 (below). The quote
  is the close of the last hourly bar ending at or before the reference
  instant, which is rounded down to the hour, and at most 24 hours old. A row
  needs the market open and not yet closed, and a two-sided quote
  1 <= bid < ask <= 99. The "quoted" sample needs a spread of at most 10c;
  "every" takes any two-sided quote (hype books are often wide, and the rule
  pays the bid side of the spread anyway).
* Bands on the YES midpoint: 1-5, 5-10, 10-20, 20-35, 35-50, 50-65, 65-80,
  80-90, 90-95, 95-99.
* Taker rule: buy NO at 100 - YES bid, pay the taker fee, hold to settlement.
  Fee per contract 0.07*m*p*(1-p) with the series' fee multiplier m, charged on
  the order and rounded UP to the cent, so it is shown for orders of 3, 10 and
  100 contracts (headline: 10). Return = total net / total paid (price + fee).
* Maker rule: rest a NO bid one cent above the best NO bid (a YES offer one
  cent below the YES ask), only when the spread is at least 2c. Filled only if
  a LATER hourly bar, before T0, prints a YES trade strictly above our offer.
  No fee where the series fee type is "quadratic"; 0.0175*p*(1-p) where it is
  "quadratic_with_maker_fees". Adverse selection = filled against unfilled
  orders, both valued at settlement at the same price.
* "Buy NO on everything": every market of the family with a two-sided quote at
  the reference time, whatever its price.
* Dependence. Markets of one game or one speech move together. Intervals are
  bootstraps over EVENT GROUPS (all markets of one game / speech / film across
  series, e.g. points, rebounds and threes of one NBA game are one group) and,
  separately, over calendar DAYS (US Eastern date of T0). The wider is quoted.
* Halves: event groups ordered by T0, first half and second half of groups.
* Multiplicity. A cell (family x band x reference time) passes when the
  first-half day-clustered interval of the taker return is above zero AND the
  second half is positive AND the neighbouring bands (same family and time)
  are positive on the whole sample. Chance benchmark: outcomes re-drawn under
  fair pricing with the same dependence (one shared shock per event group,
  sized to reproduce the observed over-dispersion of event sums; marginal
  probability = the midpoint), the whole table recomputed.

Decided after looking at the listings and the first tables (say so when
quoting the results)

* Event clock T0. Most of these markets close EARLY when the thing happens (a
  player reaches 25 points), so "hours before this market's close" would
  select on the outcome. T0 is one time per EVENT, never per market:
  - sports props: T0 = expected_expiration_time - LEAD[sport]. Kalshi sets
    expected_expiration_time before the game and does not revise it on an
    early close. LEAD is measured from the listings (`leads`): per game, the
    earliest early close of a market that resolved YES (a YES can only close
    early once play is under way), as hours before expected expiration; LEAD
    is the 98th percentile over games, rounded up to the half hour (NBA and
    NFL 6h, MLB 4h, NHL 3.5h). The allowance varies by up to an hour between
    games, so T0 is at or before the start: "1h before" means one to about
    two and a half hours before the start, never in play.
  - charts, scores, reality TV, politics: expected_expiration_time is not a
    schedule there, so T0 = the most common close time of the event's markets
    (ten-minute buckets, earliest on a tie), which is the scheduled close.
  - mention markets: the exchange closes the event hours (sometimes a day)
    after the speech, so the close is no clock. T0 = the beginning of the
    first hourly bar in which any word of the event was resolved (a YES bid of
    97c or more on a market bid at 85c or less earlier in the window), or the
    common close if none was. Known only afterwards, but the same for every
    word of the event, so it cannot favour YES or NO, and a trader knows the
    scheduled start. Every reference time and every maker fill is then before
    the speech. `report --clock close` uses the common close instead (quotes
    and fills during and after the speech) as a sensitivity.
* Mirror control for the maker rule: the same order on the other side (a YES
  bid one cent above the best YES bid, filled by a later print strictly below
  it). In a wide book both sides would earn the spread; only the difference
  says YES is overpriced.
* Pooled groups (all sports props, all mentions, everything else) for the
  calibration and trading tables.
* Archive holdout for the mention families, the only place where anything
  showed: markets of the nine largest mention series that closed 2026-06-01 to
  2026-09-17, one market in two per event, 520 markets, one request each.

Commands (each resumable; --minutes bounds the network time of that call)

    discover  GET /series per category -> families.json
    list      settled markets per series (slim), fee types
    leads     offline: measure LEAD per sport, print the family table
    bars      hourly bars around T0 through the batch endpoint (--cap markets
              per family as whole events; --thin N adds one market in N from
              every other event)
    flow      public trades (the 1,000 most recent) for 20 random markets per
              family; for mention families also every trade in the 25 hours
              before the event clock for --sample markets (default 40)
    archive   the mention holdout from the /historical endpoints
    report    offline -> data/processed/hype-no/report.txt and summary.json

Network: one request at a time, at most one per 1.5 seconds, 30 seconds or
more of back-off on HTTP 429 (a live trader shares the limit). Raw slim caches
under data/raw/hype-no/, with a request log in download-log.json.

    .venv/bin/python research/hype_no.py discover list --minutes 9
    .venv/bin/python research/hype_no.py leads
    .venv/bin/python research/hype_no.py bars --minutes 9 --cap 3000
    .venv/bin/python research/hype_no.py flow --minutes 9
    .venv/bin/python research/hype_no.py archive --minutes 9
    .venv/bin/python research/hype_no.py report
    .venv/bin/python research/hype_no.py report --clock close
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, midnight, timestamp

RAW = ROOT / "data/raw/hype-no"
OUT = ROOT / "data/processed/hype-no"
START, END = midnight(date(2026, 9, 18)), midnight(date(2026, 11, 19))
RECENT = dict(status="settled", limit=1000, min_close_ts=START, max_close_ts=END)
PACE = 1.5
EASTERN = ZoneInfo("America/New_York")
REFS = (24, 6, 1)
BANDS = ((1, 5), (5, 10), (10, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 90), (90, 95), (95, 99))
SIZES = (3, 10, 100)
MAX_PAGES = 40                       # listing pages per series (1,000 markets each)
RNG = np.random.default_rng(23)

SPORTS = {
    "NBA points": ["KXNBAPTS"],
    "NBA rebounds assists threes": ["KXNBAREB", "KXNBAAST", "KXNBA3PT"],
    "NBA other props": ["KXNBA2D", "KXNBA3D", "KXNBAPRA", "KXNBASTL", "KXNBABLK"],
    "NFL touchdown scorers": ["KXNFLANYTD", "KXNFLFIRSTTD", "KXNFL2TD", "KXNFLTD"],
    "NFL yards and receptions": ["KXNFLRECYDS", "KXNFLRSHYDS", "KXNFLPASSYDS", "KXNFLREC", "KXNFLPASSTDS", "KXNFLRRYDS", "KXNFLPASSINT"],
    "NHL scorers": ["KXNHLGOAL", "KXNHLFIRSTGOAL", "KXNHLPTS", "KXNHLAST"],
    "MLB player props": ["KXMLBHR", "KXMLBKS", "KXMLBHIT", "KXMLBHRR"],
    "Soccer goalscorers": ["KXEPLGOAL", "KXUCLGOAL", "KXLALIGAGOAL", "KXUEFANLGOAL"],
    "Parlay packs": ["KXNFLPREPACKSGP", "KXNCAAFPREPACKSGP", "KXNFLPREPACK", "KXNFLPREPACK2ML", "KXNFLPREPACK3ML", "KXNFLPREPACK1HFT",
                     "KXNCAAMBSGP"],
}
POLITICS = ["KXTRUMPAPPROVE", "KXAPRPOTUS", "KX538APPROVE", "KXEOWEEK", "KXEPSTEIN", "KXTRUTHSOCIAL", "KXTRUMPPHOTO", "KXLAGODAYS",
            "KXTRUMPACT", "KXTRUMPMEET", "KXELONTWEETS"]

last_request = [0.0]
deadline = [float("inf")]
log = Counter()


class OutOfTime(Exception):
    pass


# ---------------------------------------------------------------- network and files

def read(client, path, params):
    """One paced public read. Parsed JSON, or None on a 4xx other than 429."""
    for pause in (30, 60, 120, 240, None):
        if time.time() > deadline[0]:
            raise OutOfTime
        wait = last_request[0]+PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        last_request[0] = time.time()
        log["requests"] += 1
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params, headers={"User-Agent": "research-client"})
        except httpx.RequestError:
            resp = None
        if resp is not None and resp.status_code == 200:
            return resp.json()
        if resp is not None and resp.status_code != 429 and resp.status_code < 500:
            log[f"http_{resp.status_code}"] += 1
            return None
        log["http_429" if resp is not None and resp.status_code == 429 else "server_or_network_error"] += 1
        if pause is None:
            raise RuntimeError("read failed after long back-off: "+path)
        time.sleep(pause)


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    temp.replace(path)


def finish_log(step, began):
    old = load(RAW/"download-log.json", [])
    old.append(dict(step=step, seconds=round(time.time()-began), **log))
    dump(RAW/"download-log.json", old)
    print(f"[{step}] {dict(log)} in {time.time()-began:.0f}s", flush=True)


# ---------------------------------------------------------------- discover

def mention_family(ticker):
    if "EARNINGS" in ticker:
        return "Mentions earnings calls"
    if re.search(r"NBA|NFL|MLB|NCAA|SNF|TNF|MNF|FIGHT|NHL|CFB|WNBA|UFC|ATHLETE|LEBRON|WC|WOMENTION|GOLF|PGA|F1", ticker):
        return "Mentions sports broadcasts"
    if re.search(r"TRUMP|DJT|PRESMENTION|LASTWORDCOUNT|CRYPTOSUMMIT", ticker):
        return "Mentions Trump"
    if re.search(r"FOXNEWS|WORLDNEWS|LASTWORD|LATENIGHT|SNL|FTN|60MIN|KIMMEL|PSAKI|COLBERT|MTP|MADDOW|SURVIVOR|SOUTHPARK|LOVEISL|NEWSNATION|"
                 r"WEEKNIGHT|FNS|AWARD|ENTMENTION|MRBEAST|SWIFT|DRAKE|CANDACE|TBPN|SHIRLEY", ticker):
        return "Mentions TV and media"
    return "Mentions politicians and officials"


def entertainment_family(s):
    t, title = s["ticker"], s["title"].lower()
    if t.startswith("KXRT") or "rotten" in title or t == "KXMC":
        return "Rotten Tomatoes scores"
    if re.search(r"box office|gross|opening weekend", title):
        return "Box office"
    if "NETFLIX" in t:
        return "Netflix charts"
    if re.search(r"ELIMINATION|SURVIVOR|BIGBROTHER|LIUSA|LIUK|DWTS|DANCING|TRAITORS|BACHELOR|LOVEISLAND|IDOL|BEASTGAMES", t):
        return "Reality TV"
    if re.search(r"SPOTIFY|ALBUM|SONG|BILLBOARD|BB|TOPARTIST|YT|TOPMONTHLY|ARTIST|STREAM|CHART|WEEKSNUM1", t):
        return "Music charts"
    if re.search(r"APPRANK|GOOGLESEARCH|ELONTWEETS|VIEWERS", t):
        return "Apps search and social counts"
    return "Entertainment other"


def discover(client):
    families = defaultdict(list)
    meta = {}
    for category in ("Sports", "Entertainment", "Politics", "Mentions", "Social", "Culture"):
        data = read(client, "/series", dict(category=category, include_volume="true")) or {}
        rows = data.get("series") or []
        print(f"{category}: {len(rows)} series", flush=True)
        for s in rows:
            s["_volume"] = float(s.get("volume_fp") or 0)
            meta[s["ticker"]] = dict(title=s["title"], fee_type=s.get("fee_type"), fee_multiplier=s.get("fee_multiplier"),
                                     lifetime_volume=s["_volume"], frequency=s.get("frequency"), category=category)
        if category == "Sports":
            for name, tickers in SPORTS.items():
                families[name] += [t for t in tickers if t in meta]
        elif category == "Mentions":
            for s in rows:
                if s["_volume"] >= 5e5:
                    families[mention_family(s["ticker"])].append(s["ticker"])
        elif category == "Entertainment":
            for s in rows:
                if s["_volume"] >= 1e6 and s.get("frequency") != "annual":
                    families[entertainment_family(s)].append(s["ticker"])
        elif category == "Politics":
            families["Recurring politics"] += [t for t in POLITICS if t in meta]
        else:
            for s in rows:
                if s["_volume"] >= 1e5:
                    families["Social and viral"].append(s["ticker"])
    for t in POLITICS:
        if t in meta and t not in families["Recurring politics"]:
            families["Recurring politics"].append(t)
            for name in list(families):
                if name != "Recurring politics" and t in families[name]:
                    families[name].remove(t)
    dump(RAW/"families.json", dict(families=families, meta={t: meta[t] for ts_ in families.values() for t in ts_}))
    for name, ts_ in families.items():
        print(f"  {name:36} {len(ts_):4} series", flush=True)


def families():
    return load(RAW/"families.json", dict(families={}, meta={}))


# ---------------------------------------------------------------- list

def stamp(m, key):
    try:
        return int(timestamp(m[key]))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def slim(m):
    return dict(ticker=m["ticker"], event=m["event_ticker"], name=(m.get("yes_sub_title") or m.get("title") or "")[:60],
                title=(m.get("title") or "")[:70], open=stamp(m, "open_time"), close=stamp(m, "close_time"),
                exp=stamp(m, "expected_expiration_time"), occ=stamp(m, "occurrence_datetime"), settle=stamp(m, "settlement_ts"),
                result=m.get("result"), volume=float(m.get("volume_fp") or 0), early=bool(m.get("can_close_early")),
                kind=m.get("strike_type"), floor=m.get("floor_strike"))


def list_all(client):
    fam = families()
    order = [t for ts_ in fam["families"].values() for t in ts_]
    for n, series in enumerate(order):
        path = RAW/"markets"/f"{series}.json"
        have = load(path, None)
        if have is not None and have["done"]:
            continue
        have = dict(markets=[], done=False, pages=0, truncated=False)
        cursor, seen = "", set()
        while True:
            data = read(client, "/markets", dict(RECENT, series_ticker=series, **({"cursor": cursor} if cursor else {}))) or {}
            have["pages"] += 1
            have["markets"] += [slim(m) for m in data.get("markets", []) if m.get("result") in ("yes", "no")]
            cursor = data.get("cursor")
            if not cursor or cursor in seen or not data.get("markets"):
                break
            if have["pages"] >= MAX_PAGES:
                have["truncated"] = True
                break
            seen.add(cursor)
        have["done"] = True
        dump(path, have)
        if have["markets"] or n % 25 == 0:
            print(f"  {n+1}/{len(order)} {series:30} {len(have['markets']):6} settled markets, {have['pages']} pages"
                  f"{' TRUNCATED' if have['truncated'] else ''}", flush=True)


def markets_of(series):
    return load(RAW/"markets"/f"{series}.json", dict(markets=[]))["markets"]


# ---------------------------------------------------------------- event clock

def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def group_of(family, m):
    """All markets of one game (across series) or one speech / film / chart date."""
    if family in SPORTS:
        return family.split()[0]+":"+m["event"].split("-", 1)[-1]
    return m["event"]


ARCHIVE = "Mentions before Sep 18 (holdout)"
ARCHIVE_FROM = midnight(date(2026, 6, 1))
ARCHIVE_FAMILIES = ("Mentions Trump", "Mentions TV and media", "Mentions politicians and officials")
ARCHIVE_CAP, ARCHIVE_THIN, ARCHIVE_PAGES = 520, 2, 8


def family_names(fam):
    return list(fam["families"])+([ARCHIVE] if (RAW/"archive-series.json").exists() else [])


def family_markets(name, fam=None):
    fam = fam or families()
    out = []
    archive = name == ARCHIVE
    for series in (load(RAW/"archive-series.json", []) if archive else fam["families"].get(name, [])):
        meta = fam["meta"][series]
        for m in (load(RAW/"markets-archive"/f"{series}.json", dict(markets=[]))["markets"] if archive else markets_of(series)):
            if m["exp"] is None or m["close"] is None or m["open"] is None:
                continue
            out.append(dict(m, series=series, family=name, group=group_of(name, m), mult=float(meta.get("fee_multiplier") or 1),
                            maker_fee=meta.get("fee_type") == "quadratic_with_maker_fees"))
    return out


def clock_key(m):
    """Markets that share one allowance convention: a whole sport, or one non-sports series."""
    return m["family"].split()[0] if m["family"] in SPORTS else m["series"]


def measure_lead(ms):
    """Hours between T0 and expected_expiration_time (see the module docstring), and the groups it was measured on."""
    first = {}
    for m in ms:
        if m["result"] == "yes" and m["close"] < m["exp"]-600:      # a YES can only close early once the event is under way
            first[m["group"]] = max(first.get(m["group"], 0), m["exp"]-m["close"])
    if len(first) < 5:
        return None, len(first)
    return math.ceil(2*float(np.percentile(list(first.values()), 98))/3600)/2, len(first)


def leads():
    fam = families()
    every = {name: family_markets(name, fam) for name in family_names(fam)}
    by_key, lead = defaultdict(list), {}
    for name, ms in every.items():
        if name in SPORTS:
            for m in ms:
                by_key[clock_key(m)].append(m)
    for key, ms in by_key.items():
        lead[key] = measure_lead(ms)[0] or 0.0
    dump(OUT/"families.json", dict(table={}, lead=lead))
    out = {}
    print(f"{'family':34} {'series':>6} {'markets':>8} {'events':>6} {'days':>4} {'YES%':>5} {'med vol':>8} {'mean vol':>9} {'tot vol M':>9} "
          f"{'LEAD h':>6} {'open->T0 h 10/50/90%':>22}  fee types | examples")
    for name in every:
        ms = with_clock(name, fam)
        if not ms:
            out[name] = dict(markets=0, series_listed=len(fam["families"].get(name, [])))
            print(f"{name:34} {0:6} {0:8}   no settled markets in the window ({len(fam['families'].get(name, []))} series listed)")
            continue
        vols = np.array([m["volume"] for m in ms])
        ahead = np.percentile([(m["t0"]-m["open"])/3600 for m in ms], [10, 50, 90])
        fees = Counter(f'{fam["meta"][m["series"]]["fee_type"]} x{fam["meta"][m["series"]]["fee_multiplier"]}' for m in ms)
        live = sorted({m["series"] for m in ms})
        top = sorted(ms, key=lambda m: -m["volume"])
        examples = [f'{m["title"]} / {m["name"]}' if m["name"] not in m["title"] else m["title"] for m in top[:len(top)//2:max(1, len(top)//6)]][:3]
        out[name] = dict(series_with_markets=live, markets=len(ms), groups=len({m["group"] for m in ms}), days=len({m["day"] for m in ms}),
                         yes_share=float(np.mean([m["result"] == "yes" for m in ms])), median_volume=float(np.median(vols)), mean_volume=float(vols.mean()),
                         total_volume=float(vols.sum()), lead_hours=lead.get(clock_key(ms[0])) if name in SPORTS else None,
                         open_to_t0_hours=[float(v) for v in ahead], fee_types=dict(fees), examples=examples)
        o = out[name]
        print(f"{name:34} {len(live):6} {len(ms):8} {o['groups']:6} {o['days']:4} {100*o['yes_share']:5.1f} {o['median_volume']:8.0f} {o['mean_volume']:9.0f} "
              f"{o['total_volume']/1e6:9.1f} {o['lead_hours'] if o['lead_hours'] is not None else 'close':>6} {ahead[0]:7.1f}{ahead[1]:7.1f}{ahead[2]:8.1f}  {dict(fees)} | {examples}")
    dump(OUT/"families.json", dict(table=out, lead=lead))
    return out


CLOCK = ["start"]                    # "start": mention events are clocked by their first resolved word; "close": by the event's close


def with_clock(name, fam=None, table=None, clock=None):
    """Markets of a family with the event clock t0 and its US Eastern day (see the module docstring)."""
    lead = load(OUT/"families.json", {}).get("lead", {})
    ms = family_markets(name, fam)
    if name in SPORTS:
        for m in ms:
            m["t0"] = int(m["exp"]-3600*lead.get(clock_key(m), 0.0))
    else:
        closes = defaultdict(Counter)
        for m in ms:
            closes[m["group"]][m["close"]//600] += 1
        mode = {g: 600*min(c, key=lambda k: (-c[k], k)) for g, c in closes.items()}
        for m in ms:
            m["t0"] = mode[m["group"]]
        if name.startswith("Mentions") and (clock or CLOCK[0]) == "start":
            # The speech or show starts hours (sometimes a day) before the exchange closes the event. Its start is taken as the
            # beginning of the first hourly bar in which any word of the event was resolved: a YES bid of 97c or more on a market
            # that had been bid at 85c or less earlier in the window.
            have = load(RAW/"bars"/f"{slug(name)}.json", {})
            first = {}
            for m in ms:
                low = 100
                for bar in have.get(m["ticker"], []):
                    if bar[1] is None:
                        continue
                    low = min(low, bar[1])
                    if bar[1] >= 97 and low <= 85:
                        first[m["group"]] = min(first.get(m["group"], bar[0]-3600), bar[0]-3600)
                        break
            for m in ms:
                m["started"] = m["group"] in first
                m["t0"] = min(m["t0"], first.get(m["group"], m["t0"]))
    for m in ms:
        m["day"] = str(datetime.fromtimestamp(m["t0"], EASTERN).date())
    return ms


def pick_groups(ms, cap):
    """Whole event groups in a fixed pseudo-random order until `cap` markets are reached."""
    by = defaultdict(list)
    for m in ms:
        if m["open"] <= m["t0"]-3600:                     # opened at least an hour before the event
            by[m["group"]].append(m)
    out = []
    for g in sorted(by, key=lambda g: hashlib.sha256(g.encode()).hexdigest()):
        if len(out) >= cap:
            break
        out += by[g]
    return out


# ---------------------------------------------------------------- flow (public trades)

def flow(client, only, sample=None):
    fam = families()
    for name in fam["families"]:
        if only and name not in only:
            continue
        path = RAW/"trades"/f"{slug(name)}.json"
        have = load(path, {})
        ms = [m for m in with_clock(name, fam, clock="close") if m["volume"] >= 200]
        ms.sort(key=lambda m: hashlib.sha256(("flow"+m["ticker"]).encode()).hexdigest())
        try:
            for m in ms[:sample or FLOW_SAMPLE]:
                if m["ticker"] in have:
                    continue
                rows, cursor = [], ""
                for _ in range(FLOW_PAGES):
                    data = read(client, "/markets/trades", dict(ticker=m["ticker"], limit=1000, **({"cursor": cursor} if cursor else {}))) or {}
                    for t in data.get("trades") or []:
                        price = round(100*float(t["yes_price_dollars"]), 2) if "yes_price_dollars" in t else float(t["yes_price"])
                        rows.append([int(timestamp(t["created_time"])), price, float(t.get("count_fp", t.get("count", 0))),
                                     1 if (t.get("taker_side") or t.get("taker_outcome_side")) == "yes" else 0])
                    cursor = data.get("cursor")
                    if not cursor or not data.get("trades"):
                        break
                have[m["ticker"]] = dict(t0=m["t0"], volume=m["volume"], complete=not cursor, trades=rows)
        finally:
            dump(path, have)
        print(f"  {name:34} trades for {len(have)} markets", flush=True)


def pretrades(client, only, sample=40):
    """Mention families: every public trade of a sample of markets in the 25 hours before the event clock (start clock), to check maker fills."""
    fam = families()
    for name in fam["families"]:
        if not name.startswith("Mentions") or (only and name not in only):
            continue
        path = RAW/"trades-pre"/f"{slug(name)}.json"
        have = load(path, {})
        ms = [m for m in with_clock(name, fam) if m["volume"] >= 200 and m["open"] <= m["t0"]-3600 and m.get("started")]
        ms.sort(key=lambda m: hashlib.sha256(("pre"+m["ticker"]).encode()).hexdigest())
        try:
            for m in ms[:sample]:
                if m["ticker"] in have:
                    continue
                data = read(client, "/markets/trades", dict(ticker=m["ticker"], limit=1000, min_ts=m["t0"]-25*3600, max_ts=m["t0"])) or {}
                rows = [[int(timestamp(t["created_time"])), round(100*float(t["yes_price_dollars"]), 2) if "yes_price_dollars" in t else float(t["yes_price"]),
                         float(t.get("count_fp", t.get("count", 0))), 1 if (t.get("taker_side") or t.get("taker_outcome_side")) == "yes" else 0]
                        for t in data.get("trades") or []]
                have[m["ticker"]] = dict(t0=m["t0"], volume=m["volume"], complete=not data.get("cursor"), trades=rows)
        finally:
            dump(path, have)
        print(f"  {name:34} pre-event trades for {len(have)} markets", flush=True)


FLOW_SAMPLE = 20
FLOW_PAGES = 1
FLOW_BANDS = ((1, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 100))


def flow_report(say):
    fam = families()
    out = {}
    say("\n(2) WHO TAKES WHICH SIDE. Public trade record of a random sample of markets per family (volume >= 200, the 1,000 most recent trades of each)."
        "\n    'YES%' = share of taker CONTRACTS where the taker bought YES (the rest: taker bought NO, which includes selling YES). 'YES$%' weights by dollars"
        "\n    spent by the taker. By YES price of the trade. before = trades before the event clock T0; small = trades of 10 contracts or fewer.")
    say(f"  {'family':34} {'mkts':>4} {'trades':>7} {'YES%':>5} {'YES$%':>5} {'before T0':>9} {'after':>5} {'median mkt':>10} {'med size':>8} {'small% trades':>13} {'small YES%':>10} | "
        + " ".join(f"{lo:>3}-{hi:<3}" for lo, hi in FLOW_BANDS))
    for name in family_names(fam):
        have = load(RAW/"trades"/f"{slug(name)}.json", {})
        clock = {m["ticker"]: m["t0"] for m in with_clock(name, fam)} if have else {}
        rows = np.array([[t[0] < clock.get(k, v["t0"]), t[1], t[2], t[3]] for k, v in have.items() for t in v["trades"]], float).reshape(-1, 4)
        if len(rows) < 200:
            continue
        before, price, count, yes = rows.T
        cost = count*np.where(yes == 1, price, 100-price)

        def share(mask, w=count):
            return float((w*yes)[mask].sum()/w[mask].sum()) if w[mask].sum() else None
        per_market = [np.average([t[3] for t in v["trades"]], weights=[t[2] for t in v["trades"]]) for v in have.values() if v["trades"]]
        small = count <= 10
        o = dict(markets=len(have), trades=len(rows), yes_share=share(count > 0), yes_dollar_share=share(count > 0, cost),
                 before_t0=share(before == 1), after_t0=share(before == 0), share_of_contracts_before_t0=float(count[before == 1].sum()/count.sum()),
                 median_market=float(np.median(per_market)), median_trade_size=float(np.median(count)), small_trade_share=float(small.mean()),
                 small_trades_yes=share(small), by_price={f"{lo}-{hi}": share((price >= lo) & (price < hi)) for lo, hi in FLOW_BANDS},
                 volume_by_price={f"{lo}-{hi}": float(count[(price >= lo) & (price < hi)].sum()/count.sum()) for lo, hi in FLOW_BANDS})
        out[name] = o
        pc = lambda v: "  n/a" if v is None else f"{100*v:5.0f}"
        say(f"  {name:34} {o['markets']:4} {o['trades']:7} {pc(o['yes_share'])} {pc(o['yes_dollar_share'])} {pc(o['before_t0']):>9} {pc(o['after_t0'])} "
            f"{pc(o['median_market']):>10} {o['median_trade_size']:8.0f} {pc(o['small_trade_share']):>13} {pc(o['small_trades_yes']):>10} | "
            + " ".join(f"{pc(o['by_price'][f'{lo}-{hi}']):>7}" for lo, hi in FLOW_BANDS))
    return out


# ---------------------------------------------------------------- bars

def money(group, field):
    value = group.get(field+"_dollars", group.get(field)) if group else None
    return None if value is None else round(100*float(value), 2)


def compact(candles):
    """[end_ts, yes bid close, yes ask close, trade low, trade high, volume] per hourly bar."""
    return [[b["end_period_ts"], money(b.get("yes_bid"), "close"), money(b.get("yes_ask"), "close"), money(b.get("price"), "low"),
             money(b.get("price"), "high"), float(b.get("volume_fp", b.get("volume")) or 0)]
            for b in sorted(candles or [], key=lambda b: b["end_period_ts"])]


BACK = 48                            # hours of bars before T0 (24h reference plus a quote up to 24h old)


def bars(client, only, cap, thin=0):
    fam = families()
    for name in fam["families"]:
        if only and name not in only:
            continue
        path = RAW/"bars"/f"{slug(name)}.json"
        have = load(path, {})
        ms = with_clock(name, fam, clock="close")
        want = {m["ticker"]: m for m in pick_groups(ms, cap)}
        if thin:                                             # depth: one market in `thin` from EVERY event, for more independent events per request
            want.update({m["ticker"]: m for m in ms if m["open"] <= m["t0"]-3600
                         and int(hashlib.sha256(("thin"+m["ticker"]).encode()).hexdigest(), 16) % thin == 0})
        todo = sorted((m for m in want.values() if m["ticker"] not in have), key=lambda m: (m["t0"], m["ticker"]))
        chunks, cur = [], []
        for m in todo:
            trial = cur+[m]
            hours = (trial[-1]["t0"]-trial[0]["t0"])//3600+BACK+3
            if cur and (len(trial) > 100 or len(trial)*hours > 9000):
                chunks.append(cur)
                cur = []
            cur.append(m)
        chunks += [cur] if cur else []
        try:
            for n, group in enumerate(chunks):
                data = read(client, "/markets/candlesticks", dict(market_tickers=",".join(m["ticker"] for m in group),
                                                                  start_ts=group[0]["t0"]-3600*BACK, end_ts=group[-1]["t0"]+3600, period_interval=60))
                if data is None:
                    log["batch_refused"] += 1
                    continue
                got = {x["market_ticker"]: compact(x.get("candlesticks")) for x in data.get("markets", [])}
                for m in group:
                    have[m["ticker"]] = [b for b in got.get(m["ticker"], []) if m["t0"]-3600*BACK <= b[0] <= m["t0"]+3600]
                if n % 15 == 14:
                    dump(path, have)
        finally:
            dump(path, have)
        print(f"  {name:34} bars for {len(have)} markets ({len(chunks)} requests this call)", flush=True)


# ---------------------------------------------------------------- archive holdout (mention markets settled before 2026-09-18)

def archive(client):
    """Out-of-period check for the mention families: listings and hourly bars from the /historical endpoints.
    One request per market, so one market in ARCHIVE_THIN per event, ARCHIVE_CAP markets in all, events in pseudo-random order."""
    fam = families()
    recent = Counter(m["series"] for name in ARCHIVE_FAMILIES for m in family_markets(name, fam))
    chosen = [t for t, n in recent.most_common() if n >= 100 and not t.startswith("KXTRUMPSAY")]
    dump(RAW/"archive-series.json", chosen)
    for series in chosen:
        path = RAW/"markets-archive"/f"{series}.json"
        if load(path, None) is not None:
            continue
        have, cursor, seen = dict(markets=[], pages=0), "", set()
        while have["pages"] < ARCHIVE_PAGES:
            data = read(client, "/historical/markets", dict(series_ticker=series, limit=1000, **({"cursor": cursor} if cursor else {}))) or {}
            have["pages"] += 1
            for m in data.get("markets", []):
                if m.get("result") in ("yes", "no"):
                    v = slim(m)
                    if v["close"] and v["open"] and ARCHIVE_FROM <= v["close"] < START:
                        have["markets"].append(dict(v, exp=v["exp"] or v["close"]))
            cursor = data.get("cursor")
            if not cursor or cursor in seen or not data.get("markets"):
                break
            seen.add(cursor)
        dump(path, have)
        print(f"  {series:28} {len(have['markets']):5} archived markets closing 2026-06-01 to 2026-09-17 ({have['pages']} pages)", flush=True)
    path = RAW/"bars"/f"{slug(ARCHIVE)}.json"
    have = load(path, {})
    ms = [m for m in with_clock(ARCHIVE, fam, clock="close") if m["open"] <= m["t0"]-3600
          and int(hashlib.sha256(("thin"+m["ticker"]).encode()).hexdigest(), 16) % ARCHIVE_THIN == 0]
    ms.sort(key=lambda m: (hashlib.sha256(m["group"].encode()).hexdigest(), m["ticker"]))
    try:
        for m in ms[:ARCHIVE_CAP]:
            if m["ticker"] in have:
                continue
            data = read(client, f"/historical/markets/{m['ticker']}/candlesticks",
                        dict(start_ts=m["t0"]-3600*BACK, end_ts=m["t0"]+3600, period_interval=60))
            have[m["ticker"]] = compact((data or {}).get("candlesticks"))
    finally:
        dump(path, have)
    print(f"  {ARCHIVE}: bars for {len(have)} of {min(len(ms), ARCHIVE_CAP)} sampled markets ({len(ms)} eligible)", flush=True)


# ---------------------------------------------------------------- rows

MAX_SPREAD = 10                      # "quoted" sample; "everything" takes any two-sided quote
COLS = ("h", "bid", "ask", "y", "mult", "maker_fee", "filled", "vol_near", "volume", "t0", "g", "d", "m", "first", "whole", "yfilled")


def fee_cents(price, mult, n):
    """Taker fee per contract, in cents, for an order of n contracts at `price` cents (rounded up to the cent on the order)."""
    return np.ceil(np.round(7*mult*n*price/100*(1-price/100), 6))/n


def build(name, fam, table):
    """Column arrays of one family: one row per market x reference time with a two-sided quote."""
    have = load(RAW/"bars"/f"{slug(name)}.json", {})
    every = with_clock(name, fam, table)
    missing = {m["group"] for m in with_clock(name, fam, table, clock="close") if m["open"] <= m["t0"]-3600 and m["ticker"] not in have}
    ms = [m for m in every if m["ticker"] in have]
    if not ms:
        return None
    start = {}
    for m in ms:
        start[m["group"]] = min(start.get(m["group"], m["t0"]), m["t0"])
    groups = sorted(start, key=lambda g: (start[g], g))
    gcode = {g: i for i, g in enumerate(groups)}
    dcode = {d: i for i, d in enumerate(sorted({m["day"] for m in ms}))}
    rows, cover = [], Counter()
    for mi, m in enumerate(ms):
        bs = have[m["ticker"]]
        cover["markets with bars downloaded"] += 1
        for h in REFS:
            tq = (m["t0"]-3600*h)//3600*3600
            if m["open"] > tq:
                cover[f"{h}h not open yet"] += 1
                continue
            if m["close"] <= tq:
                cover[f"{h}h already closed"] += 1
                continue
            past = [b for b in bs if tq-86400 <= b[0] <= tq]
            if not past:
                cover[f"{h}h no bar in 24h"] += 1
                continue
            bid, ask = past[-1][1], past[-1][2]
            if bid is None or ask is None or not 1 <= bid < ask <= 99:
                cover[f"{h}h one-sided or empty"] += 1
                continue
            later = [b for b in bs if tq < b[0] <= m["t0"]]
            hi = max((b[4] for b in later if b[4] is not None and b[5] > 0), default=-1)
            lo = min((b[3] for b in later if b[3] is not None and b[5] > 0), default=999)
            near = sum(b[5] for b in bs if tq-3600 < b[0] <= tq+3600)/2
            cover[f"{h}h rows"] += 1
            rows.append((h, bid, ask, m["result"] == "yes", m["mult"], m["maker_fee"], hi > ask-1, near, m["volume"], m["t0"],
                         gcode[m["group"]], dcode[m["day"]], mi, gcode[m["group"]] < len(groups)/2, m["group"] not in missing, lo < bid+1))
    if not rows:
        return None
    a = np.array(rows, float)
    r = {c: a[:, i] for i, c in enumerate(COLS)}
    for c in ("g", "d", "m"):
        r[c] = r[c].astype(int)
    r["y"], r["first"], r["filled"], r["whole"] = r["y"] == 1, r["first"] == 1, r["filled"] == 1, r["whole"] == 1
    r["yfilled"] = r["yfilled"] == 1
    r["mid"], r["spread"] = (r["bid"]+r["ask"])/2, r["ask"]-r["bid"]
    r.update(name=name, cover=dict(cover), groups=groups, days=sorted(dcode), tickers=[m["ticker"] for m in ms], names=[m["name"] for m in ms])
    for n in SIZES:
        r[f"fee{n}"] = fee_cents(100-r["bid"], r["mult"], n)
        r[f"yfee{n}"] = fee_cents(r["ask"], r["mult"], n)
    r["cost"] = 100-r["bid"]+r["fee10"]                      # NO at 100 - YES bid, 10-contract order
    r["net"] = 100*(~r["y"])-r["cost"]
    r["ycost"] = r["ask"]+r["yfee10"]                        # control: YES at the ask
    r["ynet"] = 100*r["y"]-r["ycost"]
    r["gap"] = 100*r["y"]-r["mid"]                           # positive: YES underpriced at the midpoint
    r["mcost"] = 100-(r["ask"]-1)+np.where(r["maker_fee"] == 1, 1.75*(r["ask"]-1)/100*(1-(r["ask"]-1)/100), 0)
    r["mnet"] = 100*(~r["y"])-r["mcost"]
    r["ymcost"] = r["bid"]+1+np.where(r["maker_fee"] == 1, 1.75*(r["bid"]+1)/100*(1-(r["bid"]+1)/100), 0)   # mirror: a YES bid one cent above the best
    r["ymnet"] = 100*r["y"]-r["ymcost"]
    return r


POOLS = (("ALL SPORTS PROPS (pooled)", lambda n: n in SPORTS),
         ("ALL MENTIONS since Sep 18 (pooled)", lambda n: n.startswith("Mentions") and n != ARCHIVE),
         ("ALL CHARTS SCORES TV POLITICS (pooled)", lambda n: n not in SPORTS and not n.startswith("Mentions")))


def pool(name, rs):
    """Several families as one, for the calibration and trading tables only. Events stay distinct, calendar days are shared,
    and the halves are the first and second half of calendar days."""
    if len(rs) < 2:
        return None
    days = sorted({d for r in rs for d in r["days"]})
    code = {d: i for i, d in enumerate(days)}
    out = dict(name=name, days=days, groups=[g for r in rs for g in r["groups"]], tickers=[t for r in rs for t in r["tickers"]], cover={})
    offset_g = np.cumsum([0]+[len(r["groups"]) for r in rs])
    offset_m = np.cumsum([0]+[len(r["tickers"]) for r in rs])
    for key in rs[0]:
        if isinstance(rs[0][key], np.ndarray) and key not in ("g", "d", "m", "z"):
            out[key] = np.concatenate([r[key] for r in rs])
    out["g"] = np.concatenate([r["g"]+offset_g[i] for i, r in enumerate(rs)])
    out["m"] = np.concatenate([r["m"]+offset_m[i] for i, r in enumerate(rs)])
    out["d"] = np.concatenate([np.array([code[r["days"][k]] for k in r["d"]]) for r in rs])
    out["first"] = out["d"] < len(days)/2
    return out


def ratio(num, den, codes, mask, draws=1000):
    """sum(num)/sum(den) over the masked rows with a 95% bootstrap interval resampling whole clusters."""
    if not mask.any():
        return None
    c = codes[mask]
    _, inv = np.unique(c, return_inverse=True)
    k = inv.max()+1
    a, b = np.bincount(inv, num[mask], k), np.bincount(inv, den[mask], k)
    est = a.sum()/b.sum()
    if k < 4:
        return est, None, None, k
    pick = RNG.integers(0, k, (draws, k))
    bb = b[pick].sum(1)
    lo, hi = np.percentile(a[pick].sum(1)[bb > 0]/bb[bb > 0], [2.5, 97.5])
    return est, float(lo), float(hi), k


def wide(r, num, den, mask):
    """Estimate with the wider of the event-clustered and day-clustered intervals."""
    a, b = ratio(num, den, r["g"], mask), ratio(num, den, r["d"], mask)
    if a is None:
        return None
    if a[1] is None or b[1] is None:
        return a[0], None, None, a[3], b[3]
    pick = a if a[2]-a[1] >= b[2]-b[1] else b
    return a[0], pick[1], pick[2], a[3], b[3]


def linear(num, den, codes, mask):
    """Estimate and cluster-robust standard error of sum(num)/sum(den) (linearised)."""
    if not mask.any():
        return None
    _, inv = np.unique(codes[mask], return_inverse=True)
    k = inv.max()+1
    a, b = np.bincount(inv, num[mask], k), np.bincount(inv, den[mask], k)
    est = a.sum()/b.sum()
    if k < 3:
        return est, float("inf"), k
    return est, math.sqrt(k/(k-1)*((a-est*b)**2).sum())/b.sum(), k


def show(v, scale=1.0, unit=""):
    if v is None:
        return f"{'n/a':>22}"
    if v[1] is None:
        return f"{scale*v[0]:+6.1f}{unit} [   n/a      ]"
    return f"{scale*v[0]:+6.1f}{unit} [{scale*v[1]:+6.1f},{scale*v[2]:+6.1f}]"


def band_masks(r, h, quoted=True):
    base = (r["h"] == h) & ((r["spread"] <= MAX_SPREAD) if quoted else True)
    out = [(f"{lo}-{hi}", base & (r["mid"] >= lo) & (r["mid"] < hi)) for lo, hi in BANDS]
    return base, out


# ---------------------------------------------------------------- report

def calibration(fams, say, summary):
    say(f"\n(3) CALIBRATION. YES midpoint at the reference time against how often YES happened. Quotes two-sided with a spread of at most {MAX_SPREAD}c."
        "\n    gap = YES frequency minus midpoint, in cents: NEGATIVE means YES was overpriced (the idea). Interval: the wider of event-clustered and day-clustered."
        "\n    h1 / h2 = gap in the first and second chronological half of event groups.")
    for r in fams:
        for h in REFS:
            base, bands = band_masks(r, h)
            if base.sum() < 50:
                continue
            say(f"  {r['name']}, {h}h before  ({int(base.sum())} markets, {len(set(r['g'][base]))} events, {len(set(r['d'][base]))} days)")
            say(f"    {'band':>6} {'n':>6} {'ev':>5} {'days':>4} {'mid':>5} {'YES%':>5} {'gap c':>22} {'h1':>6} {'h2':>6}")
            one = np.ones(len(base))
            for label, mask in bands+[("all", base)]:
                if mask.sum() < 20:
                    continue
                v = wide(r, r["gap"], one, mask)
                h1, h2 = (ratio(r["gap"], one, r["g"], mask & (r["first"] == f), 1) for f in (True, False))
                say(f"    {label:>6} {int(mask.sum()):6} {v[3]:5} {v[4]:4} {r['mid'][mask].mean():5.1f} {100*r['y'][mask].mean():5.1f} {show(v)} "
                    f"{h1[0] if h1 else float('nan'):+6.1f} {h2[0] if h2 else float('nan'):+6.1f}")
                summary.setdefault("calibration", {})[f"{r['name']}|{h}|{label}"] = dict(
                    n=int(mask.sum()), events=v[3], days=v[4], mid=float(r["mid"][mask].mean()), yes=float(r["y"][mask].mean()), gap=v[0], ci=[v[1], v[2]],
                    h1=h1[0] if h1 else None, h2=h2[0] if h2 else None)


def tradeable(fams, say, summary):
    say("\n(4) THE TRADEABLE RULE. Buy NO at 100 - YES bid plus the taker fee (10-contract order), hold to settlement."
        "\n    ret% = total net / total paid (return on the money put down). net c = cents per contract. Interval: wider of event- and day-clustered."
        "\n    'YES ask%' = the mirror trade, buying YES at the ask, for comparison. 'every' = any two-sided quote, whatever the spread (the purest form).")
    for r in fams:
        for h in REFS:
            base, bands = band_masks(r, h)
            every = r["h"] == h
            if every.sum() < 50:
                continue
            say(f"  {r['name']}, {h}h before")
            say(f"    {'band':>6} {'n':>6} {'NO paid':>7} {'NO won%':>7} {'ret %':>23} {'net c':>22} {'h1 %':>6} {'h2 %':>6} {'YES ask%':>8} {'@3 c':>6} {'@100 c':>6}")
            for label, mask in bands+[("quoted", base), ("every", every)]:
                if mask.sum() < 20:
                    continue
                ret, net = wide(r, r["net"], r["cost"], mask), wide(r, r["net"], np.ones(len(base)), mask)
                h1, h2 = (ratio(r["net"], r["cost"], r["g"], mask & (r["first"] == f), 1) for f in (True, False))
                yes = r["ynet"][mask].sum()/r["ycost"][mask].sum()
                alt = [float((r["net"]+r["fee10"]-r[f"fee{n}"])[mask].mean()) for n in (3, 100)]
                say(f"    {label:>6} {int(mask.sum()):6} {r['cost'][mask].mean():7.1f} {100*(~r['y'][mask]).mean():7.1f} {show(ret, 100, '%')} {show(net)} "
                    f"{100*h1[0] if h1 else float('nan'):+6.1f} {100*h2[0] if h2 else float('nan'):+6.1f} {100*yes:+8.1f} {alt[0]:+6.2f} {alt[1]:+6.2f}")
                summary.setdefault("taker", {})[f"{r['name']}|{h}|{label}"] = dict(
                    n=int(mask.sum()), events=ret[3], days=ret[4], paid=float(r["cost"][mask].mean()), no_won=float((~r["y"][mask]).mean()),
                    ret=ret[0], ret_ci=[ret[1], ret[2]], net=net[0], net_ci=[net[1], net[2]], h1=h1[0] if h1 else None, h2=h2[0] if h2 else None,
                    yes_at_ask_ret=float(yes), net_at_3=alt[0], net_at_100=alt[1])
    say("\n    MAKER VERSION. A NO bid one cent above the best NO bid (spread >= 2c), filled only if a later hourly bar before T0 printed a YES trade strictly"
        "\n    above our offer. filled / unfilled net = settlement value minus our price for the orders that did / did not fill: the difference is adverse selection.")
    say("    mirror = the same order on the other side (a YES bid one cent above the best YES bid, filled by a later print strictly below it): if both sides earn the"
        "\n    same, the result is spread capture in a wide book and has nothing to do with YES being overpriced.")
    say(f"    {'family':34} {'ref':>4} {'orders':>6} {'fill%':>5} {'filled ret %':>23} {'filled net c':>22} {'unfilled c':>10} {'selection c':>11} {'h1 c':>6} {'h2 c':>6} "
        f"{'spread':>6} | {'mirror fill%':>12} {'mirror net c':>22}")
    tests = {"NO side": [], "mirror": []}
    for r in fams:
        for h in REFS:
            base = (r["h"] == h) & (r["spread"] >= 2)
            fill = base & r["filled"]
            if base.sum() < 50 or fill.sum() < 20:
                continue
            one = np.ones(len(base))
            ret, net = wide(r, r["mnet"], r["mcost"], fill), wide(r, r["mnet"], one, fill)
            un = r["mnet"][base & ~r["filled"]].mean() if (base & ~r["filled"]).any() else float("nan")
            h1, h2 = (ratio(r["mnet"], one, r["g"], fill & (r["first"] == f), 1) for f in (True, False))
            yfill = base & r["yfilled"]
            mirror = wide(r, r["ymnet"], one, yfill) if yfill.sum() >= 20 else None
            for side, f, value in (("NO side", fill, r["mnet"]), ("mirror", yfill, r["ymnet"])):
                a, b = linear(value, one, r["d"], f & r["first"]), linear(value, one, r["d"], f & ~r["first"])
                if a is not None and b is not None and a[2] >= MIN_DAYS and f.sum() >= 50 and "(pooled)" not in r["name"]:
                    tests[side].append((f"{r['name']} {h}h", bool(a[0]-1.96*a[1] > 0 and b[0] > 0), bool(a[0]+1.96*a[1] < 0 and b[0] < 0)))
            say(f"    {r['name']:34} {h:3}h {int(base.sum()):6} {100*fill.sum()/base.sum():5.1f} {show(ret, 100, '%')} {show(net)} {un:+10.2f} {net[0]-un:+11.2f} "
                f"{h1[0] if h1 else float('nan'):+6.1f} {h2[0] if h2 else float('nan'):+6.1f} {r['spread'][base].mean():6.1f} | {100*yfill.sum()/base.sum():12.1f} {show(mirror)}")
            summary.setdefault("maker", {})[f"{r['name']}|{h}"] = dict(orders=int(base.sum()), filled=int(fill.sum()), ret=ret[0], ret_ci=[ret[1], ret[2]], net=net[0],
                                                                      net_ci=[net[1], net[2]], unfilled_net=float(un), h1=h1[0] if h1 else None, h2=h2[0] if h2 else None,
                                                                      mean_spread=float(r["spread"][base].mean()), mirror_filled=int(yfill.sum()),
                                                                      mirror_net=mirror[0] if mirror else None, mirror_ci=[mirror[1], mirror[2]] if mirror else None)
    maker_tests(tests, say, summary)


def maker_tests(tests, say, summary):
    say("\n    Maker cells (family x reference time, at least 50 fills and 6 first-half days) whose first-half day-clustered interval is above zero with a positive"
        "\n    second half ('wins'), or below zero with a negative second half ('loses'). The mirror side is the benchmark: a wide book alone would make both sides win.")
    for side, rows in tests.items():
        wins, loses = [k for k, w, _ in rows if w], [k for k, _, l in rows if l]
        say(f"    {side:8} {len(rows)} cells: {len(wins)} win, {len(loses)} lose.  wins: {', '.join(wins) or 'none'}.  loses: {', '.join(loses) or 'none'}")
        summary.setdefault("maker_tests", {})[side] = dict(cells=len(rows), wins=wins, loses=loses)


def tape_check(fams, say, summary):
    """Maker fills against the public trade record, for the markets whose trades were sampled."""
    say("\n    MAKER FILLS AGAINST THE TRADE RECORD (mention families: every trade in the 25 hours before the event clock, for a sample of markets;"
        "\n    other families: the general trade sample where it reaches back to the reference time)."
        "\n    'through' = contracts where a taker BOUGHT YES at a price strictly above our offer between the reference time and T0; 'at' = at our offer or above."
        "\n    'other way' = contracts where a taker bought NO at or below the YES bid in the same window.")
    say(f"    {'family':34} {'ref':>4} {'orders':>6} {'bar rule fill%':>14} {'tape through%':>13} {'tape at%':>8} {'median through':>14} {'10+ through%':>12} {'100+':>5} "
        f"{'YES-buy contracts':>17} {'other way':>9}")
    for r in fams:
        have = load(RAW/"trades-pre"/f"{slug(r['name'])}.json", {}) if CLOCK[0] == "start" else {}
        have = have or load(RAW/"trades"/f"{slug(r['name'])}.json", {})
        if not have:
            continue
        for h in REFS:
            rows = []
            for i in np.flatnonzero((r["h"] == h) & (r["spread"] >= 2)):
                v = have.get(r["tickers"][r["m"][i]])
                if not v or not v["trades"]:
                    continue
                t0 = int(r["t0"][i])
                tq = (t0-3600*h)//3600*3600
                if not v["complete"] and min(t[0] for t in v["trades"]) > tq:
                    continue
                window = [t for t in v["trades"] if tq < t[0] <= t0]
                offer = r["ask"][i]-1
                rows.append((r["filled"][i], sum(t[2] for t in window if t[3] == 1 and t[1] > offer), sum(t[2] for t in window if t[3] == 1 and t[1] >= offer),
                             sum(t[2] for t in window if t[3] == 1), sum(t[2] for t in window if t[3] == 0 and t[1] <= r["bid"][i])))
            if len(rows) < 15:
                continue
            a = np.array(rows, float)
            thr = a[:, 1]
            o = dict(orders=len(a), bar_rule_fill=float(a[:, 0].mean()), tape_through=float((thr >= 1).mean()), tape_at=float((a[:, 2] >= 1).mean()),
                     median_through=float(np.median(thr[thr >= 1])) if (thr >= 1).any() else 0.0, through_10=float((thr >= 10).mean()),
                     through_100=float((thr >= 100).mean()), yes_buy_contracts=float(a[:, 3].sum()), no_buy_at_bid_contracts=float(a[:, 4].sum()))
            summary.setdefault("tape", {})[f"{r['name']}|{h}"] = o
            say(f"    {r['name']:34} {h:3}h {o['orders']:6} {100*o['bar_rule_fill']:13.0f}% {100*o['tape_through']:12.0f}% {100*o['tape_at']:7.0f}% {o['median_through']:14.0f} "
                f"{100*o['through_10']:11.0f}% {100*o['through_100']:4.0f}% {o['yes_buy_contracts']:17.0f} {o['no_buy_at_bid_contracts']:9.0f}")


def best_ref(r):
    """The reference time closest to the event that has a usable sample."""
    for h in sorted(REFS):
        if (r["h"] == h).sum() >= 50:
            return h
    return None


def tail(fams, say, summary):
    say("\n(5) TAIL AND DEPENDENCE. A book that buys 10 NO contracts on EVERY market of the family with a two-sided quote at the reference time (events downloaded whole)."
        "\n    P&L in dollars; 'stake' = money put down. over-dispersion = variance of an event's summed outcome surprises / what independent markets would give"
        "\n    (1 = independent; 5 = an event with 30 markets carries the information of 6).")
    say(f"    {'family':34} {'ref':>4} {'events':>6} {'mkts/ev':>7} {'ev P&L mean':>11} {'worst event':>24} {'NOs lost there':>14} {'worst day':>22} {'losing days':>11} {'over-disp':>9}")
    for r in fams:
        h = best_ref(r)
        if h is None:
            continue
        mask = (r["h"] == h) & r["whole"]
        if mask.sum() < 30:
            continue
        g, d = r["g"][mask], r["d"][mask]
        pnl, stake, lost = r["net"][mask]/10, r["cost"][mask]/10, r["y"][mask]
        ev, ev_stake, ev_lost, ev_n = (np.bincount(g, w) for w in (pnl, stake, lost.astype(float), np.ones(mask.sum())))
        day, day_stake = np.bincount(d, pnl), np.bincount(d, stake)
        used, used_d = ev_n > 0, day_stake > 0
        wi, wd = int(np.argmin(np.where(used, ev, np.inf))), int(np.argmin(np.where(used_d, day, np.inf)))
        p = r["mid"][mask]/100
        surprise = np.bincount(g, lost-p)[used]
        disp = float((surprise**2).mean()/np.bincount(g, p*(1-p))[used].mean())
        o = dict(ref=h, events=int(used.sum()), markets_per_event=float(ev_n[used].mean()), mean_event_pnl=float(ev[used].mean()),
                 worst_event=dict(name=r["groups"][wi], pnl=float(ev[wi]), stake=float(ev_stake[wi]), nos_lost=int(ev_lost[wi]), markets=int(ev_n[wi])),
                 worst_day=dict(day=r["days"][wd], pnl=float(day[wd]), stake=float(day_stake[wd])),
                 losing_day_share=float((day[used_d] < 0).mean()), days=int(used_d.sum()), over_dispersion=disp,
                 event_pnl_percentiles=[float(v) for v in np.percentile(ev[used], [1, 5, 25, 50, 75, 95])])
        summary.setdefault("tail", {})[r["name"]] = o
        say(f"    {r['name']:34} {h:3}h {o['events']:6} {o['markets_per_event']:7.1f} {o['mean_event_pnl']:+11.2f} "
            f"{ev[wi]:+8.0f} on {ev_stake[wi]:5.0f} ({100*ev[wi]/ev_stake[wi]:+4.0f}%) {int(ev_lost[wi]):6} of {int(ev_n[wi]):4} "
            f"{day[wd]:+8.0f} on {day_stake[wd]:6.0f} ({100*day[wd]/day_stake[wd]:+4.0f}%) {100*o['losing_day_share']:10.0f}% {disp:9.1f}")


def capacity(fams, say, summary):
    say("\n(6) CAPACITY at the reference time closest to the event (quoted sample for spread; every two-sided market for dollars)."
        "\n    'traded/h' = contracts per hour per market in the two hours around the reference time. $ per day = money put down buying NO on everything.")
    say(f"    {'family':34} {'ref':>4} {'mkts/day':>8} {'med spread':>10} {'spread<=2':>9} {'med traded/h':>12} {'zero-trade%':>11} {'med life vol':>12} {'$/day @10':>10} {'$/day @100':>10} {'@100 > traded/h':>15}")
    for r in fams:
        h = best_ref(r)
        if h is None:
            continue
        mask = (r["h"] == h) & r["whole"]
        if mask.sum() < 30:
            continue
        days = len(set(r["d"][mask]))
        o = dict(ref=h, markets_per_day=float(mask.sum()/days), median_spread=float(np.median(r["spread"][mask])), spread_le2=float((r["spread"][mask] <= 2).mean()),
                 median_traded_per_hour=float(np.median(r["vol_near"][mask])), no_trade_share=float((r["vol_near"][mask] == 0).mean()),
                 median_lifetime_volume=float(np.median(r["volume"][mask])), dollars_per_day_10=float(r["cost"][mask].sum()/10/days),
                 dollars_per_day_100=float(r["cost"][mask].sum()/days), share_100_exceeds_hourly_volume=float((r["vol_near"][mask] < 100).mean()))
        summary.setdefault("capacity", {})[r["name"]] = o
        say(f"    {r['name']:34} {h:3}h {o['markets_per_day']:8.0f} {o['median_spread']:10.0f} {100*o['spread_le2']:8.0f}% {o['median_traded_per_hour']:12.0f} "
            f"{100*o['no_trade_share']:10.0f}% {o['median_lifetime_volume']:12.0f} {o['dollars_per_day_10']:10.0f} {o['dollars_per_day_100']:10.0f} "
            f"{100*o['share_100_exceeds_hourly_volume']:14.0f}%")


MIN_ROWS, MIN_DAYS = 30, 6


def cell_table(r, net, gap):
    """Pass / fail of every band x reference cell of one family for the taker return and for the midpoint gap."""
    out = {}
    one = np.ones(len(net))
    for h in REFS:
        _, bands = band_masks(r, h)
        est = []
        for label, mask in bands:
            if mask.sum() < MIN_ROWS:
                est.append(None)
                continue
            a = linear(net, r["cost"], r["d"], mask & r["first"])
            b = linear(net, r["cost"], r["d"], mask & ~r["first"])
            c = linear(-gap, one, r["d"], mask & r["first"])
            d = linear(-gap, one, r["d"], mask & ~r["first"])
            if a is None or b is None or a[2] < MIN_DAYS:
                est.append(None)
                continue
            est.append(dict(label=label, n=int(mask.sum()), whole=net[mask].sum()/r["cost"][mask].sum(), whole_gap=-gap[mask].mean(),
                            first_ok=a[0]-1.96*a[1] > 0, second_ok=b[0] > 0, gap_first_ok=c[0]-1.96*c[1] > 0, gap_second_ok=d[0] > 0,
                            h1=a[0], h1_se=a[1], h2=b[0]))
        for i, e in enumerate(est):
            if e is None:
                continue
            near = [est[j] for j in (i-1, i+1) if 0 <= j < len(est) and est[j] is not None]
            e["passes"] = bool(e["first_ok"] and e["second_ok"] and near and all(n["whole"] > 0 for n in near))
            e["gap_passes"] = bool(e["gap_first_ok"] and e["gap_second_ok"] and near and all(n["whole_gap"] > 0 for n in near))
            out[f"{r['name']}|{h}|{e['label']}"] = e
    return out


def dependence(r, sims=40):
    """Latent within-event correlation of a Gaussian copula that reproduces the observed over-dispersion of event sums."""
    h = best_ref(r)
    mask = r["h"] == h
    p, g = np.clip(r["mid"][mask]/100, .005, .995), r["g"][mask]
    _, g = np.unique(g, return_inverse=True)
    k = g.max()+1
    denom = np.bincount(g, p*(1-p)).mean()
    seen = (np.bincount(g, r["y"][mask]-p)**2).mean()/denom
    z = np.array([statistics_ppf(v) for v in p])
    best = (float("inf"), 0.0)
    for rho in (0, .005, .01, .02, .03, .05, .075, .1, .15, .2, .3, .45, .6, .8):
        d = []
        for _ in range(sims):
            y = math.sqrt(rho)*RNG.standard_normal(k)[g]+math.sqrt(1-rho)*RNG.standard_normal(len(p)) < z
            d.append((np.bincount(g, y-p)**2).mean()/denom)
        best = min(best, (abs(np.mean(d)-seen), rho))
    return best[1], float(seen)


def statistics_ppf(p):
    from statistics import NormalDist
    return NormalDist().inv_cdf(float(p))


def multiplicity(fams, say, summary, sims=200):
    say("\n(7) MULTIPLICITY. A cell = family x price band x reference time with at least 30 quoted markets and 6 first-half days."
        "\n    Pass = first-half day-clustered 95% interval of the taker return above zero AND second half positive AND the neighbouring bands positive overall."
        "\n    'midpoint' repeats the test for buying NO at the midpoint with no fee (is YES overpriced at all, before costs?)."
        "\n    Chance = the same table with outcomes re-drawn under fair pricing (probability = the last midpoint before the event; one shared shock per event,"
        "\n    sized per family to reproduce the observed over-dispersion of event sums).")
    real, rhos = {}, {}
    for r in fams:
        real.update(cell_table(r, r["net"], r["gap"]))
        rhos[r["name"]] = dependence(r)
    passed = [k for k, e in real.items() if e["passes"]]
    gap_passed = [k for k, e in real.items() if e["gap_passes"]]
    lower = sum(e["first_ok"] for e in real.values())
    counts, gap_counts, lowers = [], [], []
    for _ in range(sims):
        n = ng = nl = 0
        for r in fams:
            rho = rhos[r["name"]][0]
            # one outcome per market, shared by its rows at every reference time; probability = midpoint closest to the event
            order = np.lexsort((r["h"], r["m"]))
            first = np.r_[True, r["m"][order][1:] != r["m"][order][:-1]]
            p_market = np.full(r["m"].max()+1, .5)
            p_market[r["m"][order][first]] = np.clip(r["mid"][order][first]/100, .005, .995)
            g_market = np.zeros(r["m"].max()+1, int)
            g_market[r["m"]] = r["g"]
            z = np.array([statistics_ppf(v) for v in p_market]) if "z" not in r else r["z"]
            r["z"] = z
            y_market = math.sqrt(rho)*RNG.standard_normal(g_market.max()+1)[g_market]+math.sqrt(1-rho)*RNG.standard_normal(len(z)) < z
            y = y_market[r["m"]]
            t = cell_table(r, 100*(~y)-r["cost"], 100*y-r["mid"])
            n += sum(e["passes"] for e in t.values())
            ng += sum(e["gap_passes"] for e in t.values())
            nl += sum(e["first_ok"] for e in t.values())
        counts.append(n); gap_counts.append(ng); lowers.append(nl)
    counts, gap_counts, lowers = np.array(counts), np.array(gap_counts), np.array(lowers)
    say(f"    cells examined: {len(real)}")
    say(f"    taker rule:  {len(passed)} cells pass; under fair pricing {counts.mean():.1f} expected (95% of simulations at most {np.percentile(counts, 95):.0f}; "
        f"share of simulations with at least as many: {100*(counts >= len(passed)).mean():.0f}%)")
    say(f"    first-half interval above zero alone: {lower} cells; expected by chance {lowers.mean():.1f}")
    say(f"    midpoint:    {len(gap_passed)} cells pass; under fair pricing {gap_counts.mean():.1f} expected (95% at most {np.percentile(gap_counts, 95):.0f}; "
        f"share with at least as many: {100*(gap_counts >= len(gap_passed)).mean():.0f}%)")
    say("    within-event correlation used per family (and observed over-dispersion): "+"; ".join(f"{k} {v[0]:.2f} ({v[1]:.1f})" for k, v in rhos.items()))
    for k in passed:
        e = real[k]
        say(f"    PASS taker    {k:50} n {e['n']:5}  first half {100*e['h1']:+6.1f}% (se {100*e['h1_se']:.1f})  second half {100*e['h2']:+6.1f}%  whole {100*e['whole']:+6.1f}%")
    for k in gap_passed:
        e = real[k]
        say(f"    PASS midpoint {k:50} n {e['n']:5}  whole gap {e['whole_gap']:+5.1f}c")
    summary["multiplicity"] = dict(cells=len(real), taker_pass=passed, taker_expected=float(counts.mean()), taker_95=float(np.percentile(counts, 95)),
                                   taker_p=float((counts >= len(passed)).mean()), gap_pass=gap_passed, gap_expected=float(gap_counts.mean()),
                                   gap_95=float(np.percentile(gap_counts, 95)), gap_p=float((gap_counts >= len(gap_passed)).mean()),
                                   first_half_above_zero=int(lower), first_half_expected=float(lowers.mean()), rho={k: v for k, v in rhos.items()},
                                   cell_detail={k: {a: (float(b) if isinstance(b, (float, np.floating)) else b) for a, b in e.items()} for k, e in real.items()})
    return real


def shrinkage(fams, say, summary):
    """Family-level taker result (all quoted bands, reference closest to the event), partially pooled as in research/calibration_map.py."""
    from research.calibration_map import shrink
    say("\n(8) SHRINKAGE. Each family's taker net (cents per contract, all quoted bands, reference closest to the event, day-clustered standard error),"
        "\n    pulled towards the mean of all families in proportion to its noise (normal-normal, half-normal(3c) prior on the spread between families)."
        "\n    'per day' = shrunk cents x quoted markets per day at 10 contracts, in dollars.")
    cells, extra = {}, {}
    one = None
    for r in fams:
        h = best_ref(r)
        if h is None:
            continue
        mask = (r["h"] == h) & (r["spread"] <= MAX_SPREAD)
        one = np.ones(len(mask))
        v = linear(r["net"], one, r["d"], mask)
        if v is None or not math.isfinite(v[1]) or v[1] == 0:
            continue
        cells[r["name"]] = (v[0], v[1])
        extra[r["name"]] = (h, mask.sum()/len(set(r["d"][mask])), r["cost"][mask].mean())
    if len(cells) < 3:
        return
    s = shrink(cells)
    say(f"    all families together: {s['overall_mean']:+.2f}c (se {s['overall_se']:.2f}); spread between families {s['between_family_sd']:.2f}c")
    say(f"    {'family':34} {'ref':>4} {'raw c':>7} {'se':>5} {'shrunk c':>8} {'sd':>5} {'P(>0)':>6} {'shrunk ret %':>12} {'$ per day @10':>13}")
    for name, v in sorted(s["families"].items(), key=lambda kv: -kv[1]["shrunk"]):
        h, per_day, paid = extra[name]
        say(f"    {name:34} {h:3}h {cells[name][0]:+7.2f} {cells[name][1]:5.2f} {v['shrunk']:+8.2f} {v['sd']:5.2f} {v['prob_positive']:6.2f} {100*v['shrunk']/paid:+12.1f} "
            f"{v['shrunk']*per_day*10/100:+13.2f}")
    summary["shrinkage"] = dict(s, raw={k: dict(net=float(v[0]), se=float(v[1]), ref=extra[k][0], markets_per_day=float(extra[k][1]), paid=float(extra[k][2]))
                                        for k, v in cells.items()})
    say("\n    The same for the MAKER version (filled orders, every family x reference time with at least 50 fills; the fill model is not shrunk, only the noise)."
        "\n    'minus mirror' = NO-side net minus the mirror YES-side net: the part that is YES being overpriced and not spread capture.")
    cells, extra = {}, {}
    for r in fams:
        for h in REFS:
            base = (r["h"] == h) & (r["spread"] >= 2)
            fill = base & r["filled"]
            if fill.sum() < 50:
                continue
            one = np.ones(len(base))
            v = linear(r["mnet"], one, r["d"], fill)
            if not math.isfinite(v[1]) or v[1] == 0:
                continue
            mirror = r["ymnet"][base & r["yfilled"]].mean() if (base & r["yfilled"]).sum() >= 20 else float("nan")
            cells[f"{r['name']} {h}h"] = (v[0], v[1])
            extra[f"{r['name']} {h}h"] = (fill.sum()/len(set(r["d"][base])), r["mcost"][fill].mean(), mirror, fill.sum()/base.sum())
    if len(cells) >= 3:
        m = shrink(cells)
        say(f"    all cells together: {m['overall_mean']:+.2f}c (se {m['overall_se']:.2f}); spread between cells {m['between_family_sd']:.2f}c")
        say(f"    {'family and reference':42} {'fills/day':>9} {'fill%':>5} {'raw c':>7} {'se':>5} {'shrunk c':>8} {'sd':>5} {'P(>0)':>6} {'shrunk ret %':>12} {'minus mirror c':>14} {'$ per day @10':>13}")
        for name, v in sorted(m["families"].items(), key=lambda kv: -kv[1]["shrunk"]):
            per_day, paid, mirror, rate = extra[name]
            say(f"    {name:42} {per_day:9.1f} {100*rate:5.0f} {cells[name][0]:+7.2f} {cells[name][1]:5.2f} {v['shrunk']:+8.2f} {v['sd']:5.2f} {v['prob_positive']:6.2f} "
                f"{100*v['shrunk']/paid:+12.1f} {cells[name][0]-mirror:+14.2f} {v['shrunk']*per_day*10/100:+13.2f}")
        summary["maker_shrinkage"] = dict(m, raw={k: dict(net=float(v[0]), se=float(v[1]), fills_per_day=float(extra[k][0]), paid=float(extra[k][1]),
                                                          mirror_net=float(extra[k][2]), fill_rate=float(extra[k][3])) for k, v in cells.items()})


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)
    fam = families()
    say("(1) FAMILIES (settled yes/no markets closing 2026-09-18 to 2026-11-18)")
    table = leads()
    summary = dict(families=table)
    summary["flow"] = flow_report(say)
    fams = [r for r in (build(name, fam, table) for name in family_names(fam)) if r is not None]
    say("\nCOVERAGE of the hourly bars (markets downloaded, and why a reference time has no row)")
    for r in fams:
        say(f"  {r['name']:34} {r['cover']}")
        summary.setdefault("coverage", {})[r["name"]] = r["cover"]
    pooled = [x for x in (pool(name, [r for r in fams if keep(r["name"])]) for name, keep in POOLS) if x is not None]
    calibration(fams+pooled, say, summary)
    tradeable(fams+pooled, say, summary)
    tape_check(fams, say, summary)
    tail(fams, say, summary)
    capacity(fams, say, summary)
    multiplicity(fams, say, summary)
    shrinkage(fams, say, summary)
    summary["requests"] = load(RAW/"download-log.json", [])
    tag = "" if CLOCK[0] == "start" else "-close-clock"
    dump(OUT/f"summary{tag}.json", summary)
    (OUT/f"report{tag}.txt").write_text("\n".join(lines)+"\n")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("steps", nargs="+", choices=("discover", "list", "leads", "flow", "bars", "archive", "report"))
    parser.add_argument("--minutes", type=float, default=9)
    parser.add_argument("--only", nargs="*", default=None, help="family names to restrict flow/bars to")
    parser.add_argument("--cap", type=int, default=3000, help="markets per family for `bars` (whole events)")
    parser.add_argument("--clock", choices=("start", "close"), default="start", help="`report`: event clock for mention families")
    parser.add_argument("--sample", type=int, default=None, help="`flow`: markets per mention family for the pre-event trade pull (default 40)")
    parser.add_argument("--thin", type=int, default=0, help="`bars`: also take one market in N from every other event")
    args = parser.parse_args()
    for step in args.steps:
        began = time.time()
        deadline[0] = began+60*args.minutes
        log.clear()
        if step in ("discover", "list", "flow", "bars", "archive"):
            with httpx.Client(timeout=90) as client:
                try:
                    dict(discover=discover, list=list_all, flow=lambda c: (flow(c, args.only, None), pretrades(c, args.only, args.sample or 40)), archive=archive,
                         bars=lambda c: bars(c, args.only, args.cap, args.thin))[step](client)
                except OutOfTime:
                    print(f"[{step}] out of time; run again to resume", flush=True)
                finally:
                    finish_log(step, began)
        elif step == "leads":
            leads()
        else:
            CLOCK[0] = args.clock
            report()


if __name__ == "__main__":
    main()
