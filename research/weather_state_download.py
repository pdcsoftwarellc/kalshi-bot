"""Download Kalshi daily high-temperature markets and afternoon one-minute bars. Public reads only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
import json
from pathlib import Path
import sys
import threading
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from api_cost import fetch, pages, save, timestamp  # noqa: E402

RAW = ROOT / "data/raw/weather-state"
# series -> (city, ASOS id used by IEM, NWS climate site, IANA zone, standard UTC offset hours)
CITIES = {
    "KXHIGHNY": ("NYC", "NYC", "KNYC", "America/New_York", -5),
    "KXHIGHCHI": ("CHI", "MDW", "KMDW", "America/Chicago", -6),
    "KXHIGHMIA": ("MIA", "MIA", "KMIA", "America/New_York", -5),
    "KXHIGHAUS": ("AUS", "AUS", "KAUS", "America/Chicago", -6),
    "KXHIGHLAX": ("LAX", "LAX", "KLAX", "America/Los_Angeles", -8),
    "KXHIGHDEN": ("DEN", "DEN", "KDEN", "America/Denver", -7),
    "KXHIGHPHIL": ("PHL", "PHL", "KPHL", "America/New_York", -5),
}
FIRST, LAST = date(2025, 12, 1), date(2026, 11, 15)
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
HOURS_BEFORE_CLOSE = 15
PACE = threading.Lock()
LAST_CALL = [0.0]


def day_of(ticker):
    part = ticker.split("-")[1]
    return date(2000 + int(part[:2]), MONTHS[part[2:5]], int(part[5:]))


def polite(client, path, params, deadline, gap=0.25):
    """fetch with pacing shared by both workers; None once the deadline has passed."""
    while time.time() < deadline:
        with PACE:
            wait = LAST_CALL[0] + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            LAST_CALL[0] = time.time()
        try:
            return fetch(client, path, params)
        except RuntimeError as error:
            if "HTTP 4" in str(error):
                return {"candlesticks": [], "error": str(error)}
            time.sleep(20)
    return None


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def listing(client, series):
    rows = {}
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        for m in pages(client, path, "markets", series_ticker=series, limit=1000, **extra):
            try:
                day = day_of(m["ticker"])
            except (ValueError, KeyError, IndexError):
                continue
            if not FIRST <= day <= LAST or m.get("result") not in ("yes", "no"):
                continue
            rows[m["ticker"]] = {
                "ticker": m["ticker"], "event": m["event_ticker"], "day": str(day),
                "archived": path.startswith("/historical"),
                "kind": m["strike_type"], "floor": m.get("floor_strike"), "cap": m.get("cap_strike"),
                "result": m["result"], "final": number(m.get("expiration_value")),
                "open": timestamp(m["open_time"]), "close": timestamp(m["close_time"]),
                "volume": float(m.get("volume_fp") or m.get("volume") or 0),
                "rules": m.get("rules_primary"),
            }
    return sorted(rows.values(), key=lambda r: (r["day"], r["ticker"]))


def money(group, field):
    value = group.get(field + "_dollars", group.get(field)) if group else None
    return None if value is None else round(100 * float(value), 2)


def compact(bars):
    out = []
    for b in sorted(bars, key=lambda b: b["end_period_ts"]):
        bid, ask, price = b.get("yes_bid"), b.get("yes_ask"), b.get("price")
        out.append([b["end_period_ts"], money(bid, "open"), money(bid, "close"), money(ask, "open"), money(ask, "close"),
                    float(b.get("volume_fp", b.get("volume")) or 0), money(price, "close")])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=38)
    args = parser.parse_args()
    deadline = time.time() + 60 * args.minutes
    client = httpx.Client(timeout=30)
    markets = {}
    for series in CITIES:
        for attempt in range(6):
            try:
                markets[series] = listing(client, series)
                break
            except RuntimeError:
                time.sleep(20)
        else:
            markets[series] = []
        save(RAW / f"{series}-markets.json", markets[series])
        print(series, len(markets[series]), "markets", round(sum(m["volume"] for m in markets[series])), "contracts", flush=True)

    # New York complete first, then the other cities on every third day so a cut-off still spans all seasons.
    order = []
    first = next(iter(CITIES))
    order += [(first, m) for m in markets[first]]
    for phase in range(3):
        for series in list(CITIES)[1:]:
            order += [(series, m) for m in markets[series] if date.fromisoformat(m["day"]).toordinal() % 3 == phase]
    got = {}

    def work(item):
        series, m = item
        start = int(m["close"]) - 3600 * HOURS_BEFORE_CLOSE
        path = (f"/historical/markets/{m['ticker']}/candlesticks" if m["archived"]
                else f"/series/{series}/markets/{m['ticker']}/candlesticks")
        data = polite(client, path, {"start_ts": start, "end_ts": int(m["close"]), "period_interval": 1}, deadline)
        if data is not None:
            got[m["ticker"]] = compact(data.get("candlesticks") or [])

    with ThreadPoolExecutor(2) as pool:
        for count, _ in enumerate(pool.map(work, order)):
            if count % 500 == 0:
                print(datetime.now().strftime("%H:%M:%S"), count, "of", len(order), "requested;", len(got), "held", flush=True)
    for series in CITIES:
        rows = [dict(m, bars=got[m["ticker"]]) for m in markets[series] if m["ticker"] in got]
        days = {}
        for m in markets[series]:
            days.setdefault(m["day"], []).append(m["ticker"] in got)
        keep = {d for d, flags in days.items() if all(flags)}
        save(RAW / f"{series}-bars.json", [r for r in rows if r["day"] in keep])
        print(series, len(keep), "complete days", flush=True)


if __name__ == "__main__":
    main()
