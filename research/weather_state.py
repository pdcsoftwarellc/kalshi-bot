"""Do Kalshi daily-high markets lag the public hourly observations? Research only; no orders.

Inputs: data/raw/weather-state/*-markets.json and *-bars.json (weather_state_download.py) and
hourly METARs from the Iowa Environmental Mesonet. Rules below were fixed before results were read.
"""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import random
import re
import statistics
import sys
import time
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from api_cost import save  # noqa: E402
from weather_state_download import CITIES, FIRST, LAST, RAW  # noqa: E402

OUT = ROOT / "data/processed/weather-state"
UTC = timezone.utc
DELAY = 20 * 60            # an observation is usable 20 minutes after its stamp
DECIDE_MINUTE = 20         # decisions are taken at H:20 local clock time
HOURS = (15, 16, 17, 18)
DROPS = (2, 4)
CAPS = (90, 95, 98)
MARGINS = (0, 1)           # certainty trade: range top must be below observed max minus margin
MAX_WAIT = 3600            # the next bar must start within an hour of the decision
IEM = "https://mesonet.agron.iastate.edu"


def download(name, url, params):
    path = RAW / name
    if path.exists():
        return path.read_text()
    for attempt in range(5):
        try:
            resp = httpx.get(url, params=params, timeout=120, headers={"User-Agent": "research-client"})
            if resp.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(resp.text)
                return resp.text
        except httpx.RequestError:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("observations unavailable: " + name)


def fahrenheit(tenths_c):
    return round(tenths_c / 10 * 9 / 5 + 32)


def observations(station):
    """[(ts, whole F, six-hour max F or None)] from routine and special METARs carrying a tenths group."""
    a, b = FIRST - timedelta(days=1), LAST + timedelta(days=2)
    text = download(f"{station}-metar.csv", IEM + "/cgi-bin/request/asos.py", [
        ("station", station), ("data", "metar"), ("year1", a.year), ("month1", a.month), ("day1", a.day),
        ("year2", b.year), ("month2", b.month), ("day2", b.day), ("tz", "UTC"), ("format", "onlycomma"),
        ("missing", "M"), ("report_type", 3), ("report_type", 4)])
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split(",", 2)
        if len(parts) < 3:
            continue
        remarks = parts[2].split(" RMK ", 1)
        if len(remarks) < 2:
            continue
        temp = re.search(r"\bT([01])(\d{3})[01]\d{3}\b", remarks[1])
        if not temp:
            continue
        ts = datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=UTC).timestamp()
        six = re.search(r"\b1([01])(\d{3})\b", remarks[1])
        rows.append((ts, fahrenheit((-1 if temp[1] == "1" else 1) * int(temp[2])),
                     fahrenheit((-1 if six[1] == "1" else 1) * int(six[2])) if six else None))
    return sorted(set(rows))


def climate_reports(site):
    """Official NWS daily climate report highs, to confirm Kalshi's settled value uses the same station."""
    highs = {}
    for year in sorted({FIRST.year, LAST.year}):
        data = json.loads(download(f"{site}-cli-{year}.json", IEM + "/json/cli.py", {"station": site, "year": year}))
        for row in data["results"]:
            if isinstance(row.get("high"), (int, float)):
                highs[row["valid"]] = row["high"]
    return highs


def bounds(m):
    if m["kind"] == "less":
        return -999, int(m["cap"]) - 1
    if m["kind"] == "greater":
        return int(m["floor"]) + 1, 999
    return int(m["floor"]), int(m["cap"])


def city_days():
    days = []
    for series, (city, station, site, zone, offset) in CITIES.items():
        listing = RAW / f"{series}-markets.json"
        if not listing.exists():
            continue
        bars = {}
        if (RAW / f"{series}-bars.json").exists():
            bars = {m["ticker"]: m["bars"] for m in json.loads((RAW / f"{series}-bars.json").read_text())}
        obs, official = observations(station), climate_reports(site)
        groups = defaultdict(list)
        for m in json.loads(listing.read_text()):
            groups[m["day"]].append(m)
        for day, markets in sorted(groups.items()):
            markets.sort(key=lambda m: bounds(m)[0])
            ranges = [bounds(m) for m in markets]
            yes = [i for i, m in enumerate(markets) if m["result"] == "yes"]
            final = markets[0]["final"]
            if final is None:       # Kalshi left the settled value blank on some days: use the NWS climate report
                final = official.get(day)
            contiguous = all(a[1] + 1 == b[0] for a, b in zip(ranges, ranges[1:]))
            if len(yes) != 1 or final is None or not contiguous or not ranges[yes[0]][0] <= final <= ranges[yes[0]][1]:
                continue
            d = date.fromisoformat(day)
            start = datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() - 3600 * offset   # midnight standard time
            days.append({
                "city": city, "series": series, "day": day, "zone": zone, "start": start, "end": start + 86400,
                "markets": markets, "ranges": ranges, "won": yes[0], "final": int(final), "official": official.get(day),
                "obs": [o for o in obs if start <= o[0] < start + 86400],
                "bars": [bars.get(m["ticker"]) for m in markets] if all(m["ticker"] in bars for m in markets) else None,
            })
    return days


def decision_time(day, hour, minute=DECIDE_MINUTE):
    d = date.fromisoformat(day["day"])
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=ZoneInfo(day["zone"])).timestamp()


def state(day, when):
    """Observed maximum and latest temperature using only reports published by `when`."""
    seen = [o for o in day["obs"] if o[0] + DELAY <= when]
    if not seen:
        return None
    # A six-hour maximum counts only if its window started inside this climate day (ten minutes of slack).
    top = max([o[1] for o in seen] + [o[2] for o in seen if o[2] is not None and o[0] - 21600 >= day["start"] - 600])
    if when - seen[-1][0] > 2 * 3600 + DELAY:
        return None         # stale feed
    return top, seen[-1][1]


def index(day, value):
    return next(i for i, (lo, hi) in enumerate(day["ranges"]) if lo <= value <= hi)


def fee(price):
    return 7 * (price / 100) * (1 - price / 100)       # cents per contract


def entry(bars, when, yes):
    """Ask at the open of the first bar starting at or after `when`; also quote state and nearby volume."""
    for bar in bars:
        begin = bar[0] - 60
        if begin >= when:
            if begin - when > MAX_WAIT:
                return None
            bid, ask = bar[1], bar[3]
            price = ask if yes else (None if bid is None else 100 - bid)
            if price is None or not 0 < price < 100:
                return None
            two_sided = bid is not None and ask is not None and bid > 0 and ask < 100
            volume = sum(b[5] for b in bars if when - 1800 <= b[0] - 60 < when + 1800)
            return price, two_sided, volume, begin - when
    return None


def trades(days):
    """{rule: [trade]}; at most one trade per city-day per rule."""
    book = defaultdict(list)
    for day in days:
        if day["bars"] is None:
            continue
        first_certain = {m: False for m in MARGINS}
        for hour in range(12, 23):
            when = decision_time(day, hour)
            if when >= day["markets"][0]["close"] - 120:
                continue
            now = state(day, when)
            if now is None:
                continue
            top, current = now
            at = index(day, top)

            def take(rule, i, yes, cap):
                found = entry(day["bars"][i], when, yes)
                if not found or found[0] > cap:
                    return False
                price, two_sided, volume, wait = found
                won = (day["won"] == i) == yes
                book[rule].append({"day": day["day"], "city": day["city"], "ticker": day["markets"][i]["ticker"],
                                   "price": price, "won": won, "net": (100 if won else 0) - price - fee(price),
                                   "two_sided": two_sided, "volume": volume, "wait": wait,
                                   "top": top, "final": day["final"]})
                return True

            if hour in HOURS:
                for drop in DROPS:
                    if top - current >= drop:
                        for cap in CAPS:
                            take(("YES range of observed max", hour, f"drop>={drop}", cap), at, True, cap)
                            if at + 1 < len(day["ranges"]):
                                take(("NO range above", hour, f"drop>={drop}", cap), at + 1, False, cap)
            for margin in MARGINS:
                below = [i for i, (lo, hi) in enumerate(day["ranges"]) if hi < top - margin]
                if not below:
                    continue
                if hour in HOURS:
                    take(("NO range below (certainty)", hour, f"margin {margin}", 98), below[-1], False, 98)
                if not first_certain[margin]:
                    first_certain[margin] = take(("NO range below (certainty)", "first 12-22", f"margin {margin}", 98),
                                                 below[-1], False, 98)
    return book


def summarize(rows, all_days, seed=7):
    if not rows:
        return None
    days = sorted({r["day"] for r in rows})
    span = (date.fromisoformat(all_days[-1]) - date.fromisoformat(all_days[0])).days + 1
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r["net"])
    rng, means = random.Random(seed), []
    pool = [by_day.get(d, []) for d in all_days]        # resample calendar days, including days with no trade
    for _ in range(2000):
        sample = [x for _ in pool for x in rng.choice(pool)]
        if sample:
            means.append(sum(sample) / len(sample))
    means.sort()
    cost = [r["price"] + fee(r["price"]) for r in rows]
    return {
        "city_days": len({(r["city"], r["day"]) for r in rows}), "days": len(days), "trades": len(rows),
        "losses": sum(not r["won"] for r in rows), "win_rate": sum(r["won"] for r in rows) / len(rows),
        "break_even": sum(cost) / len(cost) / 100, "net": sum(r["net"] for r in rows) / len(rows),
        "low": means[int(.025 * len(means))] if means else None, "high": means[int(.975 * len(means)) - 1] if means else None,
        "per_week": 7 * len(rows) / span, "median_price": statistics.median(r["price"] for r in rows),
        "median_volume": statistics.median(r["volume"] for r in rows),
        "two_sided": sum(r["two_sided"] for r in rows) / len(rows),
        "median_wait": statistics.median(r["wait"] for r in rows),
    }


def fair_value(days):
    cells = defaultdict(Counter)
    for day in days:
        for hour in range(12, 24):
            when = decision_time(day, hour)
            if when >= day["end"]:
                continue
            now = state(day, when)
            if now is None:
                continue
            top, current = now
            fall = top - current
            bucket = "0-1" if fall < 2 else "2-3" if fall < 4 else "4-7" if fall < 8 else "8+"
            step = day["won"] - index(day, top)
            cells[(hour, bucket)]["n"] += 1
            cells[(hour, bucket)]["lower" if step < 0 else "same" if step == 0 else "+1" if step == 1 else "+2 or more"] += 1
            cells[(hour, bucket)]["exact"] += day["final"] == top
            cells[(hour, bucket)]["gap"] += day["final"] - top
    return cells


def main():
    days = city_days()
    days.sort(key=lambda d: (d["day"], d["city"]))
    usable = [d for d in days if d["obs"]]
    if "--city" in sys.argv:
        usable = [d for d in usable if d["city"] == sys.argv[sys.argv.index("--city") + 1]]
    lines = []
    say = lambda *parts: lines.append(" ".join(str(p) for p in parts))  # noqa: E731
    say("City-days with settlement and observations:", len(usable), "from", usable[0]["day"], "to", usable[-1]["day"])
    say("By city (all / with bars):", {c: (sum(d["city"] == c for d in usable), sum(d["city"] == c and d["bars"] is not None for d in usable))
                                       for c in sorted({d["city"] for d in usable})})

    # Settlement source checks.
    both = [d for d in usable if d["official"] is not None]
    say("\nKalshi settled value vs NWS daily climate report (IEM copy): equal on",
        sum(d["final"] == d["official"] for d in both), "of", len(both),
        "; differences:", dict(Counter(d["final"] - d["official"] for d in both if d["final"] != d["official"])))
    gap = Counter()
    for d in usable:
        whole = state(d, d["end"] + DELAY)
        if whole:
            gap[d["final"] - whole[0]] += 1
    total = sum(gap.values())
    say("Official high minus full-day maximum of hourly/special METARs and six-hour groups (degrees: share):",
        {k: round(v / total, 4) for k, v in sorted(gap.items())}, "n", total)
    say("Official high BELOW that observed maximum:", sum(v for k, v in gap.items() if k < 0), "of", total,
        "; by 2 or more:", sum(v for k, v in gap.items() if k <= -2))

    say("\nFAIR VALUE: where the official high landed relative to the range holding the maximum observed so far")
    say("local hour (decision at H:20, obs delayed 20 min) | fall from max | n | lower | same | +1 range | +2 or more | official==observed max | mean official-observed")
    cells = fair_value(usable)
    table = []
    for (hour, bucket), c in sorted(cells.items()):
        n = c["n"]
        row = {"hour": hour, "fall": bucket, "n": n, **{k: c[k] / n for k in ("lower", "same", "+1", "+2 or more", "exact")}, "gap": c["gap"] / n}
        table.append(row)
        if n >= 20:
            say(f"{hour:>2} | {bucket:>4} | {n:>5} | {row['lower']:.3f} | {row['same']:.3f} | {row['+1']:.3f} | {row['+2 or more']:.3f} | {row['exact']:.3f} | {row['gap']:+.2f}")

    # Trading test.
    with_bars = [d for d in usable if d["bars"] is not None]
    calendar = sorted({d["day"] for d in with_bars})
    cut = calendar[int(len(calendar) * .6)]
    early_days, late_days = [d for d in calendar if d < cut], [d for d in calendar if d >= cut]
    say(f"\nTRADING: {len(with_bars)} city-days with complete bars on {len(calendar)} dates; earliest 60% = {early_days[0]}..{early_days[-1]}"
        f" ({len(early_days)} dates), latest 40% = {late_days[0]}..{late_days[-1]} ({len(late_days)} dates)")
    book = trades(with_bars)
    say("Rules tried:", len(book))
    say("rule | hour | condition | cap | part | city-days | trades | full losses | win | break-even | net c | 95% CI | trades/wk | med price | med vol +-30m | two-sided | med wait s")
    results = []
    for rule in sorted(book, key=lambda r: (r[0], str(r[1]), r[2], r[3])):
        row = {"rule": rule}
        for part, part_days in (("early", early_days), ("late", late_days)):
            chosen = set(part_days)
            s = summarize([t for t in book[rule] if t["day"] in chosen], part_days)
            row[part] = s
            if s:
                say(f"{rule[0]} | {rule[1]} | {rule[2]} | {rule[3]} | {part} | {s['city_days']} | {s['trades']} | {s['losses']} | {s['win_rate']:.3f} | "
                    f"{s['break_even']:.3f} | {s['net']:+.2f} | [{s['low']:+.2f}, {s['high']:+.2f}] | {s['per_week']:.1f} | {s['median_price']:.0f} | "
                    f"{s['median_volume']:.0f} | {s['two_sided']:.2f} | {s['median_wait']:.0f}")
            else:
                say(f"{rule[0]} | {rule[1]} | {rule[2]} | {rule[3]} | {part} | no trades")
        row["passes"] = bool(row["early"] and row["late"] and row["early"]["net"] > 0 and row["late"]["net"] > 0 and row["late"]["low"] > 0)
        results.append(row)
    say("\nPasses (positive in both parts, later interval above zero):",
        [" / ".join(str(x) for x in r["rule"]) for r in results if r["passes"]] or "none")
    ranked = sorted((r for r in results if r["early"] and r["early"]["trades"] >= 30), key=lambda r: -r["early"]["low"])
    say("Best five by lower bound on the EARLY part (at least 30 early trades):")
    for r in ranked[:5]:
        late = r["late"]
        say("  ", " / ".join(str(x) for x in r["rule"]), "early net", f"{r['early']['net']:+.2f}", "CI low", f"{r['early']['low']:+.2f}",
            "| late", "none" if not late else f"{late['net']:+.2f} [{late['low']:+.2f}, {late['high']:+.2f}] n={late['trades']}")

    losses = [t for rule, rows in book.items() if rule[0].startswith("NO range below") for t in rows if not t["won"]]
    say("\nCertainty-trade full losses (distinct):")
    for t in sorted({(t["day"], t["city"], t["ticker"], t["top"], t["final"]) for t in losses}):
        say("  ", *t)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    save(OUT / "fair_value.json", table)
    save(OUT / "trading.json", [{"rule": list(r["rule"]), "early": r["early"], "late": r["late"], "passes": r["passes"]} for r in results])
    save(OUT / "trades.json", [dict(t, rule=list(rule)) for rule, rows in book.items() for t in rows])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
