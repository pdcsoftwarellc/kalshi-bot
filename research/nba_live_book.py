"""Displayed size and lifetime of 97-99c asks in the 4th quarter, from the live recorder.

Reads data/raw/live-games/*.jsonl (written by live_games.py, read-only here; a
partial last line is skipped). For each NBA game, each time either side's ask
sits at 97, 98 or 99c in the 4th quarter or overtime, one "episode" starts. It
records the size shown at that ask and how long a buyer with a limit at that
price could still have been filled: until the side's ask first moves above it.

The recorder polls about every 15 seconds and writes a row only when the game
state or the best bid/ask price changes, so durations are good to roughly one
poll, and size is what was shown on the rows that happened to be written.
The NO side's ask is 100 minus the YES bid, with the YES bid's size.
"""
import json
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, save

RAW = ROOT / "data/raw/live-games"
OUT = ROOT / "data/processed/nba-fills"


def load():
    games, passes = {}, set()
    for file in sorted(RAW.glob("*.jsonl")):
        with file.open() as lines:
            for line in lines:
                try:
                    r = json.loads(line)
                    passes.add(r["ts"])
                    if not r["event"].startswith("KXNBAGAME"):
                        continue
                    q, s = r["quote"], r.get("state") or {}
                    bid, ask = round(100*float(q["yes_bid_dollars"])), round(100*float(q["yes_ask_dollars"]))
                    row = dict(ts=r["ts"], period=int(s.get("period") or 0), clock=s.get("period_remaining_time"),
                               status=str(s.get("status", "")).lower(), home=s.get("home_points"), away=s.get("away_points"),
                               yes=(ask, float(q["yes_ask_size_fp"] or 0)), no=(100-bid, float(q["yes_bid_size_fp"] or 0)))
                except (ValueError, KeyError, TypeError):
                    continue                                   # a partial last line, or a row without a book
                games.setdefault(r["ticker"], []).append(row)
    return games, sorted(passes)


def episodes(ticker, rows, passes):
    found = []
    live = lambda r: r["period"] >= 4 and r["status"] not in ("closed", "complete", "final", "finished", "ended")
    for side in ("yes", "no"):
        for i, r in enumerate(rows):
            price, size = r[side]
            if not (live(r) and 97 <= price <= 99):
                continue
            if i and live(rows[i-1]) and rows[i-1][side][0] == price:
                continue                                       # the same episode continues
            end, sizes, censored = None, [size], True
            for later in rows[i+1:]:
                if later[side][0] > price or not live(later):
                    end, censored = later["ts"], not live(later)
                    break
                if later[side][0] == price:
                    sizes.append(later[side][1])
            last_pass = passes[-1]
            found.append(dict(ticker=ticker, side=side, ask=price, start=r["ts"], period=r["period"], clock=r["clock"],
                              score=f'{r["away"]}-{r["home"]}', size_at_start=size, min_size_seen=min(sizes),
                              seconds_available=(end if end is not None else last_pass)-r["ts"],
                              ended_by="price moved away" if not censored else "game or recording ended"))
    return sorted(found, key=lambda e: e["start"])


def main():
    games, passes = load()
    gaps = sorted(b-a for a, b in zip(passes, passes[1:]))
    report = dict(nba_games_recorded=len(games), games_reaching_4th_quarter=sum(any(r["period"] >= 4 for r in g) for g in games.values()),
                  poll_gap_seconds=dict(median=median(gaps), p95=gaps[int(.95*len(gaps))], max=gaps[-1]) if gaps else None, games={})
    every = []
    for ticker, rows in sorted(games.items()):
        rows.sort(key=lambda r: r["ts"])
        eps = episodes(ticker, rows, passes)
        every += eps
        report["games"][ticker] = dict(rows=len(rows), last_period=rows[-1]["period"], last_clock=rows[-1]["clock"], episodes=eps)
    if every:
        firsts = [g["episodes"][0] for g in report["games"].values() if g["episodes"]]
        for name, group in (("all_episodes", every), ("first_episode_per_game", firsts)):
            sizes, secs = [e["size_at_start"] for e in group], [e["seconds_available"] for e in group]
            report[name] = dict(n=len(group), games=len({e["ticker"] for e in group}), median_size=median(sizes), min_size=min(sizes),
                                share_size_ge_100=sum(s >= 100 for s in sizes)/len(sizes),
                                share_size_ge_1000=sum(s >= 1000 for s in sizes)/len(sizes),
                                median_seconds=median(secs), share_ge_15s=sum(s >= 15 for s in secs)/len(secs),
                                share_ge_60s=sum(s >= 60 for s in secs)/len(secs))
    save(OUT/"live-book.json", report)
    print(f'NBA games in the recording: {report["nba_games_recorded"]}, reaching the 4th quarter: {report["games_reaching_4th_quarter"]}; '
          f'poll gaps {report["poll_gap_seconds"]}')
    for ticker, g in report["games"].items():
        print(f'\n{ticker}: {g["rows"]} rows, last seen period {g["last_period"]} clock {g["last_clock"]}, {len(g["episodes"])} episodes')
        for e in g["episodes"]:
            print(f'  {e["side"]:3} ask {e["ask"]}c  Q{e["period"]} {e["clock"]:>6} score {e["score"]:>8}  size {e["size_at_start"]:10.0f} '
                  f'(min seen {e["min_size_seen"]:10.0f})  available {e["seconds_available"]:6.0f}s  ({e["ended_by"]})')
    for name in ("all_episodes", "first_episode_per_game"):
        if name in report:
            print(f"\n{name}: {report[name]}")


if __name__ == "__main__":
    main()
