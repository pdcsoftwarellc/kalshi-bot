"""Download hourly bars for the part of research/favorites.py's seeded sample that was never fetched.

The original run fetched 939 of its 3,391 sampled markets (shuffled, so a fair subset) before its time budget
ran out. The remaining 2,452 were chosen by the same seed before any result existed, so they are an untouched
holdout. Public, unauthenticated Kalshi market-data reads only; at most one request every 3 seconds, 60-300 s back-off on HTTP 429; generic User-Agent; no
account, no orders. Bars are cached under data/processed/audit/inventory/favorites-holdout/ (never the
original cache) in the original's slimmed format. Safe to rerun: finished markets are skipped.
"""
import hashlib, json, random, sys, time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import favorites as f

BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEST = ROOT/"data/processed/audit/inventory/favorites-holdout/bars"
GAP = 3.0


def main(budget):
    DEST.mkdir(parents=True, exist_ok=True)
    picked = f.sample(f.references(f.load_universe()))
    random.Random(5).shuffle(picked)                      # the original's own order
    stop, last, n = time.time()+budget, 0.0, dict(done=0, bars=0, empty=0, skipped=0, error=0, http429=0)
    limited, streak = None, 0
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        for m in picked:
            paths, params = f.bar_request(m)
            if any((f.RAW/"bars"/(hashlib.sha256(json.dumps([p, params], sort_keys=True).encode()).hexdigest()+".json")).exists() for p in paths):
                continue                                  # fetched by the original study
            out = DEST/(m["ticker"]+".json")
            if out.exists():
                n["skipped"] += 1
                continue
            if time.time() > stop:
                break
            rows = None
            for path in paths:
                tries = 0
                while True:
                    wait = last+GAP-time.time()
                    if wait > 0:
                        time.sleep(wait)
                    last = time.time()
                    try:
                        r = client.get(BASE+path, params=params)
                    except httpx.RequestError:
                        tries += 1
                        if tries >= 3:
                            break
                        time.sleep(30)
                        continue
                    if r.status_code == 429:
                        n["http429"] += 1
                        limited = limited or time.time()
                        if time.time()-limited > 900:
                            print("STOP: rate-limit responses for more than 15 minutes", n, flush=True)
                            return
                        pause = min(60*(streak+1), 300)
                        streak += 1
                        print(f"429: waiting {pause}s", flush=True)
                        time.sleep(pause)
                        continue
                    limited, streak = None, 0
                    if r.status_code >= 500:
                        tries += 1
                        if tries >= 3:
                            break
                        time.sleep(30)
                        continue
                    rows = f.slim_bars(r.json()) if r.status_code == 200 else []
                    break
                if rows:
                    break
            if rows is None:
                n["error"] += 1
                continue
            out.write_text(json.dumps(rows, separators=(",", ":")))
            n["done"] += 1
            n["bars" if rows else "empty"] += 1
            if n["done"] % 100 == 0:
                print(n, flush=True)
    left = sum(1 for m in picked if not (DEST/(m["ticker"]+".json")).exists() and not any((f.RAW/"bars"/(hashlib.sha256(json.dumps([p, f.bar_request(m)[1]], sort_keys=True).encode()).hexdigest()+".json")).exists() for p in f.bar_request(m)[0]))
    print("finished pass", n, "still missing", left, flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 540)
