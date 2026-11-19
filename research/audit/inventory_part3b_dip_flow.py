"""Audit helper (read-only): intervals the dip_flow report does not give, from data/processed/favorite-dip/flow-rows.json.
One row = one filled 80c bid on a 90c+ tennis favourite = one match (checked below), so resampling rows resamples matches.
Return per bet = (100*payout - 80)/80, no fee (as dip_flow.result; ATP and WTA would owe an assumed 0.28c maker fee).
COMPUTED here (4000 bootstrap draws):
  1. each flow split at the HOLDOUT median (as the script): mean % of the low and high side per period, and the
     difference high minus low with a 95% interval, per period and pooled;
  2. sized bids: per-contract return when filled for min(Q, seller-initiated contracts below 80c in 30 minutes),
     and the paired difference Q=1000 minus Q=100 and Q=10000 minus Q=100, resampling matches."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
rows = [r for r in json.load(open(ROOT/"data/processed/favorite-dip/flow-rows.json")) if r.get("found")]
print("fills with a trade record", len(rows), "distinct events", len({r["event"] for r in rows}))
rng = np.random.default_rng(11)
ret = lambda r: 100*(100*r["payout"]-80)/80
GROUPS = {"all tours": None, "Challenger+WTA": ("Challenger", "WTA"), "Challenger+WTA+WTA Challenger": ("Challenger", "WTA", "WTA Challenger")}
KEYS = ("seller_sweeps", "seller_sweep_median", "seller_sweep_max", "seller_top_share", "seller_share_dip", "seller_share_below", "dip_volume",
        "below_5m", "low_5m", "high_2m_before", "first_print_size", "seconds_into_bar", "volume_below_before_t0", "price_10m_before",
        "warmup_spread", "spread_before_dip", "minutes_before_settlement", "start", "price_after_5m")
KNOWN_AT_FILL = {"high_2m_before", "price_10m_before", "warmup_spread", "spread_before_dip", "start", "seller_share_before", "first_print_size"}


def diff_ci(lo, hi):
    if len(lo) < 5 or len(hi) < 5:
        return None
    lo, hi = np.array(lo), np.array(hi)
    d = hi[rng.integers(0, len(hi), (4000, len(hi)))].mean(axis=1)-lo[rng.integers(0, len(lo), (4000, len(lo)))].mean(axis=1)
    return np.percentile(d, [2.5, 97.5])


print("\n1. FLOW SPLITS (cut = holdout median of the group). 'entry' = knowable when the bid fills or before; 'after' = known only later")
for name, tours in GROUPS.items():
    part = [r for r in rows if tours is None or r["tour"] in tours]
    for key in KEYS:
        hold = [r[key] for r in part if r["period"] == "holdout" and r.get(key) is not None]
        if len(hold) < 20:
            continue
        cut = float(np.median(hold))
        line = f"  {name:30} {key:26} {'entry' if key in KNOWN_AT_FILL else 'after':5} cut {cut:9.2f}"
        for period in ("holdout", "recent", "both"):
            sub = [r for r in part if period in (r["period"], "both") and r.get(key) is not None]
            lo, hi = [ret(r) for r in sub if r[key] <= cut], [ret(r) for r in sub if r[key] > cut]
            ci = diff_ci(lo, hi)
            if ci is None:
                line += f" | {period} n {len(lo)}/{len(hi)} too few"
                continue
            line += f" | {period} low {np.mean(lo):+.1f}% (n={len(lo)}) high {np.mean(hi):+.1f}% (n={len(hi)}) high-low {np.mean(hi)-np.mean(lo):+.1f} [{ci[0]:+.1f},{ci[1]:+.1f}]"
        print(line)

print("\n2. SIZED BIDS: % per contract (of 80c) when filled for min(Q, seller-initiated contracts below 80c in 30 min)")
for name, tours in (("Challenger+WTA", ("Challenger", "WTA")), ("Challenger", ("Challenger",)), ("WTA", ("WTA",)), ("WTA Challenger", ("WTA Challenger",)),
                    ("Challenger+WTA+WTA Challenger", ("Challenger", "WTA", "WTA Challenger")), ("ATP", ("ATP",)), ("ITF men", ("ITF men",))):
    for period in ("holdout", "recent", "both"):
        part = [r for r in rows if r["tour"] in tours and period in (r["period"], "both")]
        if len(part) < 10:
            continue
        net = np.array([ret(r) for r in part])
        sold = np.array([r["below_30m_sold"] for r in part])
        pick = rng.integers(0, len(part), (4000, len(part)))
        line = f"  {name:30} {period:7} matches {len(part):3} median contracts available {np.median(sold):6.0f} (p25 {np.percentile(sold, 25):.0f})"
        per = {}
        for q in (1, 100, 1000, 10000):
            got = np.minimum(q, sold)
            per[q] = (got[pick]*net[pick]).sum(axis=1)/np.maximum(got[pick].sum(axis=1), 1e-9)
            point = (got*net).sum()/got.sum()
            lo, hi = np.percentile(per[q], [2.5, 97.5])
            line += f" | Q={q}: {point:+.1f}% [{lo:+.1f},{hi:+.1f}] mean filled {got.mean():.0f}"
        for q in (1000, 10000):
            d = per[q]-per[100]
            point = (np.minimum(q, sold)*net).sum()/np.minimum(q, sold).sum()-(np.minimum(100, sold)*net).sum()/np.minimum(100, sold).sum()
            line += f" | Q={q} minus Q=100: {point:+.1f} [{np.percentile(d, 2.5):+.1f},{np.percentile(d, 97.5):+.1f}]"
        print(line)
