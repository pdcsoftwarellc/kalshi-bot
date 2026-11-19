"""Shared helpers for the part-2 inventory audit: event-clustered bootstrap and chronological halves.

Read-only over saved outputs. Percent per bet = net cents / entry price cents (profit / stake).
"""
import random
from collections import defaultdict


def boot(rows, key, val, draws=3000, seed=7):
    """Mean of val over rows, with a 95% interval resampling whole events (key)."""
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(val(r))
    gs = [(sum(g), len(g)) for g in groups.values()]
    n = sum(c for _, c in gs)
    if not n:
        return None
    m = sum(s for s, _ in gs)/n
    if len(gs) < 10:
        return dict(n=n, events=len(gs), mean=m, lo=None, hi=None)
    rng = random.Random(seed)
    means = []
    k = len(gs)
    for _ in range(draws):
        s = c = 0
        for _ in range(k):
            a, b = gs[rng.randrange(k)]
            s += a
            c += b
        means.append(s/c)
    means.sort()
    return dict(n=n, events=k, mean=m, lo=means[int(.025*draws)], hi=means[int(.975*draws)])


def halves(rows, key, order, val):
    """Split events chronologically (by each event's first order value) into two halves; mean of val in each."""
    first = {}
    for r in rows:
        k = key(r)
        o = order(r)
        if k not in first or o < first[k]:
            first[k] = o
    ev = sorted(first, key=lambda k: first[k])
    early = set(ev[:len(ev)//2])
    a = [val(r) for r in rows if key(r) in early]
    b = [val(r) for r in rows if key(r) not in early]
    mean = lambda v: sum(v)/len(v) if v else None
    return (len(a), mean(a)), (len(b), mean(b))


def fmt(b, h=None, unit="%"):
    if b is None:
        return "n=0"
    s = f'n={b["n"]} ev={b["events"]} mean={b["mean"]:+.2f}{unit}'
    if b["lo"] is not None:
        s += f' [{b["lo"]:+.2f},{b["hi"]:+.2f}]'
    if h:
        (na, ma), (nb, mb) = h
        s += f' halves {ma:+.2f}{unit}(n={na}) / {mb:+.2f}{unit}(n={nb})' if ma is not None and mb is not None else ''
    return s
