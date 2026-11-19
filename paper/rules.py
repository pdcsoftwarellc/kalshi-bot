"""Decision logic for the two paper-traded rules. No network, no clock, no files.

Tennis dip: a favourite quoted 90c or more before play gets a resting bid at
80c, held to settlement. NBA late favourite: in the 4th quarter or overtime,
buy the leader when its ask is 97-99c; a second version rests a bid 1c below
that ask. Evidence for both is in research/parallel-round4.md.

Prices are whole cents on the side being bought ("yes" or "no").
"""
from datetime import datetime
import re
from statistics import median

PRE = {"not_started", "scheduled"}
DIP_LIVE = {"live", "inprogress"}
DONE = {"closed", "ended", "complete", "finished", "final", "cancelled", "canceled"}

DIP_START, DIP_BID, DIP_RELATIVE, DIP_DEEP, DIP_HIGH, DIP_LOW = 90, 80, 10, 65, 85, 85
DIP_FLOOR = 80                                     # simulated only: favourites at 80-90c before play
MIN_QUOTES, KEEP_QUOTES, MIN_SPAN, MAX_SPREAD, MAX_RANGE = 10, 15, 180, 4, 3
NBA_BAND, NBA_MIN_LEAD = (97, 99), 6


def qualifying(market):
    """"first" for a first-round tennis qualifying match, "later" for later qualifying rounds, else None.

    Read from the market's rules text ("... Qualification Round 1 ...", "... Qualification Final ...").
    "Qualifiers" is not qualifying: it names team ties such as the Billie Jean King Cup Qualifiers.
    A plain "Qualification" with no round is a first round (149 of 157 in the archive); a plain
    "Qualifying" is a qualifying final (5 of 5), so it counts as later.
    """
    found = re.search(r"\bQualif(ication|ying)\b( Final| Round (\d+))?", market.get("rules_primary") or "")
    if not found:
        return None
    if found.group(2) is None:
        return "first" if found.group(1) == "ication" else "later"
    return "first" if found.group(3) == "1" else "later"


def top(book):
    """Best bid and ask with displayed size for each side of one order book."""
    levels = book.get("orderbook_fp") or {}

    def best(side):
        rows = levels.get(side) or []
        if not rows:
            return None, 0.0
        price, size = max(rows, key=lambda row: float(row[0]))
        return round(100*float(price)), float(size)
    yes, yes_size = best("yes_dollars")
    no, no_size = best("no_dollars")
    return {"yes": dict(bid=yes, bid_size=yes_size, ask=None if no is None else 100-no, ask_size=no_size),
            "no": dict(bid=no, bid_size=no_size, ask=None if yes is None else 100-yes, ask_size=yes_size)}


def listed(market):
    """(yes bid, yes ask) from a market listing row, or None when the quote is unusable before play."""
    try:
        bid, ask = round(100*float(market["yes_bid_dollars"])), round(100*float(market["yes_ask_dollars"]))
    except (KeyError, TypeError, ValueError):
        return None
    return (bid, ask) if 0 < bid <= ask < 100 and ask-bid <= MAX_SPREAD else None


def favourite(quotes, floor=DIP_START):
    """(side, pre-match price) when recent pre-match midpoints are stable and one side is at `floor` (90c) or more.

    `quotes` is a list of (time, yes midpoint), oldest first.
    """
    quotes = quotes[-KEEP_QUOTES:]
    mids = [mid for _, mid in quotes]
    if len(quotes) < MIN_QUOTES or quotes[-1][0]-quotes[0][0] < MIN_SPAN or max(mids)-min(mids) > MAX_RANGE:
        return None
    mid = median(mids)
    side, start = ("yes", mid) if mid > 50 else ("no", 100-mid)
    return (side, start) if start >= floor else None


def prints(trades, side, limit, after):
    """What the trade record says about a resting bid at `limit` on `side`, for trades later than `after`.

    A print strictly below the bid means someone sold through it, so the bid
    was filled. A print at the bid is a "touch": queue position would decide.
    """
    out = dict(filled_at=None, touched_at=None, size_through=0.0, size_at=0.0, latest=after)
    for t in sorted(trades, key=lambda t: t["created_time"]):
        when = datetime.fromisoformat(t["created_time"].replace("Z", "+00:00")).timestamp()
        if when <= after:
            continue
        out["latest"] = when
        price, size = round(100*float(t[f"{side}_price_dollars"])), float(t["count_fp"])
        if price < limit:
            out["size_through"] += size
            out["filled_at"] = out["filled_at"] or when
        elif price == limit:
            out["size_at"] += size
            out["touched_at"] = out["touched_at"] or when
    return out


def through(trades, side, limit, after, until):
    """Contracts that printed strictly below `limit` on `side`, later than `after` and no later than `until`."""
    total = 0.0
    for t in trades:
        when = datetime.fromisoformat(t["created_time"].replace("Z", "+00:00")).timestamp()
        if after < when <= until and round(100*float(t[f"{side}_price_dollars"])) < limit:
            total += float(t["count_fp"])
    return total


def nba_leader(event, ticker, state):
    """The side ("yes" or "no") leading in the 4th quarter or overtime, else None.

    Event tickers end with the away code then the home code; the market's own
    code is the last part of its ticker.
    """
    try:
        teams, code = event.rsplit("-", 1)[1][7:], ticker.rsplit("-", 1)[1]
        period, home, away = int(state.get("period") or 0), int(state["home_points"]), int(state["away_points"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    yes_home, yes_away = teams.endswith(code), teams.startswith(code)
    if state.get("status") != "inprogress" or period < 4 or home == away or yes_home == yes_away:
        return None
    return "yes" if (home > away) == yes_home else "no"


def taker_fee(price):
    """Kalshi's standard taker fee per contract, in cents."""
    return 7*price/100*(1-price/100)


def net(price, payout, taker):
    """Profit per contract in cents; a resting fill is costed with no fee."""
    return 100*payout-price-(taker_fee(price) if taker else 0)
