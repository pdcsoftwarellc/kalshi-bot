# Fourth round: late favourites, limit orders and the search for a year-round edge

November 17–18, 2026. This round started from a tennis strategy proposal, moved to Kalshi-only data when the tennis subscription was dropped, and ended with about thirty hypotheses tested across sports, weather, non-sports categories and short-dated markets. Nothing here has been traded. Every figure is a simulation on historical or recorded data.

## Summary

Two candidates passed their tests. One apparent pass was withdrawn after a data-timing flaw was found. Everything else failed.

| Candidate | Rule | Evidence | Status |
| --- | --- | --- | --- |
| Dip bids on heavy tennis favourites | Pre-match favourite at 90¢ or more; rest a bid at 80¢; hold to settlement | Holdout of 15,515 matches: 279 fills, 85.6% won against 80% needed, +5.6¢ per contract (95% interval +1.0 to +9.5). Positive in both halves and at every bid level. WTA and Challenger carry it; ATP main tour was negative in both samples. | Passed. Fills and size unconfirmed live. |
| NBA late favourites | 4th quarter or overtime; buy the leader when its ask is 97–99¢; hold to settlement | 899 games: +0.78¢ on the earlier 60%, +1.38¢ on the later 40% (interval +0.40 to +2.04). Held on a separate January–September price-only holdout. Trade tape shows the size was there. | Passed. Not yet paper traded. |
| MLB lead and outs | Lead of 3+ runs with at least as many runs ahead as the trailing side has outs left | +1.32¢ on later games as first measured, falling to +0.38¢ when the game state is delayed three minutes | Withdrawn. Mostly a timing artefact. |

The practical conclusion is narrower than the round set out to find. Kalshi prices publicly available information quickly and accurately. The two survivors are both cases where the price of a specific contest situation looks slightly wrong, and neither is a speed or information advantage.

## Rules of evidence used throughout

* Entries pay the next minute's opening ask after the signal. A resting order counts as filled only when a trade printed strictly through its price in a later minute.
* Taker fee 0.07·p·(1−p) per contract. Resting fills are costed at zero with the full taker fee shown as the worst case.
* One entry per event per rule. Positions are held to the exchange settlement value.
* Games are ordered by date. Rules are chosen on the earliest 60% and scored once on the latest 40%. Where a rule was fixed in advance, older history serves as a holdout.
* Nothing known only after the event (such as minutes until settlement) may select a trade. Selecting by time to settlement made late tennis entries look perfect (1,305 wins from 1,305) and is an illusion.

## Dip bids on heavy favourites

`research/favorite_dip.py`. The pre-match price is the median midpoint of the first 15 one-minute bars in the six hours before settlement, which must be stable to within 3¢. A bid rests at a fixed level at least 8¢ below it.

Discovery sample, September 18 to November 16 (3,989 matches, 2,839 with a clean pre-match favourite at 70¢ or more):

| Pre-match favourite | Bid | Orders | Filled | Won | Needed | Net per filled contract |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 90¢+ | 80¢ | 256 | 74 | 86.5% | 80% | +6.5¢ |
| 90¢+ | 70¢ | 256 | 50 | 80.0% | 70% | +10.0¢ |
| 80–90¢ | 70¢ | 420 | 208 | 70.7% | 70% | +0.7¢ |
| 70–80¢ | 60¢ | 788 | 447 | 55.0% | 60% | −5.0¢ |

Holdout, January 1 to September 17 (15,515 matches, 10,654 clean favourites, 854 at 90¢+):

| Bid | Filled | Won | Needed | Net | 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| 80¢ | 279 (33%) | 85.6% | 80% | +5.6¢ | +1.0 to +9.5 |
| 75¢ | 212 (25%) | 81.0% | 75% | +6.0¢ | +0.9 to +11.1 |
| 70¢ | 163 (19%) | 75.3% | 70% | +5.3¢ | −1.4 to +11.4 |
| 60¢ | 126 (15%) | 68.2% | 60% | +8.2¢ | −0.5 to +16.1 |

By tour at the 80¢ bid: WTA 75 fills, +8.0¢ (0.0 to +14.7); Challenger 156 fills, +7.1¢ (+1.8 to +12.2); ATP 48 fills, −3.0¢. The autumn sample ordered the tours the same way (+10.0¢, +7.2¢, −8.6¢). The tour split was examined after the result.

Supporting observations:

* Heavy favourites are slightly underpriced before play. On the men's holdout, 90–95¢ favourites (454 matches, average 92.4¢) won 93.9%, and 95¢+ favourites (202, average 96.3¢) won 98.0%. The autumn sample showed the same (94.5% at 92.5¢, 100% at 96.4¢). Below 90¢ tennis is priced almost exactly.
* Fills were not run over. A minute after a fill the favourite's midpoint was 1.0¢ above the bid on average.
* The effect is specific to the heaviest favourites. Across about 14,000 matches, take-profit orders on underdogs, moderate favourites and even matches filled exactly as often as a fair random walk predicts (`research/autosell.py`).
* The same direction appears in older NBA games (14 fills at an 80¢ bid, +12.9¢), too few to count.

Expected activity is about 33 fills a month across the three tours, or about 27 on WTA and Challenger alone. A filled trade loses outright roughly one time in seven.

Unproven: the fill model uses one-minute bars and knows neither queue position nor the size traded at the bid; displayed size during a dip has not been observed; the pre-match price is inferred from a stable quote and not from a known start time.

## NBA late favourites

Price alone (`research/near_end.py`), buying once a side's ask reaches a threshold:

| Sample | Entry | Trades | Losses | Net | 95% interval |
| --- | --- | ---: | ---: | ---: | --- |
| Sep 18 – Nov 16 | ≥98¢ | 218 | 0 | +1.62¢ | — |
| Jan 1 – Sep 17 holdout | ≥97¢ | 747 | 9 | +1.10¢ | +0.25 to +1.78 |
| Jan 1 – Sep 17 holdout | ≥98¢ | 724 | 6 | +0.81¢ | +0.13 to +1.38 |

With play-by-play joined (`research/game_state.py`, `research/rule_search.py`; 899 games, 539 to choose and 360 to score):

* The plain rule earned +0.78¢ (488 trades, 6 losses) then +1.38¢ (317 trades, 2 losses). It was positive in six of eight months.
* The highest-profit rule out of 7,700 tried made +1.65¢ on the games it was chosen from and −0.09¢ afterwards. The ten best picks averaged +0.12¢ later. The same collapse occurred in every sport searched.
* Game state alone, at any price, earns nothing. The price already contains lead and clock.
* Being near the end is not what makes an entry safe. Entries with under three minutes left earned +0.09¢, and five of nine losing trades were leads of 2–5 points with 25 seconds or less to play. Requiring a 10-point lead cut losses from 9 in 806 to 3 in 537 (+1.48¢). That screen was found after looking at the losers.
* Delaying the game state by two minutes leaves the result unchanged (+0.72¢, +1.31¢), so it does not depend on the play-by-play timing flaw described below.

Fills (`research/nba_fills.py`, all 805 entries against the public trade tape):

* Contracts taken at our price or better within three minutes: median 21,525 (earlier) and 27,475 (later). 100 contracts were available in about 99% of entries and within ten seconds in 79–84%; 1,000 in about 94–95%.
* Using the price actually paid for the first 100 contracts, the result is +0.91¢ and +1.43¢.
* A flat 1¢ of slippage gives −0.15¢ and +0.44¢. The edge is thin enough that this matters; the tape says it is rare.
* A bid resting 1¢ below the ask for three minutes filled 61% and 56% of the time and earned +1.58¢ and +2.70¢ per filled contract (+1.36¢ and +2.48¢ at the full taker fee). Per signal that is +0.96¢ and +1.51¢, against +0.78¢ and +1.38¢ paying the ask. The advantage rests on six losses among filled trades.

## The withdrawn MLB result and the timing flaw

The play-by-play archive stamps some rows with a score that has not yet happened. In baseball the first pitch of an at-bat already shows that at-bat's runs; in basketball a substitution row between free throws shows the made free throw. `research/decided.py` found this while explaining an impossible loss. The cached rows under `data/raw/game-state/` carry the lookahead.

Re-scoring the MLB rule with the state delayed:

| Delay | Earlier games | Later games |
| --- | --- | --- |
| 15 s | +0.88¢ | +1.32¢ (+0.24 to +2.19) |
| 90 s | +0.37¢ | +0.81¢ (−0.31 to +1.70) |
| 180 s | +0.05¢ | +0.38¢ |
| 300 s | −0.10¢ | +0.52¢ |

A long delay also discards timely entries, so the truth is probably slightly above zero. There is no longer evidence of an edge. Any future rule that uses score from this archive should credit points only at the scoring play's own stamp, as `research/decided.py` does.

## What failed

| Idea | Result | Code |
| --- | --- | --- |
| 15-40 tennis strategy | Break before deuce 61.3% of 5,205 games, exactly what ordinary serving predicts (servers won 62.3% of pressure points against 62.8% otherwise). Fair-value round trip −4 to −6¢. Live: 18 paper trades, −4.2¢ each. | `fifteen_forty.py`, `fifteen_forty_tape.py` |
| Bookmaker consensus against Kalshi | 2,749 matches. Kalshi and Pinnacle differ by 1.0–1.5¢, about the spread. Buying gaps: +1.2¢ (−2.6 to +5.1) on 411 trades. | `consensus_history.py` |
| Late rule on price alone, other sports | Tennis −1.87¢ at 98¢; NFL −1.58¢ later; NHL −0.67¢ and +0.02¢; MLB −0.61¢ on 1,672 holdout trades; college basketball −0.20¢ (−1.51 to +0.84) on 679 later trades. | `near_end.py`, `rule_search.py`, `college_basketball.py` |
| Seventeen other sports families | None passes. European soccer looked clean on three weeks (3 losses in 211) and lost in both halves of 730 older league games. Table tennis, eBasketball and most esports have empty books. | `sports_survey.py` |
| NBA and MLB spreads and totals | Plain rule −0.8¢ to −4.2¢ later. Every loss came in a game the winner bet won. | `spreads_totals.py` |
| Already-decided markets | About 1% are still offered below 99¢ and the offer is gone within a minute. Live: 382 decided markets, no offers, resting 99¢ bids 13,000–108,000 deep. All takers combined earn $155–835 a week. | `decided.py`, `decided_live.py` |
| Linked prices that must add up | 8 live scans of about 25,000 markets, 5,328 order books: no basket profitable after fees. 13 one-minute historical blips, none surviving a one-contract fee. | `consistency.py`, `consistency_history.py` |
| Near-certain contracts outside sports | 90–99¢ contracts days before close: +1.61¢ earlier, −4.80¢ later (−9.6 to −0.5). Resting bids were adversely selected: filled orders lost 6.4% of the time against 3.3% unfilled. | `favorites.py` |
| Underdog and favourite autosell, two-sided dip bids | Negative in every bucket and sport. A sell at twice the entry price filled 45–50% of the time, as a fair market predicts. | `autosell.py` |
| Weather after the high is set | None of 54 rules passed. When the observed-peak range was right there was nothing offered (187 of 270 days); when it was cheap it lost every time (38 of 38). | `weather_state.py` |
| Kalshi's score feed against its order book | The book finishes repricing a tennis point 5–10 seconds before the feed shows it. Buying at feed time lost 3.2¢ on 104 points. | `feed_vs_book.py` |
| Short-dated crypto, index and hourly markets | Plain rule about zero on Bitcoin 15-minute, negative on hourly crypto and S&P. A distance-to-strike screen was unproven. Dropped at the owner's request. | `short_dated.py`, `crypto_forward.py` |

A third-party backtest page for the Bitcoin 15-minute series claimed +$145,159 in 30 days from 14 rules. It showed one in-sample window with no later results. Its two fully specified rules (buy the favourite at 88¢+ in the last seven minutes) lost 0.5–0.8¢ per contract on our data, inside and outside its window.

## Findings about the data and the venue

* Kalshi publishes live game state for its sports events through `/milestones` and `/live_data`. For tennis it delivered points a median of 10 seconds before the paid feed (18 of 19 points). Finished games keep only their final state, so state-based rules need a recording or an outside archive.
* The `/markets` listing returns a cached quote that can lag the order book by six seconds or more. Use `/markets/{ticker}/orderbook` for prices. `research/live_games.py` reads the listing and should be treated as a score recorder only.
* Fees differ by series. NBA, NHL, ATP and WTA charge fees on resting orders; ATP Challenger and daily temperature do not. MLB's taker multiplier is 0.5. The maker rate was not confirmed against the fee schedule.
* Markets settled before September 18, 2026 are served only from the `/historical/` endpoints, with slightly different field names.
* The public API is rate-limited to a few requests a second in practice. Several studies cover less history than planned for that reason, and say so in their outputs.

## What to do next

1. Build a paper trader on the order-book endpoint that runs both candidates: an 80¢ resting bid on WTA and Challenger favourites priced 90¢+ before play, and the NBA rule in two versions (paying the ask, and a bid 1¢ below). Log displayed size at every entry.
2. Run it for at least two weeks. The dip rule needs roughly 30 fills and the NBA rule 40 before the live win rates say anything.
3. Fund a small pilot only if the paper results match. At 100 contracts per trade the simulations imply roughly $200 a month from the dip rule and $150–200 from NBA in season, with single losses of about $80 and $97.

## Reproduction

Run from the repository root. Raw downloads are cached under `data/raw/`; outputs are under `data/processed/`.

```sh
.venv/bin/python research/favorite_dip.py --groups tennis
.venv/bin/python research/favorite_dip.py --groups tennis --archived
.venv/bin/python research/near_end.py
.venv/bin/python research/near_end.py --archived --groups mlb nba --start 2026-01-01 --end 2026-09-18 --name holdout
.venv/bin/python research/game_state.py
.venv/bin/python research/rule_search.py
.venv/bin/python research/nba_fills.py --report-only
.venv/bin/python research/autosell.py
.venv/bin/python research/fifteen_forty.py
.venv/bin/python research/consensus_history.py
.venv/bin/python research/live_games.py --every 15 --hours 24
```

The subagent-written studies document their own commands in their module docstrings: `college_basketball.py`, `sports_survey.py`, `spreads_totals.py`, `decided.py`, `consistency.py`, `favorites.py`, `weather_state.py`, `feed_vs_book.py`, `short_dated.py`.
