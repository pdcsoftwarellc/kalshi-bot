# Testing the monthly API cost

The experiment reports are versioned alongside the code. Raw recordings, API response caches, fitted models, and generated ledgers are local files. Replay examples with dated capture IDs require those local inputs.

Run the historical ATP experiment from the repository root. Results and individual order and fill records are saved under `data/processed/api-cost-research/`. The source requests are cached in `data/raw/cost-research-v1/`. The current cache reproduces the November 17, 2026 snapshot offline. A new download after archival will require routing requests to Kalshi's historical endpoints.

```sh
.venv/bin/python research/api_cost.py --monthly-api-cost 80
.venv/bin/python research/forward_odds.py
```

The historical study estimates an overall Elo rating from ATP singles fixtures starting July 1, 2026. Both players need five eligible matches. Each UTC day's estimates use results dated at least seven days earlier. A trade decision occurs one hour after both markets in a pair have opened, rounded up to the minute. The player identity must match uniquely and both providers' start times must be later than the five-minute order window. The study uses settled events and final fixture metadata, so schedule revisions and missing coverage can affect the sample.

For each pair the alphabetically first ticker represents both possible sides. Entry requires ten cents of estimated edge after fees and a two-sided quote from the preceding five minutes. One contract is simulated at the quoted limit only when an ordinary same-side taker trade of at least one contract prints at or through the limit between two seconds and five minutes later. Public trades and minute quotes cannot establish order-book queue position. The study assumes the general quadratic taker fee at multiplier one, rounded up to the cent. No account-specific fees or rebates are used. [Kalshi market history](https://docs.kalshi.com/getting_started/historical_data) · [Fee schedule](https://kalshi.com/fee-schedule)

Each evaluation period starts with $25 of simulated cash. Orders reserve their full cost until expiry or settlement. Profit is attributed to the entry period using the exchange payout. The second period spans half of November and receives half the monthly subscription cost. The all-win ceiling assumes every signal fills at its limit and pays $1. It measures maximum earnings at the specified size and prices.

The forward command takes one observation of each matched upcoming event. For each bookmaker with prices on both players, it normalizes the implied probabilities and takes the median across at least three books. Only scheduled, non-live fixtures with more than 30 minutes remaining by both sources can qualify. A five-cent estimated net edge triggers a second depth check at least two seconds later. Snapshots include receipt times, source odds, status, both start times and the book. Odds freshness before receipt, common pricing sources, and retirement settlement differences remain potential sources of error. Repeated snapshots of one match are observations of the same event. Run results are local paper records.

## First service break

This study uses the fixtures with a verified identity, settlement and prematch
quote from the ATP cost experiment. Coverage therefore depends on the earlier
study's filters. An early break is the first service break in the first six
ordinary games of set one. A strong favorite has a prematch midpoint of at
least 0.60. The broken player receives in the next game. The adjustment adds a
regularized intercept, game number and game deficit to the log odds of the
prematch midpoint. October fixtures settled before November train the model.
November 1–15 fixtures check prediction error. The historical input lacks
post-break receipt times, so these results measure outcome prediction.

```sh
.venv/bin/python research/break_study.py
.venv/bin/python research/record_break.py --seconds 240
.venv/bin/python research/score_break.py data/raw/first-break/<capture-id>.jsonl
```

The initial reference price is the latest fresh two-sided minute quote one
hour after listing. Sparse books in the first live trial motivated a second
reference time, 30 minutes before the earlier provider or exchange start. A
separate fit uses that reference on historical fixtures from the original
sample. The same November matches can appear in both checks.

The recorder stamps Kalshi snapshots and incremental depth changes, tennis
updates and REST score polls with local UTC and monotonic receipt times. It
tracks the open ATP markets discovered at launch and ends after the specified
duration. Each new recording starts a new baseline. The scorer requires an
observed transition from an unfinished game to a confirmed break on the same
feed, consistent set totals, and a new game at love-all. Historical catch-up
messages cannot trigger a purchase. A sequence gap invalidates the local
order books until replacement snapshots arrive.

Two seconds after receiving the break, the scorer reads the book as it stood
at that instant. It requires a two-sided book updated within five seconds,
one contract of displayed depth at the ask, and at least five cents of modeled
edge after a conservative entry fee. A fill is simulated. The scorer saves a
paper report under `data/processed/first-break/`. Re-running the scorer on
the same tape updates any paper fill with the published Kalshi settlement
when available. Count at most one entry per fixture across recordings. The
capture has no order-submission path.

## Parallel strategy experiments

The [first round](parallel-experiments.md) includes the Rocha cash-out replay. The [second round](parallel-round2.md) covers favorites under break-point pressure, WTA favorite pricing, one-sided aggressive trades, displayed depth, and passive spread capture. Each report identifies its observation window, order assumptions, costs, and supporting ledgers.

## Native trading frequency at confirmed set results

`research/set_frequency.py` replays the frozen native ATP model at the first
minute close on or after both players' set-winner contracts publish complementary
binary settlements. The primary checkpoint is set one. A secondary checkpoint
uses the same settings at a confirmed one-set tie. The market must still be
open. The score prefix must agree with the completed set results. The study
assumes those points have reached the score feed by decision time. A late
exchange settlement can leave this forecast behind play in the following set.

For the October and November 1–15 windows, the replay buys at the whole-cent
limit only when a non-block same-side taker print of at least one contract
appears at that price or better between two and five seconds after the decision.
Public minute quotes and prints cannot establish private queue position or the
exact time a historical score update was received. Cash and open cost share
the $25 budget, with one contract per signal and a hold through match settlement.
At these checkpoints a best-of-three match permits at most two signals. Between-set
opportunities and point-by-point opportunities have different frequencies.

```sh
.venv/bin/python scripts/download_data.py --source api-tennis --start 2026-10-16 --end 2026-11-16 --fixtures-dir data/raw/set-frequency-atp
.venv/bin/python research/set_frequency.py
```

This uses the native-atp fitted artifacts, their original July–October fixture
cache, and recorded trading settings. Per-period ledgers, exclusion counts,
fees, net trading profit, API overhead and the all-signals-win ceiling are in
`data/processed/native-atp/set-frequency/`. [Kalshi candlesticks](https://docs.kalshi.com/api-reference/market/get-market-candlesticks) · [Kalshi fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf)

## Market-informed forecast and point-feed timing check

`research/market_anchor.py` uses the prematch midpoint as the starting YES odds
and multiplies those odds by the ratio of the native model's current odds to
its opening odds. The opening model averages both possible starting servers.
The midpoint must come from a two-sided quote with a spread of four cents or
less, no more than five minutes old, at the reference time 30 minutes before the
earlier recorded provider or exchange start. Historical schedules may have been
revised. Both arms use identical matches, native confidence, set confirmation
times, trading thresholds and cash simulation. September 22–30 adds an earlier
post-training slice. October and November have appeared in previous studies.
The output includes each arm's ledger and the time from the close of the set
market to the decision. This interval measures the research checkpoint's delay.
It does not reveal the timestamp of the point that ended the set.

`research/score_feed_markout.py` asks how much executable price movement
remains after a point receipt on the recorded Challenger feeds. It includes
only single, actionable point updates. Record the point winner's sized ask two
seconds after receipt and their sized bid 30 seconds after receipt, with two
cent-rounded taker fees. Quotes must be two-sided, show at least one whole
contract per side, and be at most five seconds old. These are displayed-price
estimates. Windows can overlap and the report gives per-window and
match-weighted averages. The fresh three-minute capture is reported separately.

```sh
.venv/bin/python research/market_anchor.py
.venv/bin/python research/score_feed_markout.py data/raw/first-break-challenger/1794948159745656000.jsonl data/raw/first-break-challenger/1794952034312915000.jsonl --fresh data/raw/first-break-challenger/1794954154690820000.jsonl
```

Outputs live under `data/processed/native-atp/market-anchor/` and
`data/processed/native-atp/score-feed-markout/`.

The [third batch](parallel-round3.md) tests bookmaker revisions, deciding-set prices, first-set point dominance, routing between player contracts, and a separately recorded point-confirmed odds-revision rule.
