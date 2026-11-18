# Kalshi Tennis Trading Bot

A live tennis trading bot that predicts match win probabilities in real time using a point-by-point Bayesian model and trades (or paper trades) on Kalshi prediction markets.

The research workflow can download ATP singles history from API-Tennis, train a
point model, and score held-out matches. Trading replay additionally requires
timestamped predictions, Kalshi quotes with depth, and exchange settlements.
The [research guide](research/README.md) links the experiments, results, and replay commands.

---

## How it works

### Prediction pipeline

Each live point from the API-Tennis feed triggers the following chain:

```
API-Tennis WebSocket
    → TennisFeed          parse point event
    → MatchTracker        GBDT + BayesianUpdater → win probability
    → MarketFinder        fetch real Kalshi bid/ask prices
    → evaluate_signal     edge + quarter-Kelly sizing
    → RiskManager         exposure cap
    → TradingGuard        stop-loss / pause
    → TradingBot         simulate paper fill, log to DB
    → BroadcastHub        push update to web portal
```

### Model

The model is a three-layer stack:

| Layer | What it does | File |
|-------|-------------|------|
| **GBDT (XGBoost)** | Predicts P(server wins point) from pre-match + in-play features | `tennis/model.py` |
| **BayesianUpdater** | Beta-Binomial conjugate update of p_serve after each observed point | `tennis/updater.py` |
| **Markov chain** | Propagates p_serve analytically through game → set → match (Newton & Keller 2005) | `tennis/markov.py` |
| **Calibrator** | Symmetric odds calibration fitted on validation match states | `tennis/calibration.py` |

At match start the GBDT estimates a separate service prior for each player. Each observed point updates only the player who served it. Paired posterior draws propagate uncertainty through the actual score and service order.

### Features

Pre-match features (computed once per match):
- Surface-specific Elo with an all-surface fallback when history is missing
- Rolling serve/return win % (last 52 weeks)
- Head-to-head record
- Tournament level, surface, ranking difference

In-play features (updated each point):
- Running serve points won %
- Return points won %
- Momentum over last 5 and 10 points
- Set/game score, tiebreak flag, final-set flag

### Signal evaluation

`integration/edge.py` — four sequential gates before a trade is placed:

1. **Confidence** — Bayesian CI width narrow enough (`MIN_CONFIDENCE`, default 0.60)
2. **Points observed** — at least `MIN_POINTS_OBSERVED` (default 10) played
3. **Edge** — `model_prob − market_ask > MIN_EDGE_PCT` (default 0.05) on YES, or equivalent on NO
4. **Net dollar edge** — edge in dollars after the bot’s assumed $0.07/contract entry fee > `NET_EDGE_MIN_DOLLARS` (default $0.10)

Position sizing: **quarter-Kelly**
```
cost_per_contract = price_in_dollars + entry_fee
kelly = (edge - entry_fee) / (1 - cost_per_contract)
contracts = floor(kelly / 4 × bankroll / cost_per_contract)
```
Capped at `MAX_CONTRACTS_PER_SIGNAL` (default 20).

---

## Paper trading

By default (`python bot.py`) the bot simulates fills in `TradingBot._execute_signal`:

- **Prices** are fetched from the **production Kalshi API** with no authentication — `/markets` is publicly readable, giving real live bid/ask spreads.
- **Orders** are never sent. The bot updates in-memory risk/position tracking directly.
- **All simulated fills are persisted** to `DB_PATH` with `mode="paper"` and are visible in the web portal.
- Restarts reset in-memory exposure and positions while preserving DB history.
- Paper fills assume execution at the listed price; they do not model queue priority,
  available liquidity, slippage, or settlement. They are not evidence of realized profit.

### Rule-based paper trader

`paper/` runs the rules that came out of the [fourth research round](research/parallel-round4.md)
and the follow-up studies in `research/dip_*.py`, `research/favourite_gap.py` and
`research/edge_reality.py`. Without `--live` it uses public Kalshi reads only and cannot
place an order.

| Rule | What it does | Real orders with `--live` |
| --- | --- | --- |
| Tennis dip | Favourite at 90¢+ before play; an 80¢ bid from the first live pass, resting on the underdog's market | Qualifying rounds on WTA and ATP Challenger only |
| Tennis dip variants | The same with the bid 10¢ below the pre-match price, and at 65¢ | No |
| Tennis pre-match | Buy a qualifying-round favourite at 90¢+ at the ask before play | No |
| NBA late favourite | Fourth quarter or overtime, leader ahead by 6+, ask 97–99¢: buy at the ask, or bid 1¢ below | Only with `--live-rules tennis_dip nba_take` |
| College basketball dip | The three tennis dip bids on men's college games | No |

Main-draw matches, ATP main tour, WTA Challenger and NBA leads under 6 are simulated and
reported as separate rows.

With `--live`, real orders go to the production account through `paper/broker.py`. New
live orders stop when unsettled exposure would pass `--max-exposure`, when settled losses
reach `--max-loss`, or while a file named `STOP` exists in `data/paper/`. Resting orders
already on the exchange stay there until the match ends or they are cancelled by hand.

```sh
.venv/bin/python -m paper.trader run --hours 24    # simulated; one pass every 20 seconds
.venv/bin/python -m paper.trader run --hours 336 --live --contracts 5 --max-exposure 60 --max-loss 30
.venv/bin/python -m paper.trader report
```

- A resting bid counts as filled only when a trade prints below it; a taker buy is limited
  to the size displayed at the ask. Positions are held to the exchange settlement value.
- State and a journal are kept in `data/paper/`, so a restart resumes. A match that starts
  while the trader is not running is skipped, because its pre-match price was not observed.
- Run only one copy against a state folder.
- `research/live_games.py --every 15 --hours 336` records Kalshi's live scores beside both
  markets' quotes, from 45 minutes before each start, for later score-aware analysis.

---

## Setup

### Requirements

- Python 3.12+
- On macOS, XGBoost and LightGBM require `brew install libomp`.
- `requirements.lock` pins the dependencies verified on Python 3.12 / Apple Silicon.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
```

Without `uv`, use `python3.12 -m venv .venv`, activate it, then run
`python -m pip install -r requirements.lock` and `python -m pip install --no-deps -e .`.

### Environment variables

For a new checkout, copy `.env.example` to `.env` (preserve an existing `.env`)
and run `chmod 600 .env`. Fill in the values locally; do not paste secrets in chat.
Store the private key under the Git-ignored `secrets/` directory and restrict the
key's permissions with `chmod 600 secrets/kalshi-private.pem`.

```bash
# Required for live score feed
API_TENNIS_KEY=your_api_tennis_key

# Required only for live trading (--live --prod)
# Not used in paper mode
KALSHI_API_KEY=your_kalshi_uuid
KALSHI_PRIVATE_KEY_PATH=secrets/kalshi-private.pem

# Evaluation limits from .env.example (code defaults are larger)
MIN_EDGE_PCT=0.05
MIN_CONFIDENCE=0.60
MIN_POINTS_OBSERVED=10
MAX_CONTRACTS_PER_MATCH=5
MAX_TOTAL_EXPOSURE=25
MAX_CONTRACTS_PER_SIGNAL=1
WEB_PORT=8765
```

RSA and Ed25519 PEM keys are supported. API key IDs and private keys must belong
to the same environment; demo execution uses `DEMO_KALSHI_API_KEY` and
`DEMO_KALSHI_PRIVATE_KEY_PATH`. See [Kalshi's key documentation](https://docs.kalshi.com/getting_started/api_keys).

The project-root `.env` loads regardless of the working directory. Existing shell
environment variables take precedence. Relative data/key paths resolve from the
project root. The bot creates its SQLite database on first run. The example
uses `data/paper-trading.db` for paper trades. Local databases are ignored by Git.
The portal can persist threshold overrides in its database, which take precedence
on restart.

Run the read-only checks before starting:

```bash
python scripts/check_env.py
python scripts/check_env.py --online
# Optional: verify production credentials with a GET balance request; no orders
python scripts/check_env.py --online --kalshi-env prod
python -m pytest -q
```

The checker hides key values and balances, returns a nonzero exit code for missing
requirements, and checks installed model libraries. `--online` tests Kalshi public
market data and API-Tennis REST access; WebSocket subscription coverage must still
be verified during paper evaluation.

### Historical research data

The bare `python scripts/download_data.py` command uses the legacy GitHub
repositories, which returned 404. To reproduce the small research run in this
checkout, use the configured API-Tennis subscription from the project root.
The downloader caches each UTC day and resumes from existing files. Use
`--update` to refresh a date range. [API-Tennis fixture documentation](https://api-tennis.com/documentation)

```bash
source .venv/bin/activate
python scripts/download_data.py --source api-tennis --start 2025-06-30 --end 2025-07-05
python scripts/build_features.py --fixtures-dir data/raw/api_tennis --train-before 2025-07-02 --test-from 2025-07-04 --out-dir data/processed/wimbledon-smoke --artifacts-dir data/artifacts/wimbledon-smoke
python scripts/train.py --processed-dir data/processed/wimbledon-smoke --artifacts-dir data/artifacts/wimbledon-smoke
python scripts/backtest.py --test-data data/processed/wimbledon-smoke/features_test.parquet --artifacts-dir data/artifacts/wimbledon-smoke --plots-dir data/processed/wimbledon-smoke/evaluation --latency-samples 8
```

The importer includes finished ATP singles matches with consistent point, game,
set and match scores. It reports exclusions. Missing player history falls back
to tour averages. For unfamiliar tournaments without surface information the
surface features remain unknown. API-Tennis player IDs belong to a different
namespace from ATP IDs. Newly trained artifacts identify their source so the tracker can use provider keys directly. Older artifacts use the ATP name resolver.
Use a longer history and a new, untouched test period for strategy selection.

### Native model research run

The November 2026 native-model experiment uses complete ATP singles fixtures from
July 1 to October 15. Training ends August 31, validation ends September 20,
and the chronological test starts September 22. Set `ARTIFACTS_DIR=data/artifacts/native-atp`
in the shell to select this model for a local paper run or an environment check.

```sh
python scripts/download_data.py --source api-tennis --start 2026-07-01 --end 2026-10-15 --fixtures-dir data/raw/native-atp
python scripts/build_features.py --fixtures-dir data/raw/native-atp --train-before 2026-09-01 --test-from 2026-09-22 --out-dir data/processed/native-atp --artifacts-dir data/artifacts/native-atp
python scripts/train.py --processed-dir data/processed/native-atp --artifacts-dir data/artifacts/native-atp
python scripts/backtest.py --test-data data/processed/native-atp/features_test.parquet --artifacts-dir data/artifacts/native-atp --plots-dir data/processed/native-atp/evaluation --latency-samples 40
python research/model_market_audit.py --artifacts-dir data/artifacts/native-atp --out-dir data/processed/native-atp/market-audit
python research/native_replay.py data/raw/first-break-challenger/*.jsonl
```

The market audit uses the previously cached ATP cohort. The tape replay runs
the native feed, tracker and prediction engine in receipt order and passes their
YES-oriented predictions and sized quotes into the fee-and-fill replay. Each
recording starts a separate cash ledger. `--settle` retrieves published payouts
for forecasted markets. Full native payloads are present in new recordings.
Older enriched Challenger tapes support first-set snapshots with complete point
logs. An empty or unpriced book, a recording gap or an incomplete score history
limits coverage.

### Trading replay

```bash
python scripts/replay_trades.py tests/fixtures/replay.jsonl --fee 0.02
```

The example is a deterministic accounting fixture. Supply JSON Lines sorted by
local receipt time (`ts_ms` in milliseconds). Each event has a Kalshi `ticker`.

| Event kind | Fields |
| --- | --- |
| `quote` | `yes_bid`, `yes_ask` in whole cents, with executable `bid_size`, `ask_size` in whole contracts. A size of zero marks an unavailable side. |
| `prediction` | `model_prob` for YES on this ticker, `confidence`, `points_observed`, computed using only information received by this timestamp. |
| `settlement` | `yes_cents`, the published YES payout per contract. NO pays its complement. |

A prediction reads the last fresh quote. The first eligible later quote after
`--latency-ms` supplies the simulated execution price and size. Orders expire
after `--max-age-ms`. Each ticker can have one pending order. Fills use the
available cash, visible depth, and limit price. This approximation omits queue
position and liquidity changes between observations. The report shows settled
profit, fees, return on settled cost, drawdown at settlements and open cost.

Use `--fee` for a flat cost scenario or `--taker-multiplier` for general quadratic
taker pricing. Choose `--fee-precision 0.0001` or `0.01` for the applicable period
and account. The multiplier and precision must apply throughout the replay.
The signal uses a single-contract fee estimate and the ledger rounds the fee
once per simulated fill. [Kalshi fee schedule](https://kalshi.com/fee-schedule)

### Training

```bash
python scripts/build_features.py   # build parquet feature files
python scripts/train.py            # train GBDT + calibrator
python scripts/audit_model.py      # check for leakage / overfitting
python scripts/backtest.py         # evaluate on held-out test set
python scripts/tune_updater.py     # grid search BayesianUpdater params
```

Artifacts are saved to `data/artifacts/`: `gbdt_model.pkl`, `calibrator.pkl`, `player_elo.parquet`, `player_stats.parquet`, `h2h.parquet`.

---

## Running the bot

```bash
# Paper trading (default) — real prices, simulated fills
python bot.py

```

Web portal starts automatically at **http://127.0.0.1:8765**.

### Live trading readiness

The existing `--live` / `--prod` flags remain available, but this version is not
ready for unattended real-money execution:

- `KalshiClient.place_order()` still uses the removed `/portfolio/orders` write
  endpoint. Migrate to [Create Order V2](https://docs.kalshi.com/api-reference/orders/create-order-v2)
  and validate its request/response handling in demo.
- The execution path counts an accepted order as a complete fill. It must track
  actual fills, partial fills, outstanding orders, and reconciliation after restarts.
- Stop-loss and force-exit controls currently block new entries; they do not
  liquidate existing positions. The configured stop-loss is not a guaranteed loss cap.
- The six-day research sample covers one tournament. Evaluate with realistic
  fees, fresh executable quotes, and settlement accounting before choosing a live budget.

---

## Project structure

```
bot.py                      Main entry point
config.py                   All env-var configuration

tennis/
    model.py                XGBoost point-win GBDT
    updater.py              Beta-Binomial Bayesian updater
    markov.py               Point→game→set→match Markov chain
    calibration.py          Isotonic regression calibrator
    engine.py               PredictionEngine (ties it all together)
    features.py             Feature engineering
    elo.py                  Surface-specific ELO tracker
    types.py                MatchState, PredictionResult dataclasses

integration/
    kalshi_client.py        KalshiClient, HybridKalshiClient
    preflight.py            Local startup validation
    market_finder.py        Resolves live matches to Kalshi market tickers
    tennis_feed.py          API-Tennis WebSocket + REST fallback
    match_tracker.py        Per-match session management
    edge.py                 Edge computation + quarter-Kelly sizing
    risk.py                 Exposure tracking and caps
    player_resolver.py      Player name → ATP ID resolution

web/
    app.py                  FastAPI portal + WebSocket hub
    db.py                   SQLite persistence (trades, signals, match state)
    guard.py                Stop-loss, profit target, force-exit logic
    state.py                Runtime-adjustable thresholds

scripts/
    check_env.py            Read-only dependency, artifact, and connection checks
    build_features.py       Feature dataset construction
    train.py                Model training
    backtest.py             Held-out test evaluation
    audit_model.py          Leakage and overfitting audit
    tune_updater.py         BayesianUpdater hyperparameter search
```

---

## Key design decisions

**Production prices for paper mode.** `KalshiClient` reads public production prices
without credentials while `TradingBot` simulates fills locally. Separate demo
credentials are needed for demo execution.

**No GBDT/updater blending.** An earlier design blended the GBDT estimate and the Bayesian posterior at 20 points. This was removed — the updater already handles the prior-to-data transition cleanly through its Beta-Binomial parameterisation.

**In-play calibration.** The first validation dates select the GBDT stopping point. The later dates fit an odds slope to opening and every tenth pre-point match state. Each match has the same total weight. Player-swap symmetry and a weak penalty towards identity preserve sensible behavior between observed scores. The backtest reports raw, calibrated and score-only errors on later matches.

**Server identity.** Main-tour point histories encode game, set, and tiebreak service boundaries. The feature builder preserves those boundaries and computes each row from the score before its target point.

---

## Academic references

| Paper | Relevance |
|-------|-----------|
| Newton & Keller (2005) — *Probability of winning at tennis I* | Foundation of the Markov chain in `markov.py` |
| Klaassen & Magnus (2001) — *On the Independence and Identical Distribution of Points in Tennis* | Points are not iid; server advantage varies with score pressure |
| Ingram (2019) — *A point-based Bayesian hierarchical model to predict the outcome of tennis matches* | Validates the Beta-Binomial conjugate approach in `updater.py` |
| Sipko & Knottenbelt (2015) — *Machine Learning for Prediction of Professional Tennis Matches* | GBDT approach and feature selection |
