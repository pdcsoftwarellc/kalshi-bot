# Four tennis market experiments

These exploratory replays used October and November 1–15, 2026. Earlier research used these periods too. November is a comparison period, and a new forward sample would be needed to confirm a selected rule. Each study used one contract per leg and $25 of simulated cash. The API subscription costs $80 per month across the project.

| Hypothesis and rule | Observations | October net | Nov 1–15 net |
| --- | --- | ---: | ---: |
| Two linked player contracts sometimes offer enough of a discount to buy both YES legs. Enter at a combined cost of 98¢ or less including fees. | 80 ATP pairs and 29,236 aligned one-minute closes. Lowest costs were $1.01 and $1.02 respectively. | $0.00 | $0.00 |
| A 5¢ move from two hours to 30 minutes before a scheduled match continues. Buy the direction of the move when both spreads are at most 4¢. | 322 eligible ATP matches. Largest observed move between these two endpoints was 3¢. | $0.00 | $0.00 |
| The market underprices fatigue after a deciding-set match the previous day. Buy the opponent with no prior-day singles appearance, 30 minutes before the earlier listed start. | 20 quoted signals, 12 simulated fills. Prior completion verified against timestamped exchange settlements. | −$0.08 | −$0.27 |
| A strong Challenger favorite that dips and starts to recover tends to rebound enough for an early sale. | 223 matched events from a fixed one-in-eight hash sample, 57 signals, 38 simulated round trips. | −$0.70 | −$0.96 |

Net figures deduct simulated exchange fees. Subscription overhead is shared. Each study has its own coverage and cash ledger. Mutually exclusive player contracts can have special settlement terms on a cancellation. The paired-price study uses the actual leg payouts. Minute closes cannot resolve a fleeting discount between updates.

## The Rocha recording

The November 17 Montevideo match between Henrique Rocha and Federico Aguilar Cardozo was selected by the user and excluded from the October/November comparison. The replay stops at 1:51:08 p.m. Chicago time. The quote at 12:20 p.m., the earliest minute in the five-minute window ending at 12:25, showed a 90¢ bid and a 91¢ ask for Rocha. API-Tennis listed the match at 12:35 p.m. The replay uses the provider's scheduled time, which may differ from actual first serve.

* At 12:56 p.m., the first eligible rebound offered Rocha YES at 87¢ after the midpoint dropped from 92.5¢ to 82¢ and recovered by at least 2¢. An 11.39-contract YES taker print at 87¢ followed at 12:56:04.
* The 95¢ exit limit had supporting sell-side volume at 1:15:27 p.m. A NO taker traded 34.54 contracts at a YES-equivalent price of 96¢. The simulation credited the 95¢ limit.
* The modeled gain was 8¢ gross and 6¢ net on one contract. The chart showed another dip around 1:05 p.m. while the simulated position was open.

The inclusive five-minute quote boundary mattered for Rocha. A first pass inadvertently used 299 seconds and excluded his quote. That stricter pass also lost money on the broader sample (−$0.58 in October and −$0.96 in November). The corrected window was applied to every match.

## Rebound protocol

Select one favorite per matched ATP Challenger fixture with a valid midpoint of at least 70¢ and spread of at most 4¢ inside the five minutes ending ten minutes before the API-Tennis scheduled start. Historical fixtures can reflect schedule revisions. Screen the first three hours following that scheduled start. Record a drawdown of at least 10¢ from a high seen in the preceding 30 minutes. When the midpoint rises 2¢ from its running low and remains at least 5¢ below the high, try to buy one contract at the ask (at most 91¢, spread at most 4¢). Gaps longer than five minutes restart the pattern. Take at most one entry signal per fixture.

Use the first subsequent minute bid that is at least 8¢ above entry, 8¢ below entry, or at least 30 minutes after entry to initiate a sale. Each limit needs a qualifying public trade at or through the limit in the 60 seconds following a two-second delay. The trade must be on the correct taker side, at least one contract, and outside the block-trade facility. Charge the limit and a general quadratic taker fee rounded up to the cent on each fill. If an exit fails, reprice at later bids through the fourth hour, then account for the published settlement. The clip can end with an open position. The price and quantity of a historical trade provide evidence of liquidity. Actual queue position, routing and fills remain uncertain. [Kalshi candlesticks](https://docs.kalshi.com/api-reference/market/get-market-candlesticks) · [Fee schedule](https://kalshi.com/fee-schedule)

Across the broader sample, 19 of 38 round trips made money. Gross trading P&L was −$0.42 and entry/exit fees cost $1.24. The loss after fees was $1.66. Both periods fell short of the API cost.

## Reproduce

Run these from the project root. The scripts read the project caches and save period summaries and trade ledgers under `data/processed/parallel-research/`.

```sh
.venv/bin/python research/paired_prices.py
.venv/bin/python research/price_drift.py
.venv/bin/python research/short_rest.py
.venv/bin/python research/favorite_rebound.py
.venv/bin/python research/favorite_rebound.py --clip-only
```
