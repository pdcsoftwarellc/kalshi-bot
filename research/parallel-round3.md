# Third batch of tennis trading hypotheses

We used two separate live capture windows for the bookmaker experiments and historical replays for the other three rules. Each historical period starts with $25 in simulated cash. The $80 monthly tennis subscription is shared. Historical trade evidence and sampled displayed depth are proxies for fills.

| Test | Fixed decision | What would support the hypothesis |
| --- | --- | --- |
| Live odds revision | On a reported stable score, buy the side whose margin-adjusted win probability moved at least 3 percentage points between consecutive active odds polls and still exceeds the sized Kalshi ask plus fees by 5¢. | A fill at the limit after a two-second delay and positive proceeds after selling at the bid 30 seconds later, replicated across matches. |
| Deciding-set pricing | After both set settlements establish 1–1, compare match and deciding-outcome books. Try one complementary two-leg basket per event only at an all-in price of 98¢ or less. | The discount repeats, evidence supports both legs, and settlements cover fees and any one-leg losses. |
| First-set point dominance | Fit a single nonnegative adjustment to the live market log odds using the difference in smoothed service-point win fractions. Fit on October alone, apply the coefficient unchanged in November. | Lower forward forecast error than the market, and a tradable after-fee gap of at least 5¢. |
| Cheaper player contract | For the original native-model order intents, buy YES on the chosen player or NO on the opposing player using the cheaper verified route. Break price ties in favor of the original. | Better net P&L on the identical decisions using each route's own quote, print evidence, and actual settlement. |
| Odds change with a point | In a new recording, require a single ordinary point toward the side of a ≥3-point odds revision and a fresh 5¢ net gap. Check score, server, identity and sized quotes. | Positive after-fee cash-outs on independent matches, with enough entries to make the $80 bill economically plausible. |

## Historical findings

* **Set strength.** The positive serve-dominance coefficient fitted to 319 October matches was zero. On 89 later November matches the resulting forecast equaled the market midpoint. The rule made zero trades. In the preselected 7–5 or 7–6 subgroup there were 102 October and 34 November matches. Point histories lack individual receipt timestamps. The public set-settlement marker can lag actual play.
* **Routing.** In October, November 1–15 and September 22–30, the frozen native model produced 27 qualifying order intents. Both books were available on all 27. The cheaper route was selected three times, all in October. Each arm recorded nine simulated fills. Net profit was −$0.40 with the original routing and −$0.38 when routed by cost. Routing preserved the player exposure for normally completed matches. Final payouts are taken from the purchased contract, including any exceptional grading.
* **Decider pricing.** There were 230 confirmed one-set ties. Among 214 verified best-of-three cases with open match contracts, no dedicated third-set-winner pair was listed. The separately identified exact 2–1 score contract offered a comparison on 212. Minimum two-leg costs with cent-rounded fees were $1.01 in September, $1.01 in October and $0.99 in November. Zero met the 98¢ threshold. Retirement and fair-value rules can cause the legs to settle differently. [Set winner rules](https://assets.kalshi.com/contract_terms/TENNISSETWINNER.pdf) · [Set score rules](https://assets.kalshi.com/contract_terms/TENNISSETSCORE.pdf)

**Fee sensitivity and actual payout.** The sole quote below $1.00 was the November 3 Rinderknech–Cerúndolo match. CER match YES cost 1¢, and RIN to win 2–1 YES cost 96¢. The initial whole-cent fee screen put these at 99¢ all in. We then applied direct-member rounding to all candidate legs and repeated selection at the same 98¢ cutoff. One basket qualified at 97.34¢. Both contracts had a qualifying subsequent trade. The provider recorded a retirement at one set apiece with an unfinished deciding set. The exchange paid $0 on the match leg and $0.89 on the exact-score leg. Net simulated loss was $0.0834. There were no other qualifying baskets in that fee scenario. This sensitivity was investigated after observing the near miss. User-specific fee treatment is unverified. [Fee precision](https://docs.kalshi.com/getting_started/fee_rounding) · [Kalshi fee schedule](https://kalshi.com/fee-schedule)

## Live findings

The first four-minute recording contained 192 usable odds observations across four Challenger matches. Of 26 price changes, only one kept the same reported tennis context. It was 1.72 percentage points and did not meet the 3-point threshold. There were no paper entries. The second live rule was specified after this observation and uses a separate recording. Both rules use synchronized local receipt clocks. A vendor update time alone cannot establish that a bookmaker led the exchange.

The fresh four-minute point-confirmed recording observed four live matches. Three had usable matched scores and odds. Fourteen legal single-point transitions included five odds moves of at least three percentage points toward the point winner, across two fixtures. All five failed the five-cent entry screen. The largest indicated gap after the entry fee was 1.91¢. There were no entries, exits or open positions. An earlier interrupted instrumentation tape lacks the added score corroboration and is excluded.

## Reproduction

The public sources and local cache requirements are described in [the research README](README.md). Run the scripts from the repository root. Detailed outputs are under `data/processed/parallel-round3/`, including the per-event ledgers, tapes and fee assumptions.

```sh
.venv/bin/python research/set_residual.py
.venv/bin/python research/decider_basis.py
.venv/bin/python research/decider_basis.py --direct-fees
.venv/bin/python research/player_routing.py
.venv/bin/python research/live_odds_revision.py --replay data/processed/parallel-round3/live_odds_revision/1794955056688402000.jsonl
.venv/bin/python research/live_odds_revision.py --replay data/processed/parallel-round3/live_point_revision/1794955627102953000.jsonl
```
