# Second round of tennis strategy research

We tested five rules with fixed thresholds and one contract per signal. October 2026 was the screening period for the retrospective trading simulations. November 1–15 was a separate comparison. We also used September 19–30 for a historical check on three rules. Previous research has examined some of these markets. Selecting a rule from this batch would require a fresh forward test.

## Historical trading results

Trading profit in dollars includes simulated trading fees. The $80 monthly data subscription is shared across the project. Each strategy had a separate $25 cash simulation.

| Rule | Sep 19–30 | October | Nov 1–15 | Total | Exposure observed |
| --- | ---: | ---: | ---: | ---: | --- |
| Buy a WTA favorite quoted at 80–90¢ | −0.87 | −4.14 | +0.48 | −4.53 | 28 wins from 38 print-supported fills |
| Follow one-sided ATP taker volume | −0.10 | −2.53 | −0.21 | −2.84 | 57 positions, including one carried to settlement |
| Rest a bid and try to resell at the original ask | 0.00 | 0.00 | 0.00 | 0.00 | Four attempted entries from 454 fresh quotes, zero qualifying fills |

**WTA favorites.** Thirty minutes before the earlier published match start, buy one favorite at its ask with a spread of at most three cents. Hold to settlement. Winning contracts paid $28 in total and entries with fees cost $32.53. On this price mix the break-even win rate was 85.6%, compared with 73.7% observed. The positive November result came from three wins in three fills. Paying one extra cent per entry would have brought the overall result to −$4.87.

**Taker volume.** Ten minutes after the scheduled ATP start, count non-block prints in the preceding five minutes. Require at least five distinct prints, 100 contracts in total, and one taker side accounting for at least 80% of volume. Buy that side at its ask with a spread of at most three cents. Attempt to sell ten minutes after the fill. Refresh unsuccessful sell limits each minute for up to 15 minutes and carry remaining inventory to the exchange settlement. The 57 closed positions lost $0.80 gross and incurred $2.04 in fees. Paying an extra cent per simulated exchange fill would have brought the loss to $3.97. Market and tennis schedules had to agree within an hour. In October this excluded 262 of 332 linked matches. Historical scheduled time can differ from actual first serve.

**Passive spread.** Thirty minutes before play, look for a three- to six-cent spread on the canonical player contract. To count a resting bid fill, require a later sell-side public print at least a cent through the bid with at least one contract. Attempt a passive sale at the original ask after verifying the order would still be passive. A missed exit carries through settlement. A control buying at the ask on the same opportunities also recorded zero fills. These observations measure the scarcity of qualifying entries at this observation time.

## Order-book pressure and tennis scores

**Depth imbalance.** On the ATP tour, require 80% of quantity on one side within two cents of each best bid, a spread of at most two cents, and three seconds of continuous pressure. Buy the strong side after two seconds of latency and attempt an exit at the bid after a 30-second hold plus two seconds of routing time. The earlier recordings generated one simulated round trip on Zverev–Shelton, buying NO at 67¢ and selling at 66¢. With four cents in fees it lost five cents. Another 180-second recording of the same match contained 10,519 book changes and zero qualifying signals. There was one independent active match across these ATP captures. A separate 150-second Challenger capture supplied four additional active matches. Applying the same frozen rule there produced four round trips, none profitable, with an eight-cent gross loss and 14 cents in fees. Including that transfer test, five independent matches produced a 27-cent simulated net loss.

**Defended break point.** For a favorite priced at 70% or higher, label the first first-set service game in which the favorite faces break point. Separate a defended hold from a conceded break. Then test whether an observed hold is followed by an executable entry under the earlier ten-cent dip, two-cent recovery and eight-cent exit rule. The final match outcomes in the identifiable ATP and sampled Challenger cohort were:

| Period | Favorite won after holding | Favorite won after conceding |
| --- | ---: | ---: |
| October | 37 / 52 (71.2%) | 28 / 41 (68.3%) |
| November 1–15 | 14 / 16 (87.5%) | 11 / 18 (61.1%) |

That comparison describes outcomes after two different score states. It cannot establish what an order would have cost when the bot learned the result of the game. An eligible forward trade requires a receipt timestamp, a fully observed pressure game and matching executable book evidence on both legs. The historical files have no receipt time for the service-game confirmation. Five forward recordings captured five distinct active ATP or Challenger fixtures. Two strong Challenger favorites were already playing when the recorder started. Neither could establish the complete first pressure-game history required for entry. There were zero eligible starts, fills or open positions under this rule.

## Evidence and reproduction

The aggregate trade studies use subsequent same-side, non-block trades as execution evidence. A public print can support a price and size while leaving queue position and actual fill unresolved. The depth and score studies use local timestamped market observations. Simulated taker costs use the general quadratic multiplier of one rounded up to cents. The passive study records the series maker fee and a taker-fee stress case. [Kalshi trade documentation](https://docs.kalshi.com/api-reference/market/get-trades) · [Kalshi fee schedule](https://kalshi.com/fee-schedule)

Each script writes its summary and underlying ledgers to its subdirectory in `data/processed/parallel-round2/`. Run from the project root:

```sh
.venv/bin/python research/wta_tail.py
.venv/bin/python research/trade_burst.py
.venv/bin/python research/passive_spread.py
.venv/bin/python research/depth_pressure.py --forward data/raw/first-break/1794947237589196000.jsonl
.venv/bin/python research/depth_pressure.py --forward data/raw/first-break-challenger/1794948159745656000.jsonl --output-dir data/processed/parallel-round2/depth_pressure/challenger
.venv/bin/python research/break_hold.py
```
