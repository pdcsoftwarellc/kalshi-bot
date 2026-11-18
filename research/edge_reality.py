"""How much of the two surviving edges is likely real? A statistical audit, offline only.

Reads the per-order rows already on disk (nothing is fetched) and writes
data/processed/edge-reality/summary.json. Six steps:

1. Rebuild the tennis dip table (favourite at 90c+ before play, 80c resting bid,
   strict fills) per tour and period. Matches present in both the archived and
   the recent file are kept in the archive only. itf-women-orders.json is picked
   up automatically once it exists.
2. Pooled win rate, and a hierarchical model on the win-rate scale:
   logit(p_tour) ~ Normal(mu, tau), wins ~ Binomial(fills, p_tour), flat prior on
   mu, half-normal(0.5) prior on tau (a uniform prior is run as a sensitivity
   check). Computed on a (mu, tau) grid, no sampler needed. Net cents = 100p-80.
   Heterogeneity: Pearson chi-square with a Monte Carlo p-value, and the Bayes
   factor of "one shared edge" (tau = 0) against the tau prior.
3. Tour selection: simulate equal true edges at these sample sizes, pick the best
   tours after the fact, and measure how far their observed mean sits above truth.
4. Out-of-sample subsets: which fills are clean for which decision.
5. Multiple testing: p-values for each passing result, Bonferroni and
   Benjamini-Hochberg for about 40 ideas, and the posterior chance each is real
   under stated priors. NBA entries are split by whether the game falls inside
   the window the price rule was first noticed on (18 Sep to 16 Nov).
6. Fills needed to tell an edge of a given size from zero.

Fractional settlements (0.5, 0.84, 0.94) count as fractional wins.

    .venv/bin/python research/edge_reality.py
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
DIP = ROOT / "data/processed/favorite-dip"
OUT = ROOT / "data/processed/edge-reality"
BID, FAVOURITE = 80, 90
TOURS = {"KXATPMATCH": "ATP", "KXATPCHALLENGERMATCH": "Challenger", "KXWTAMATCH": "WTA",
         "KXWTACHALLENGERMATCH": "WTA Challenger", "KXITFMATCH": "ITF men", "KXITFWMATCH": "ITF women"}
FILES = (("tennis-archived-orders.json", "holdout"), ("tennis-orders.json", "recent"),
         ("wta-challenger-orders.json", "recent"), ("itf-men-orders.json", "recent"),
         ("itf-women-orders.json", "recent"))
CLAIMED = {("Challenger", "holdout"): (156, 7.08), ("Challenger", "recent"): (47, 7.23), ("WTA", "holdout"): (75, 8.00),
           ("WTA", "recent"): (20, 10.0), ("ATP", "holdout"): (48, -3.04), ("ATP", "recent"): (7, -8.6),
           ("WTA Challenger", "recent"): (23, 15.7), ("ITF men", "recent"): (189, -1.4)}
DISCOVERY_START = datetime(2026, 9, 18, tzinfo=timezone.utc).timestamp()   # NBA price rule first seen from here on
IDEAS = 40
RNG = np.random.default_rng(11)


def load_dip():
    """Filled 80c orders on 90c+ favourites, one per match, archive first. Also the raw recent rows for the mismatch check."""
    seen, fills, raw, overlap, orders = set(), [], [], set(), defaultdict(int)
    for name, period in FILES:
        path = DIP / name
        if not path.exists():
            continue
        for r in json.loads(path.read_text()):
            if r["limit"] != BID or r["start"] < FAVOURITE:
                continue
            tour = TOURS.get(r["event"].split("-")[0])
            if tour is None:
                continue
            row = dict(event=r["event"], tour=tour, period=period, settle=r["settle"], payout=r["payout"], net=100*r["payout"]-BID)
            if r["strict"]:
                raw.append(row)
            if r["event"] in seen:
                overlap.add(r["event"])
                continue
            seen.add(r["event"])
            orders[tour, period] += 1
            if r["strict"]:
                fills.append(row)
    return fills, raw, overlap, orders


def summary(rows, draws=4000):
    """Fills, win rate, net, bootstrap interval, and a one-sided test against the 80% a fair 80c contract needs."""
    n = len(rows)
    if not n:
        return dict(fills=0)
    nets = np.array([r["net"] for r in rows])
    boot = RNG.choice(nets, (draws, n)).mean(1)
    z = nets.mean()/(100*(.8*.2)**.5/n**.5)
    return dict(fills=n, wins=round(float(sum(r["payout"] for r in rows)), 2), win_rate=round(float(np.mean([r["payout"] for r in rows])), 4),
                net_cents=round(float(nets.mean()), 2), ci95=[round(float(v), 2) for v in np.percentile(boot, [2.5, 97.5])],
                z=round(float(z), 2), p_one_sided=float(stats.norm.sf(z)))


# ---------- step 2: hierarchical model ----------

def hierarchical(counts, tau_prior="half-normal 0.5", draws=40000):
    """Grid posterior for logit(p_i) ~ N(mu, tau). counts: {tour: (wins, fills)}."""
    names = list(counts)
    y = np.array([counts[k][0] for k in names], float)
    n = np.array([counts[k][1] for k in names], float)
    mus = np.linspace(logit(.55), logit(.97), 141)
    taus = np.linspace(0, 1.5, 76)
    z = np.linspace(-5, 5, 201)
    wz = stats.norm.pdf(z)
    wz /= wz.sum()
    theta = mus[:, None, None]+taus[None, :, None]*z[None, None, :]             # mu x tau x z
    logp, log1p = np.log(expit(theta)), np.log(expit(-theta))
    like = [y[i]*logp+(n[i]-y[i])*log1p for i in range(len(names))]            # log likelihood of each tour at each node
    marginal = sum(np.log((np.exp(l-l.max())*wz).sum(-1))+l.max() for l in like)   # mu x tau
    prior = stats.halfnorm.logpdf(taus, scale=.5) if tau_prior.startswith("half") else np.zeros_like(taus)
    post = np.exp(marginal+prior[None, :]-(marginal+prior[None, :]).max())
    post /= post.sum()
    tau_post = post.sum(0)
    # Bayes factor for tau = 0 (one shared win rate) against the tau prior, mu integrated out under its flat prior.
    by_tau = np.exp(marginal-marginal.max()).sum(0)
    prior_w = np.exp(prior)/np.exp(prior).sum()
    bf_shared = float(by_tau[0]/(by_tau*prior_w).sum())
    pick = RNG.choice(post.size, draws, p=post.ravel())
    mi, ti = np.unravel_index(pick, post.shape)
    sample = {}
    for i, name in enumerate(names):
        w = np.exp(like[i][mi, ti]-like[i][mi, ti].max(1, keepdims=True))*wz
        cdf = np.cumsum(w, 1)/w.sum(1, keepdims=True)
        zi = z[(cdf < RNG.random(draws)[:, None]).sum(1).clip(max=len(z)-1)]
        sample[name] = 100*expit(mus[mi]+taus[ti]*zi)-BID
    sample["a new tour"] = 100*expit(mus[mi]+taus[ti]*RNG.standard_normal(draws))-BID
    sample["common mean"] = 100*expit(mus[mi])-BID

    def describe(v):
        return dict(mean=round(float(v.mean()), 2), ci95=[round(float(q), 2) for q in np.percentile(v, [2.5, 97.5])],
                    prob_positive=round(float((v > 0).mean()), 3))
    out = dict(tau_prior=tau_prior, tau_logit=dict(mean=round(float((taus*tau_post).sum()), 3), mode=round(float(taus[tau_post.argmax()]), 2),
                                                 ci95=[round(float(taus[np.searchsorted(np.cumsum(tau_post), q)]), 2) for q in (.025, .975)]),
               between_tour_sd_cents_at_mean=round(float(np.mean(taus[ti]*100*expit(mus[mi])*(1-expit(mus[mi])))), 2),
               bayes_factor_one_shared_edge_vs_tours_differ=round(bf_shared, 3),
               tours={k: dict(raw_net=round(100*counts[k][0]/counts[k][1]-BID, 2), fills=int(counts[k][1]), **describe(sample[k])) for k in names},
               new_tour=describe(sample["a new tour"]), common_mean=describe(sample["common mean"]))
    return out, sample


def heterogeneity(counts, sims=200000):
    """Pearson chi-square for one shared win rate; p-value by simulation under the pooled rate."""
    y = np.array([v[0] for v in counts.values()], float)
    n = np.array([v[1] for v in counts.values()], float)
    p = y.sum()/n.sum()

    def chi(yy):
        return ((yy-n*p)**2/(n*p*(1-p))).sum(-1)
    observed = chi(y)
    simulated = chi(RNG.binomial(n.astype(int), p, (sims, len(n))))
    net = 100*y/n-BID
    var = 1e4*p*(1-p)/n
    w = 1/var
    q = (w*(net-(w*net).sum()/w.sum())**2).sum()
    tau2 = max(0.0, (q-(len(n)-1))/(w.sum()-(w**2).sum()/w.sum()))               # DerSimonian-Laird, cents squared
    return dict(tours=list(counts), pooled_win_rate=round(float(p), 4), pooled_net_cents=round(float(100*p-BID), 2),
                chi_square=round(float(observed), 2), df=len(n)-1, p_chi_square=float(stats.chi2.sf(observed, len(n)-1)),
                p_simulated=float((simulated >= observed).mean()), i_squared=round(float(max(0, 1-(len(n)-1)/observed)), 2),
                between_tour_sd_cents_method_of_moments=round(tau2**.5, 2))


# ---------- step 3: selection ----------

def selection_bias(fills, p, keep, sims=200000):
    """Equal true win rate p in every tour: observed fill-weighted net of the `keep` best-looking tours minus truth."""
    n = np.array(fills)
    rate = RNG.binomial(n, p, (sims, len(n)))/n
    order = np.argsort(-rate, 1)[:, :keep]
    chosen_n = n[order]
    observed = (np.take_along_axis(rate, order, 1)*chosen_n).sum(1)/chosen_n.sum(1)
    return round(float(100*(observed.mean()-p)), 2)


def selected_posterior(counts, chosen, sims=4000):
    """Selection-aware check of the shrunk estimate: draw truths from the fitted model, re-run the experiment,
    keep runs where the same tours come out on top, and read off the chosen tours' true net."""
    model, sample = hierarchical(counts)
    names = list(counts)
    n = np.array([counts[k][1] for k in names], int)
    idx = RNG.integers(0, len(sample[names[0]]), 200000)
    truth = np.stack([(sample[k][idx]+BID)/100 for k in names], 1)
    # sample[...] are marginal draws per tour from one joint draw of (mu, tau), so rows are joint draws.
    rate = RNG.binomial(n, truth)/n
    top = np.argsort(-rate, 1)[:, :len(chosen)]
    want = sorted(names.index(c) for c in chosen)
    hit = (np.sort(top, 1) == want).all(1)
    w = n[want]
    return dict(prob_these_tours_rank_top=round(float(hit.mean()), 3),
                true_net_given_selected=round(float(100*((truth[hit][:, want]*w).sum(1)/w.sum()).mean()-BID), 2))


# ---------- step 5: NBA and multiple testing ----------

def nba_entries():
    rows = json.loads((ROOT/"data/processed/game-state/nba-minutes.json").read_text())
    games = {}
    for r in sorted(rows, key=lambda r: r["ts"]):
        games.setdefault(r["event"], []).append(r)
    games = sorted(games.values(), key=lambda g: g[0]["ts"])
    cut = int(len(games)*.6)

    def first(game, lead):
        return next((r for r in game if 97 <= r["ask"] <= 99 and r["lead"] >= lead), None)
    out = {}
    for lead in (1, 6):
        hits = [(i >= cut, first(g, lead)) for i, g in enumerate(games)]
        out[lead] = [dict(h, later=later, in_discovery_window=h["ts"] >= DISCOVERY_START) for later, h in hits if h]
    return out, games[cut][0]["day"]


def nba_test(hits):
    """Net per contract and the chance of this few losses if every entry were exactly fair after the taker fee."""
    if not hits:
        return dict(trades=0)
    nets = np.array([h["net"] for h in hits])
    ask = np.array([h["ask"] for h in hits])/100
    lose = 1-(ask+.07*ask*(1-ask))                              # fair loss chance per entry
    dist = np.zeros(len(hits)+1)
    dist[0] = 1
    for q in lose:                                              # exact distribution of the number of losses
        dist[1:] = dist[1:]*(1-q)+dist[:-1]*q
        dist[0] *= 1-q
    losses = int(sum(not h["won"] for h in hits))
    p = float(dist[:losses+1].sum())
    boot = RNG.choice(nets, (4000, len(nets))).mean(1)
    return dict(trades=len(hits), losses=losses, fair_losses_expected=round(float(lose.sum()), 1), net_cents=round(float(nets.mean()), 2),
                ci95=[round(float(v), 2) for v in np.percentile(boot, [2.5, 97.5])], p_one_sided=p, z=round(float(stats.norm.isf(p)), 2),
                sd_per_trade_cents=round(float(nets.std()), 1))


def lead_filter_luck(hits, grid=(4, 6, 8, 10, 12, 15), sims=20000):
    """If losses were unrelated to the lead (within an ask price), how often would the best lead screen from the
    grid remove as large a share of them as the one found after looking at the losers?"""
    lead = np.array([h["lead"] for h in hits])
    ask = np.array([h["ask"] for h in hits])
    lost = np.array([not h["won"] for h in hits])
    net = np.array([h["net"] for h in hits])

    def gain(mask_lost, k):
        keep = lead >= k
        value = np.where(mask_lost, -ask-.07*ask*(100-ask)/100, 100-ask-.07*ask*(100-ask)/100)
        return value[keep].mean()-value.mean()
    observed = {k: float(gain(lost, k)) for k in grid}
    best = np.empty(sims)
    for s in range(sims):
        shuffled = lost.copy()
        for a in (97, 98, 99):
            idx = np.where(ask == a)[0]
            shuffled[idx] = RNG.permutation(lost[idx])
        best[s] = max(gain(shuffled, k) for k in grid)
    return dict(losses_by_lead=sorted(int(v) for v in lead[lost]), share_of_entries_with_lead_under_6=round(float((lead < 6).mean()), 3),
                gain_cents_by_min_lead={k: round(v, 2) for k, v in observed.items()},
                p_lead6_alone=float(stats.hypergeom.sf(int(lost[lead < 6].sum())-1, len(lead), int(lost.sum()), int((lead < 6).sum()))),
                p_best_screen_in_grid=float((best >= observed[6]-1e-9).mean()))


def chance_real(p_value_z, se, edge_max, priors=(.05, .1, .25, .5)):
    """Posterior chance the edge is real. H0: zero edge. H1: edge uniform on (0, edge_max) cents. Normal likelihood."""
    xbar = p_value_z*se
    grid = np.linspace(0, edge_max, 400)
    bf = float(stats.norm.pdf(xbar, grid, se).mean()/stats.norm.pdf(xbar, 0, se))
    return dict(bayes_factor_for_real=round(bf, 1), **{f"prior_{int(100*q)}pct": round(q*bf/(q*bf+1-q), 2) for q in priors})


def adjust(p_values, m=IDEAS):
    """Bonferroni and Benjamini-Hochberg, assuming these are the smallest p-values among m tests."""
    order = sorted(p_values, key=p_values.get)
    out = {}
    for rank, k in enumerate(order, 1):
        p = p_values[k]
        out[k] = dict(p=p, bonferroni_p=min(1.0, p*m), passes_bonferroni_5pct=p*m < .05, bh_q=min(1.0, p*m/rank), passes_fdr_10pct=p*m/rank < .10)
    return out


def fills_needed(edge, sd, alpha=.05, power=.8):
    return int(np.ceil(((stats.norm.isf(alpha)+stats.norm.isf(1-power))*sd/edge)**2))


def main():
    fills, raw, overlap, orders = load_dip()
    by = defaultdict(list)
    for r in fills:
        by[r["tour"], r["period"]].append(r)
    report = {}

    # 1. table
    table, mismatches = {}, []
    raw_by = defaultdict(list)
    for r in raw:
        raw_by[r["tour"], r["period"]].append(r)
    print("1. DIP TABLE (80c bid, 90c+ favourite, strict fills; matches in both files kept in the holdout)")
    for key in sorted(by):
        s = summary(by[key])
        s["orders"] = orders[key]
        rawn, rawnet = len(raw_by[key]), float(np.mean([r["net"] for r in raw_by[key]]))
        s["before_dedup"] = dict(fills=rawn, net_cents=round(rawnet, 2))
        claim = CLAIMED.get(key)
        if claim:
            s["claimed"] = dict(fills=claim[0], net_cents=claim[1])
            if claim[0] != s["fills"] or abs(claim[1]-s["net_cents"]) > .06:
                mismatches.append(f"{key[0]} {key[1]}: claimed {claim[0]} fills {claim[1]:+.2f}c; de-duplicated {s['fills']} fills "
                                  f"{s['net_cents']:+.2f}c (before de-duplication {rawn} fills {rawnet:+.2f}c)")
        table[" / ".join(key)] = s
        print(f"  {key[0]:15} {key[1]:8} orders {orders[key]:4} fills {s['fills']:4} win {s['win_rate']:.3f} net {s['net_cents']:+6.2f}c "
              f"ci {s['ci95']}  (raw {rawn} fills {rawnet:+.2f}c)")
    for m in mismatches:
        print("  MISMATCH", m)
    report["1_table"] = dict(rows=table, mismatches=mismatches, matches_in_both_files=len(overlap),
                             itf_women_included=(DIP/"itf-women-orders.json").exists())

    # 2. pooling
    counts = defaultdict(lambda: [0.0, 0])
    for r in fills:
        counts[r["tour"]][0] += r["payout"]
        counts[r["tour"]][1] += 1
    counts = {k: tuple(v) for k, v in sorted(counts.items(), key=lambda kv: -kv[1][1])}
    pooled = summary(fills)
    model, sample = hierarchical(counts)
    flat, _ = hierarchical(counts, "uniform 0-1.5")
    three = {k: counts[k] for k in ("ATP", "Challenger", "WTA")}
    model3, sample3 = hierarchical(three)
    w = {k: counts[k][1] for k in ("Challenger", "WTA")}
    combo = (sample["Challenger"]*w["Challenger"]+sample["WTA"]*w["WTA"])/sum(w.values())
    combo3 = (sample3["Challenger"]*w["Challenger"]+sample3["WTA"]*w["WTA"])/sum(w.values())
    alln = sum(v[1] for v in counts.values())
    everything = sum(sample[k]*counts[k][1] for k in counts)/alln

    def desc(v):
        return dict(mean=round(float(v.mean()), 2), ci95=[round(float(q), 2) for q in np.percentile(v, [2.5, 97.5])], prob_positive=round(float((v > 0).mean()), 3))
    report["2_pooling"] = dict(all_tours_pooled=pooled, hierarchical=model, hierarchical_uniform_tau_prior=flat, hierarchical_original_three_tours=model3,
                               heterogeneity_all_tours=heterogeneity(counts), heterogeneity_original_three=heterogeneity(three),
                               challenger_plus_wta_shrunk=desc(combo), challenger_plus_wta_shrunk_three_tour_model=desc(combo3),
                               all_tours_at_current_fill_mix_shrunk=desc(everything))
    print(f"\n2. POOLED all tours: {pooled['fills']} fills, win {pooled['win_rate']:.3f}, net {pooled['net_cents']:+.2f}c ci {pooled['ci95']}")
    print(f"   between-tour sd (logit) mean {model['tau_logit']['mean']} ci {model['tau_logit']['ci95']}; about {model['between_tour_sd_cents_at_mean']}c; "
          f"Bayes factor shared-edge vs differ {model['bayes_factor_one_shared_edge_vs_tours_differ']}")
    for k, v in model["tours"].items():
        print(f"   {k:15} fills {v['fills']:4} raw {v['raw_net']:+6.2f}c  shrunk {v['mean']:+6.2f}c ci {v['ci95']} P(>0) {v['prob_positive']}")
    print(f"   a new tour: {model['new_tour']}; common mean: {model['common_mean']}")
    print(f"   uniform tau prior: tau {flat['tau_logit']}, about {flat['between_tour_sd_cents_at_mean']}c, BF {flat['bayes_factor_one_shared_edge_vs_tours_differ']}, "
          f"tours { {k: (v['mean'], v['ci95']) for k, v in flat['tours'].items()} }")
    print(f"   three-tour model: tau {model3['tau_logit']}, BF {model3['bayes_factor_one_shared_edge_vs_tours_differ']}, "
          f"tours { {k: (v['mean'], v['ci95']) for k, v in model3['tours'].items()} }")
    for name in ("heterogeneity_all_tours", "heterogeneity_original_three"):
        h = report["2_pooling"][name]
        print(f"   {name}: chi2 {h['chi_square']} df {h['df']} p {h['p_simulated']:.4f}, I2 {h['i_squared']}, between-tour sd {h['between_tour_sd_cents_method_of_moments']}c")
    print(f"   Challenger+WTA shrunk: {desc(combo)}; three-tour model: {desc(combo3)}")

    # 3. selection
    p_all = pooled["win_rate"]
    n3 = [counts[k][1] for k in ("ATP", "Challenger", "WTA")]
    p3 = sum(counts[k][0] for k in three)/sum(n3)
    nall = [v[1] for v in counts.values()]
    raw_cw = 100*(counts["Challenger"][0]+counts["WTA"][0])/(counts["Challenger"][1]+counts["WTA"][1])-BID
    atp_n, atp_y = counts["ATP"][1], counts["ATP"][0]
    sel = dict(equal_edge_bias_cents={"best 2 of the original 3": selection_bias(n3, p3, 2),
                                      f"best 2 of all {len(nall)}": selection_bias(nall, p_all, 2),
                                      f"best 3 of all {len(nall)}": selection_bias(nall, p_all, 3)},
               challenger_plus_wta_raw_net=round(raw_cw, 2),
               equal_edge_three_tour_net=round(100*p3-BID, 2),
               prob_atp_this_bad_if_equal_edge=float(stats.binom.cdf(int(round(atp_y)), atp_n, p3)),
               prob_worst_of_three_this_bad=float((RNG.binomial(n3, p3, (200000, 3))/np.array(n3)).min(1).__le__(atp_y/atp_n).mean()),
               selection_aware_three_tour_model=selected_posterior(three, ("Challenger", "WTA")))
    report["3_selection"] = sel
    print(f"\n3. SELECTION: bias if all tours equal {sel['equal_edge_bias_cents']}; Challenger+WTA raw {raw_cw:+.2f}c vs three-tour pooled {100*p3-BID:+.2f}c")
    print(f"   P(ATP this bad | equal edge) {sel['prob_atp_this_bad_if_equal_edge']:.4f}; P(worst of three this bad) {sel['prob_worst_of_three_this_bad']:.4f}")
    print(f"   selection-aware: {sel['selection_aware_three_tour_model']}")

    # 4. clean subsets
    clean = lambda rows: [r for r in rows if r["event"] not in overlap]
    hold = [r for r in fills if r["period"] == "holdout"]
    newer = [r for r in fills if r["tour"] in ("WTA Challenger", "ITF men", "ITF women")]
    oos = {"A. holdout, three tours pooled, minus matches also in the discovery file (clean for the pooled rule)": summary(clean(hold)),
           "A2. same, holdout as originally scored": summary(hold),
           "B. tours added after the tour choice (WTA Challenger, ITF): clean for 'the rule works across tennis'": summary(newer),
           "C. A and B together: every fill not used to find the rule": summary(clean(hold)+newer),
           "D. Challenger+WTA holdout minus overlap (NOT clean: the tour choice saw it)": summary([r for r in clean(hold) if r["tour"] in ("Challenger", "WTA")]),
           "E. Challenger+WTA recent (discovery sample; NOT clean)": summary([r for r in fills if r["period"] == "recent" and r["tour"] in ("Challenger", "WTA")])}
    halves = sorted(clean(hold), key=lambda r: r["settle"])
    oos["A by half: first"], oos["A by half: second"] = summary(halves[:len(halves)//2]), summary(halves[len(halves)//2:])
    report["4_out_of_sample"] = oos
    print("\n4. OUT OF SAMPLE")
    for k, v in oos.items():
        print(f"   {k}: {v['fills']} fills net {v['net_cents']:+.2f}c ci {v['ci95']} z {v['z']} p {v['p_one_sided']:.4f}")

    # 5. NBA and multiple testing
    nba, split_day = nba_entries()
    parts = {}
    for lead, name in ((1, "plain"), (6, "lead 6+")):
        h = nba[lead]
        parts[name] = {"earlier 60%": nba_test([x for x in h if not x["later"]]), "later 40%": nba_test([x for x in h if x["later"]]),
                       "all": nba_test(h), "before the discovery window (clean)": nba_test([x for x in h if not x["in_discovery_window"]]),
                       "inside the discovery window (18 Sep on)": nba_test([x for x in h if x["in_discovery_window"]]),
                       "later 40%, before the discovery window": nba_test([x for x in h if x["later"] and not x["in_discovery_window"]])}
    luck = lead_filter_luck(nba[1])
    price_only = {}
    path = ROOT/"data/processed/near-end/holdout-trades.json"
    if path.exists():
        trades = [t for t in json.loads(path.read_text()) if t["series"] == "KXNBAGAME"]
        for thr in sorted({t["threshold"] for t in trades}):
            price_only[f">={thr}c"] = nba_test([dict(net=t["net_cents"], ask=t["price"], won=t["payout"] > .5) for t in trades if t["threshold"] == thr])
    pv = {"dip, clean holdout, three tours pooled": oos["A. holdout, three tours pooled, minus matches also in the discovery file (clean for the pooled rule)"]["p_one_sided"],
          "NBA plain, games before the discovery window": parts["plain"]["before the discovery window (clean)"]["p_one_sided"],
          "NBA plain, 'later 40%' as reported": parts["plain"]["later 40%"]["p_one_sided"]}
    for thr in (">=97c", ">=98c"):
        if thr in price_only:
            pv[f"NBA price-only Jan-Sep holdout {thr}"] = price_only[thr]["p_one_sided"]
    a = oos["A. holdout, three tours pooled, minus matches also in the discovery file (clean for the pooled rule)"]
    b = parts["plain"]["before the discovery window (clean)"]
    real = {"dip (H1: edge uniform 0-10c)": chance_real(a["z"], 40/a["fills"]**.5, 10),
            "NBA plain, clean games (H1: edge uniform 0-2c)": chance_real(b["z"], b["net_cents"]/b["z"] if b["z"] > 0 else 1, 2)}
    if ">=97c" in price_only:
        c = price_only[">=97c"]
        real["NBA price-only holdout >=97c (H1: edge uniform 0-2c)"] = chance_real(c["z"], c["net_cents"]/c["z"], 2)
    alpha = .025
    report["5_multiple_testing"] = dict(
        nba=parts, nba_split_day=split_day, nba_price_only_holdout_by_threshold=price_only, lead_filter=luck,
        adjusted={f"{m} tests": adjust({k: v for k, v in pv.items() if "as reported" not in k}, m) for m in (20, IDEAS, 80)},
        p_values=pv, chance_real=real,
        global_null=dict(tests=IDEAS, pass_level_one_sided=alpha, expected_false_passes=IDEAS*alpha,
                         prob_two_or_more_false_passes=float(stats.binom.sf(1, IDEAS, alpha))))
    print(f"\n5. NBA (60/40 split falls on {split_day}; discovery window starts 2026-09-18)")
    for name, d in parts.items():
        for k, v in d.items():
            if v["trades"]:
                print(f"   {name:8} {k:42} trades {v['trades']:4} losses {v['losses']} (fair {v['fair_losses_expected']:5.1f}) net {v['net_cents']:+.2f}c ci {v['ci95']} p {v['p_one_sided']:.4f} z {v['z']}")
    for k, v in price_only.items():
        print(f"   price-only holdout {k}: trades {v['trades']} losses {v['losses']} (fair {v['fair_losses_expected']}) net {v['net_cents']:+.2f}c p {v['p_one_sided']:.4f}")
    print(f"   lead filter: {luck}")
    for m, d in report["5_multiple_testing"]["adjusted"].items():
        for k, v in d.items():
            print(f"   {m}: {k}: p {v['p']:.4f} Bonferroni {v['bonferroni_p']:.3f} BH q {v['bh_q']:.3f}")
    print(f"   chance real: {real}")
    print(f"   global null: {report['5_multiple_testing']['global_null']}")

    # 6. sample sizes
    need = {}
    for label, sd, edges in (("dip (sd per fill at an 86% win rate, cents)", 100*(.86*.14)**.5, (2, 3, 4, 5, 6, 7)),
                             ("NBA plain (sd per trade from the data)", parts["plain"]["all"]["sd_per_trade_cents"], (.25, .5, .75, 1, 1.5)),
                             ("NBA lead 6+ (sd per trade from the data)", parts["lead 6+"]["all"]["sd_per_trade_cents"], (.5, 1, 1.5))):
        need[label] = dict(sd=round(float(sd), 1), fills_for_80pct_power_at_5pct_one_sided={f"{e}c": fills_needed(e, sd) for e in edges})
    report["6_fills_needed"] = need
    print("\n6. FILLS NEEDED (80% power, one-sided 5%)")
    for k, v in need.items():
        print(f"   {k}: sd {v['sd']}c  {v['fills_for_80pct_power_at_5pct_one_sided']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
