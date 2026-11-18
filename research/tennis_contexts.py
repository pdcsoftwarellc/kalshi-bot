"""Is pre-match mispricing in Kalshi tennis concentrated in identifiable contexts?

Offline only. Replays the listings and one-minute bars already cached under
data/raw/cost-research-v1 (the HTTP client is a stub that answers 404) and
writes data/processed/tennis-contexts/.

  build    One row per listed match on ATP, WTA, ATP Challenger, WTA
           Challenger, ITF men and ITF women (archive first, so a match listed
           twice keeps its archive copy): both players (name and Kalshi's
           competitor id), tournament and round from the rules text, settlement
           value, and - where the six-hour bar window is cached and the first
           15 midpoints are stable (favorite_dip.py's definition, but with NO
           price floor) - the favourite, its pre-match price, the quote at the
           end of the warm-up, the detected start of play, the in-play low and
           the last 25 bars before the close (for the retirement detector).
  report   Sections 1-6 of the question: qualifying unpacked, walk-forward
           player effects, tournament effects, underdogs, retirements, the
           tradeable versions and the multiplicity statement; then (7), tables
           drawn AFTER the screen to describe its leads (the draw ladder, open
           interest thirds, bid ladders, the dip rule split three ways).
           Retirement status comes from the tennis provider's fixtures already
           on disk (ATP and Challenger only); nothing is fetched.

Conventions
* All gaps are on the FAVOURITE: 100 x settlement value - pre-match price, in
  cents. A negative gap means the underdog was underpriced by the same amount.
  Fractional settlements count as fractional wins. Prices of 99c+ are dropped.
* "holdout" = settled before 2026-09-18 (Kalshi's archive), "recent" = after.
* Walk-forward: a player's history at a match is every listed match of that
  player that SETTLED before this match's pricing time (settlement minus six
  hours, the start of the bar window). Nothing later is used.
* Intervals: 95% bootstrap, resampling PLAYERS (the favourite, or the player
  the context is about), 2,000 draws, so a player with many matches does not
  count as many independent observations.
* Tradeable version: buy the favourite at the ask at the end of the warm-up
  (or the underdog at 100 - favourite bid) and pay the taker fee
  0.07*p*(1-p); hold to settlement.
* Dip filter: the live rule (favourite 90c+, resting bid at 80c, filled when a
  later in-play minute prints strictly below 80c): do fills in the context win
  more than fills outside it?
* Multiplicity: every context x price-band cell is one test of "the gap
  differs from the rest of the band". z uses the larger of the cluster-robust
  standard error and the one the prices imply (p(1-p)), so a small group with
  no losses cannot look certain. The report counts the cells, how many have
  the same sign and |z| > 1.96 in both periods, and how many would by chance;
  then shrinks every contrast with a normal-normal model fitted over all
  cells (as edge_reality.py does across tours).
* Rounds are re-parsed here: favourite_gap.parse_rules reads "Qualification
  Final" as a Final, "1/16 Finals" as Finals and misses "round of 32".
* Players are keyed by name joined with Kalshi's competitor id (the ids are
  not stable across series, so neither alone is enough).

    .venv/bin/python research/tennis_contexts.py build
    .venv/bin/python research/tennis_contexts.py report

Public market data already on disk; no orders, no authentication, no network.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import median
import sys
import unicodedata

import httpx
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fetch, save, timestamp
from research.dip_archive_tours import cached
from research.favorite_dip import WARMUP
from research.favourite_gap import CUTOFF, STUB, TOURS, bar_params, detect_start, listing, mid, parse_rules, side_path
from research.near_end import cents

OUT = ROOT / "data/processed/tennis-contexts"
TAIL = 25


def study(job):
    """One match from its cached bars: favourite, pre-match price, start of play, in-play low, the last bars."""
    path, ticker, settle, close = job
    params = bar_params(settle)
    if not cached(path, params):
        return dict(bars=0)
    with httpx.Client(transport=STUB) as client:
        bars = sorted(fetch(client, path, params).get("candlesticks") or [], key=lambda b: b["end_period_ts"])
    if not bars:
        return dict(bars=0)
    out = dict(bars=len(bars), stable=False)
    mids = []
    for b in bars[:WARMUP]:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        if bid is None or ask is None or not 0 < bid <= ask < 100 or ask-bid > 4:
            break
        mids.append((bid+ask)/2)
    # The last bars on the YES side of the first market, whatever the warm-up looked like.
    yes_path = side_path(bars, True)
    out["tail_yes"] = [[int(close-p[0]), p[1], p[2], p[3], p[4], p[5]] for p in yes_path[-TAIL:]]
    if len(mids) < WARMUP or max(mids)-min(mids) > 3:
        return out
    yes = median(mids) > 50
    start = median(mids) if yes else 100-median(mids)
    p = side_path(bars, yes)
    began = detect_start(p)
    place = WARMUP-1
    out.update(stable=True, yes=yes, start=start, bid0=p[place][1], ask0=p[place][2], detected=began is not None,
               spread_warmup=float(np.mean([q[2]-q[1] for q in p[:WARMUP]])), volume_warmup=sum(q[5] for q in p[:WARMUP]),
               first_bar_ts=p[0][0])
    after = [q[3] for q in p[(began+1 if began is not None else WARMUP):] if q[3] is not None]
    highs = [q[4] for q in p[(began+1 if began is not None else WARMUP):] if q[4] is not None]
    out.update(low_in_play=min(after) if after else None, high_in_play=max(highs) if highs else None)
    if began is not None:
        last = began-1
        now = next(((j, mid(p[j])) for j in range(last, -1, -1) if mid(p[j]) is not None), None)
        hour_back = [q for q in p[:last+1] if q[0] > p[last][0]-3600]
        when = datetime.fromtimestamp(p[began][0], timezone.utc)
        out.update(start_ts=p[began][0], price_before_play=now[1] if now else None,
                   bid_before_play=p[now[0]][1] if now else None, ask_before_play=p[now[0]][2] if now else None,
                   volume_60m=sum(q[5] for q in hour_back), prematch_volume=sum(q[5] for q in p[place+1:began]),
                   open_interest=p[last][6], hour_utc=when.hour, weekday=when.weekday())
    return out


def build(workers):
    rows, jobs, seen = [], [], set()
    with httpx.Client(transport=STUB) as client:
        for archived in (True, False):
            for series in TOURS:
                events, path, lo, hi = listing(client, series, archived)
                kept = 0
                for event, group in events.items():
                    if event in seen or len(group) != 2:
                        continue
                    a, b = sorted(group, key=lambda m: m["ticker"])
                    try:
                        outcome, settle = float(a["settlement_value_dollars"]), int(timestamp(a["settlement_ts"]))
                        other = float(b["settlement_value_dollars"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    seen.add(event)
                    kept += 1
                    tournament, stage = parse_rules(a.get("rules_primary"))
                    close = int(timestamp(a["close_time"])) if a.get("close_time") else settle
                    rows.append(dict(
                        event=event, series=series, tour=TOURS[series], archived=archived, ticker=a["ticker"], settle=settle, close=close,
                        in_bar_range=lo <= settle < hi, period="holdout" if settle < CUTOFF else "recent",
                        opened=int(timestamp(a["open_time"])) if a.get("open_time") else None,
                        p1=a.get("yes_sub_title"), p2=b.get("yes_sub_title"),
                        id1=(a.get("custom_strike") or {}).get("tennis_competitor"), id2=(b.get("custom_strike") or {}).get("tennis_competitor"),
                        outcome=outcome, outcome_other=other, tournament=tournament, round=stage,
                        volume1=float(a.get("volume_fp") or a.get("volume") or 0), volume2=float(b.get("volume_fp") or b.get("volume") or 0),
                        walkover_rule="resolve to $0.50" in (a.get("rules_secondary") or "")))
                    jobs.append((path.format(t=a["ticker"]), a["ticker"], settle, close))
                print(f"  {TOURS[series]:15} {'archive' if archived else 'recent':8} events {kept:6}", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for n, (row, extra) in enumerate(zip(rows, pool.map(study, jobs, chunksize=200))):
            row.update(extra)
            if n % 10000 == 0:
                print(f"  bars {n} of {len(rows)}", flush=True)
    for tour in TOURS.values():
        for period in ("holdout", "recent"):
            part = [r for r in rows if r["tour"] == tour and r["period"] == period]
            print(f"  {tour:15} {period:8} listed {len(part):6} with bars {sum(r['bars'] > 0 for r in part):6} "
                  f"stable {sum(r.get('stable', False) for r in part):6}")
    save(OUT/"rows.json", rows)



# ---------------------------------------------------------------- report: preparation

RNG = np.random.default_rng(9)
BANDS = (("50-60", 50, 60), ("60-70", 60, 70), ("70-80", 70, 80), ("80-90", 80, 90), ("90-99", 90, 99), ("all", 50, 99))
EDGE = ("Challenger", "WTA", "WTA Challenger")
LEVEL = {"ATP": 3, "WTA": 3, "Challenger": 2, "WTA Challenger": 2, "ITF men": 1, "ITF women": 1}
DAY = 86400
COVERAGE = datetime(2026, 4, 11, tzinfo=timezone.utc).timestamp()      # every tour is listed from here on
HISTORY = datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()        # "new to the books" is only meaningful after this
COUNTRIES = {
    "TUN": "Monastir|Tunis", "EGY": "Sharm ElSheikh|Hurghada|Soma Bay", "GRE": "Heraklion|Hersonissos|Athens",
    "ESP": "Madrid|Valencia|Villena|Mallorca|Seville|Tenerife|Barcelona|Manacor|Segovia|Pozoblanco|Murcia|Menorca|Montemar|Baza|Yecla|"
           "Las Palmas de Gran Canaria|Nules|Melilla|Zaragoza|Reus|Logrono|Gijon|Pontevedra|Sabadell|Zamora|Badalona|Torelló|Vigo|Oviedo|"
           "La Bisbal D Emporda",
    "ITA": "Rome|Santa Margherita di Pula|Genoa|Palermo|Parma|Trieste|San Gregorio di Catania|Como|Cordenons|Olbia|Biella|Bari|Monza|"
           "Florence|Catania|Todi|Milan|Francavilla|Naples|Taranto|Perugia|Cervia|Rovereto|Tolentino|Vicenza|Barletta|Cattolica|Cesenatico|"
           "Brescia|Cagliari|Courmayeur|Foggia|Modena|Selva Gardena|Pozzuoli|Bologna|Viserba di Rimini|Fiano Romano|San Marino",
    "USA": "Cincinnati|Indian Wells|Miami|US Open Men Singles|US Open Women Singles|Austin|San Diego|Washington|Newport|Charleston|"
           "Winston Salem|Bloomfield Hills|Fort Worth|Las Vegas NV|Hilton Head Island SC|Norman OK|Tiburon|Columbus|Lexington|Sioux Falls|"
           "Brownsburg|Sarasota|Lincoln (NE)|Berkeley CA|Cary|Little Rock|Charlottesville|Lexington SC|Savannah|Memphis|Dallas|Knoxville|"
           "Tallahassee|Houston|Delray Beach|Cleveland|New Braunfels TX|Templeton CA|Philadelphia|Columbus OH|Phoenix|Naples FL|Redding CA|"
           "Harlingen TX|Ithaca|Nashville TN|Tyler|Ann Arbor MI|Orlando FL|Weston FL|Austin TX|Lincoln NE|Baton Rouge|Sumter SC|Lexington KY|"
           "Fayetteville AR|Bakersfield CA|Edmond OK|Columbia SC|Midland|Champaign IL|Champaign|Pembroke",
    "CAN": "Toronto|Montreal|Vancouver|Quebec City|Winnipeg|Granby|Saguenay|Trois-Rivières|Drummondville",
    "POR": "Oeiras|Porto|Lisbon|Maia|Sintra|Braga|Estoril|Figueira Da Foz|Caldas da Rainha|Quinta do Lago|Leiria|Evora|Setubal|Vale do Lobo|"
           "Loule|Funchal",
    "TUR": "Istanbul|Antalya|Samsun|Adana|Ankara|Kayseri", "AUS": "Brisbane|Australian Open Men Singles|Australian Open Women Singles|"
           "Adelaide|Sydney|Wagga Wagga|Darwin|Tamworth|Hobart", "NZL": "Auckland",
    "CHN": "Maanshan|Wuning|Shanghai|Beijing|Tianjin|Ningbo|Jingshan|Jiujiang|Wuhan|Guangzhou|Qian Daohu|Luan|Guiyang|Shenzhen|"
           "Guangzhou Nansha|Jinan|Zhangjiagang|Wuxi|Suzhou|Hangzhou|Chengdu|Huzhou|Yinchuan|Shenyang|Kunshan|Guangzhou (Huangpu)|Hong Kong",
    "GBR": "Wimbledon Men Singles|Wimbledon Women Singles|Nottingham|Birmingham|Roehampton|Eastbourne|London|Ilkley|Glasgow|Wrexham|"
           "Sheffield|Redbridge|Aldershot",
    "FRA": "French Open Men Singles|French Open Women Singles|Paris|Lyon|Cassis|Brest|Mouilleron-Le-Captif|Rennes|Roanne|Saint Tropez|"
           "Rouen|Troyes|Royan|Contrexeville|St. Brieuc|Orleans|Reims|Villeneuve d'Ascq|Cherbourg|Thionville|Lille|Strasbourg|Quimper|"
           "Montpellier|Aix en Provence|Cap d'Agde|Bordeaux|Saint-Malo|Sarreguemines|Pau|Saint-Palais-sur-Mer|Cherbourg-en-Cotentin|"
           "Poitiers|Le Neubourg|Les Sables d Olonne|Dijon|Forbach|Nevers|Metz|Rodez|Bagnères-de-Bigorre|Plaisir + H|Monte Carlo",
    "GER": "Hamburg|Stuttgart|Augsburg|Hagen|Bonn|Braunschweig|Halle|Munich|Berlin|Bad Rappenau|Koblenz|Bad Homburg|Offenbach|Meerbusch|"
           "Erwitte|Essen|Allershausen",
    "BUL": "Plovdiv|Burgas|Varna|Pazardzhik|Stara Zagora", "SRB": "Kursumlijska Banja|Zlatibor", "SGP": "Singapore", "VIE": "Phan Thiet",
    "RWA": "Kigali", "UAE": "Dubai|Abu Dhabi|Fujairah", "QAT": "Doha", "BRN": "Manama",
    "CZE": "Prague|Ostrava|Liberec|Prostejov|Pardubice|Karlovy Vary",
    "BRA": "Curitiba|Sao Paulo|Maceio|Santos|Rio de Janeiro|Costa do Sauipe|São Carlos|Campinas|Sao Leopoldo|Santa Cruz do Sul|Piracicaba|"
           "Brasilia|Belem|Belém|Campos do Jordão|Campos do Jordao|Salvador|Porto Velho|João Pessoa|Maringa|Campina Grande|Barueri|Sao Luis|"
           "Itajai|Lajeado",
    "ROU": "Iasi|Targu Mures|Brasov|Bucharest|Cluj Napoca|Slobozia|Bistrita|Constanta|Campulung|Buzau", "ECU": "Quito|Guayaquil",
    "COL": "Bogota|Cali|Bucaramanga|Barranquilla|Sopo", "MDA": "Chisinau", "IND": "Bengaluru|Chennai|Mumbai|Pune|New Delhi|Solapur",
    "BOL": "Santa Cruz", "KOR": "Seoul|Yeongwol|Busan|Gwangju|Incheon",
    "JPN": "Tokyo|Osaka|Yokkaichi|Yokohama|Miyazaki|Matsuyama|Hamamatsu|Sapporo|Nanao|Urayasu|Kyoto|Tanagura|Yanagawa City|Kobe",
    "CRO": "Bol|Zadar|Zagreb|Split|Umag|Dubrovnik|Makarska|Sibenik", "RSA": "Centurion", "CHI": "Santiago|Antofagasta|Concepcion|Colina",
    "PAR": "Asuncion", "AUT": "Kitzbuhel|Vienna|Tulln|Mauthausen|Linz", "TPE": "Taipei", "ANG": "Luanda", "SWE": "Bastad|Stockholm|Falun|Taby",
    "MEX": "Morelia|Guadalajara|Morelos|Mexico City|San Luis Potosi|Monterrey|Merida|Los Cabos|Acapulco|Queretaro|Cancun|Huamantla|"
           "Chihuahua|Metepec|Mazatlan",
    "NED": "S-Hertogenbosch|Bunschoten|Rotterdam|Haren|Oldenzaal", "MAR": "Rabat|Marrakech|Casablanca",
    "POL": "Grodzisk Mazowiecki|Poznan|Szczecin|Warsaw|Krakow|Radom|Szczawno|Bytom|Bydgoszcz", "THA": "Nonthaburi|Hua Hin", "IDN": "Bali",
    "ARG": "Neuquén|Trelew|Buenos Aires|Villa Maria|Rosario|Cordoba|San Miguel de Tucuman|Tigre|Roldan|Luján|Mar Del Plata|Cipolletti",
    "KAZ": "Astana|Shymkent|Almaty", "AZE": "Baku", "SUI": "Verbier|Zug|Sion|Basel|Lugano|Montreux|Gstaad|Geneva|Lausanne", "UZB": "Fergana",
    "MLT": "Malta", "FIN": "Tampere|Helsinki", "HUN": "Szabolcsveresmart|Székesfehérvár|Budapest|Pécs|Pecs", "SVK": "Bratislava|Kosice",
    "BEL": "Liege|Brussels|Wanfercée-Baulet", "GEO": "Telavi", "LUX": "Petange", "IRL": "Dublin", "CIV": "Abidjan", "ARM": "Tsaghkadzor",
    "PER": "Lima", "MAS": "Kuala Lumpur", "DOM": "Cap Cana", "PAK": "Islamabad", "SLO": "Ljubljana|Maribor", "COG": "Brazzaville",
    "URU": "Montevideo", "PHI": "Manila", "JAM": "Kingston"}
REGIONS = {"W Europe": "ESP ITA POR GBR FRA GER AUT SUI NED BEL LUX IRL SWE FIN MLT GRE",
           "E Europe+Turkey": "BUL SRB CZE ROU MDA CRO POL HUN SVK SLO TUR GEO ARM AZE", "N America": "USA CAN",
           "Latin America": "BRA ARG CHI COL ECU BOL PAR PER URU MEX DOM JAM",
           "Asia-Pacific": "CHN JPN KOR TPE SGP VIE THA IDN MAS PHI AUS NZL IND PAK KAZ UZB",
           "Africa+Mid-East": "TUN EGY RWA UAE QAT BRN RSA ANG MAR CIV COG"}
CITY = {city.lower(): code for code, cities in COUNTRIES.items() for city in cities.split("|")}
REGION = {code: region for region, codes in REGIONS.items() for code in codes.split()}
RESORTS = {"monastir", "sharm elsheikh", "hurghada", "antalya", "heraklion", "kursumlijska banja", "santa margherita di pula"}


def parse_stage(row):
    """Tournament and stage from the two pieces favourite_gap.parse_rules returns. That parser reads
    'Qualification Final' as a Final, '1/16 Finals' as Finals and misses lower-case 'round of 32'; this one does not."""
    full = " ".join(x for x in (row["tournament"], row["round"]) if x)
    found = re.search(r"\s+(qualification final|qualification round \d+|qualification|qualifying|1/\d+[- ]finals?|semi[- ]?finals?|"
                      r"quarter[- ]?finals?|round of \d+|finals?|group \w+|\w+ group|group stage|round robin|round)\s*$", full, re.I)
    if not found:
        return full or None, None
    text, name = found.group(1).lower(), full[:found.start()].strip(" ,-:")
    if text.startswith("qualif"):
        stage = "QF" if text.endswith("final") else "Q"+text[-1] if text[-1].isdigit() else "Q"
        return name, "qual "+stage
    if text.startswith("1/"):
        n = int(re.search(r"1/(\d+)", text).group(1))
        return name, {4: "QF", 2: "SF"}.get(n, f"R{2*n}")
    if text.startswith("round of"):
        return name, "R"+text.split()[-1]
    if text.startswith("quarter"):
        return name, "QF"
    if text.startswith("semi"):
        return name, "SF"
    if text.startswith("final"):
        return name, "F"
    if "group" in text or "robin" in text:
        return name, "group"
    return name, None


def city_of(name):
    city = re.sub(r"^(\d{4} )?(ATP Challenger|WTA 125K|125k|Challenger|ATP|WTA|[MW]\d+(\+H)?)\s+", "", name or "", flags=re.I)
    city = re.sub(r"^(125k|Challenger)\s+", "", city, flags=re.I)
    city = re.sub(r"\s+\(moved.*$", "", city)
    return re.sub(r"\s+(\d+|II|III)$", "", city).strip().lower()


def tier_of(row, name):
    if row["tour"].startswith("ITF"):
        found = re.match(r"([MW])(\d+)", name or "")
        return found.group(0) if found else "ITF other"
    if re.search(r"Open (Men|Women) Singles|Wimbledon", name or ""):
        return "Grand Slam"
    return row["tour"]


class Union:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def prepare():
    """Stage, tournament instance, region, player keys and walk-forward player history for every stable row."""
    rows = sorted(json.loads((OUT/"rows.json").read_text()), key=lambda r: r["settle"])
    union = Union()
    for r in rows:
        for name, ident in ((r["p1"], r["id1"]), (r["p2"], r["id2"])):
            union.join(("name", (name or "").strip().lower()), ("id", ident) if ident else ("name", (name or "").strip().lower()))
    last, counter, instances = {}, 0, defaultdict(lambda: defaultdict(int))
    for r in rows:
        r["name"], r["stage"] = parse_stage(r)
        r["qual"] = bool(r["stage"]) and r["stage"].startswith("qual")
        r["city"] = city_of(r["name"])
        r["country"] = CITY.get(r["city"])
        r["region"] = REGION.get(r["country"], "unmapped")
        r["tier"] = tier_of(r, r["name"])
        key = (r["tour"], r["name"])
        if key not in last or r["settle"]-last[key][0] > 4*DAY:
            counter += 1
            last[key] = [r["settle"], counter]
        last[key][0] = r["settle"]
        r["inst"] = last[key][1]
        if r["stage"] and r["stage"][0] == "R":
            instances[r["inst"]][int(r["stage"][1:])] += 1
        r["k1"], r["k2"] = union.find(("name", (r["p1"] or "").strip().lower())), union.find(("name", (r["p2"] or "").strip().lower()))
        fav1 = r.get("yes", True)
        r["pay_fav"] = None
        if r.get("stable"):
            r["pay_fav"] = r["outcome"] if fav1 else 1-r["outcome"]
            r["kf"], r["kd"] = (r["k1"], r["k2"]) if fav1 else (r["k2"], r["k1"])
    for r in rows:
        sizes = instances.get(r["inst"], {})
        biggest = max(sizes) if sizes else None
        r["first_main"] = bool(biggest) and r["stage"] == f"R{biggest}" and sizes[biggest] >= 4
    # Player histories in settlement order.
    history = defaultdict(list)
    for r in rows:
        for key, pay, other in ((r["k1"], r["outcome"], r["k2"]), (r["k2"], r["outcome_other"], r["k1"])):
            price = None
            if r.get("stable"):
                price = r["start"] if key == r["kf"] else 100-r["start"]
            history[key].append((r["settle"], pay, price, LEVEL[r["tour"]], r["inst"], r["qual"], other))
    times = {k: np.array([e[0] for e in v]) for k, v in history.items()}
    for r in rows:
        if not r.get("stable"):
            continue
        priced_at = r["settle"]-6*3600
        begins = r.get("start_ts") or r["settle"]-2*3600
        for side, key, other in (("f", r["kf"], r["kd"]), ("d", r["kd"], r["kf"])):
            past = history[key][:int(np.searchsorted(times[key], priced_at))]
            h = dict(n=len(past))
            if past:
                h["days_idle"] = (begins-past[-1][0])/DAY
                h["last_seen"] = past[-1][0]
                h["rest_hours"] = (begins-past[-1][0])/3600
                priced = [e for e in past[-12:] if e[2] is not None and e[0] > priced_at-90*DAY][-5:]
                if len(priced) >= 3:
                    h["form"] = float(np.mean([e[1]-e[2]/100 for e in priced]))
                same = [e for e in past[-12:] if e[4] == r["inst"]]
                h["wins_here"] = sum(e[1] > .5 for e in same)
                h["qualified_here"] = any(e[5] for e in same)
                h["lost_qualifying_here"] = any(e[5] and e[1] < .5 for e in same)
                recent = [e for e in past if e[0] > priced_at-365*DAY]
                h["top_level"] = max((e[3] for e in recent), default=0)
                h["matches_above"] = sum(e[3] > LEVEL[r["tour"]] for e in recent)
                lp = next((e for e in reversed(past) if e[2] is not None), None)
                h["upset_last"] = bool(lp) and lp is past[-1] and lp[1] > .5 and lp[2] < 40
                h["lost_as_favourite_last"] = bool(lp) and lp is past[-1] and lp[1] < .5 and lp[2] > 60
                h["beat_opponent_before"] = any(e[6] == other and e[1] > .5 for e in past)
            r["h"+side] = h
    return rows


# ---------------------------------------------------------------- report: statistics

class Sample:
    """The stable matches as arrays. Everything is on the favourite: gap = 100 x payout - pre-match price."""

    def __init__(self, rows):
        self.rows = rows = [r for r in rows if r.get("stable") and 50 <= r["start"] < 99 and r["bid0"] is not None and r["ask0"] is not None]
        ids = {}
        self.price = np.array([r["start"] for r in rows])
        self.pay = np.array([r["pay_fav"] for r in rows])
        self.gap = 100*self.pay-self.price
        self.recent = np.array([r["period"] == "recent" for r in rows])
        self.kf = np.array([ids.setdefault(r["kf"], len(ids)) for r in rows])
        self.kd = np.array([ids.setdefault(r["kd"], len(ids)) for r in rows])
        ask_f = np.array([r["ask0"] for r in rows], float)
        ask_d = 100-np.array([r["bid0"] for r in rows], float)
        self.net_f = 100*self.pay-ask_f-7*ask_f/100*(1-ask_f/100)
        self.net_d = 100*(1-self.pay)-ask_d-7*ask_d/100*(1-ask_d/100)
        self.dip = np.array([r["start"] >= 90 and r["low_in_play"] is not None and r["low_in_play"] < 80 for r in rows])
        self.edge = np.array([r["tour"] in EDGE for r in rows])
        self.tour = np.array([r["tour"] for r in rows])

    def band(self, lo, hi):
        return (self.price >= lo) & (self.price < hi)


def boot(values, clusters, draws=1000):
    """Mean and a 95% interval resampling clusters (players)."""
    if len(values) < 10:
        return None
    _, inverse = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inverse, values), np.bincount(inverse)
    pick = RNG.integers(0, len(sums), (draws, len(sums)))
    means = sums[pick].sum(1)/counts[pick].sum(1)
    return [round(float(v), 2) for v in np.percentile(means, [2.5, 97.5])]


def contrast(y, x, clusters, price=None):
    """Mean of y where x minus mean where not x. The standard error is the larger of the cluster-robust one and,
    when prices are given, the one the prices themselves imply (variance p(1-p)), so that a small group with no
    losses does not get a standard error near zero."""
    x = x.astype(float)
    if x.sum() < 2 or (1-x).sum() < 2:
        return None, None
    diff = y[x == 1].mean()-y[x == 0].mean()
    resid = y-np.where(x == 1, y[x == 1].mean(), y[x == 0].mean())
    centred = x-x.mean()
    _, inverse = np.unique(clusters, return_inverse=True)
    score = np.bincount(inverse, centred*resid)
    se = float(np.sqrt((score**2).sum())/(centred**2).sum())
    if price is not None:
        var = price*(100-price)
        se = max(se, float(np.sqrt(var[x == 1].sum()/x.sum()**2+var[x == 0].sum()/(1-x).sum()**2)))
    return float(diff), se


def ols(y, X, clusters):
    """Coefficients and cluster-robust standard errors."""
    XtX = np.linalg.pinv(X.T@X)
    beta = XtX@X.T@y
    resid = y-X@beta
    _, inverse = np.unique(clusters, return_inverse=True)
    scores = np.zeros((inverse.max()+1, X.shape[1]))
    np.add.at(scores, inverse, X*resid[:, None])
    cov = XtX@(scores.T@scores)@XtX
    return beta, np.sqrt(np.diag(cov))


def describe(s, mask, clusters, side="f"):
    """Gap, win rate and the taker trade for a set of matches; side 'd' scores buying the underdog."""
    n = int(mask.sum())
    if n == 0:
        return dict(n=0)
    net = s.net_f if side == "f" else s.net_d
    out = dict(n=n, price=round(float(s.price[mask].mean()), 1), won=round(float(100*s.pay[mask].mean()), 1),
               gap=round(float(s.gap[mask].mean()), 2), gap_ci=boot(s.gap[mask], clusters[mask]),
               take=round(float(net[mask].mean()), 2), take_ci=boot(net[mask], clusters[mask]), take_side="favourite" if side == "f" else "underdog")
    fills = mask & s.dip
    if fills.sum():
        out["dip_fills"] = int(fills.sum())
        out["dip_net"] = round(float((100*s.pay[fills]-80).mean()), 2)
        out["dip_ci"] = boot(100*s.pay[fills]-80, clusters[fills])
    return out


def band_text(ci):
    return "      n/a     " if not ci else f"[{ci[0]:+6.2f},{ci[1]:+6.2f}]"


def show(label, d):
    if not d.get("n"):
        return f"    {label:44} none"
    text = (f'    {label:44} n {d["n"]:5} price {d["price"]:4.1f} won {d["won"]:5.1f}% gap {d["gap"]:+6.2f} {band_text(d["gap_ci"])} '
            f'| buy {d["take_side"][:3]} at ask+fee {d["take"]:+6.2f} {band_text(d["take_ci"])}')
    if d.get("dip_fills"):
        text += f' | 80c dip fills {d["dip_fills"]:3} net {d["dip_net"]:+6.2f} {band_text(d.get("dip_ci"))}'
    return text


# ---------------------------------------------------------------- report: contexts

def contexts(s):
    """Every context tested: name -> (flag, defined-where mask, whose player the clusters are, section)."""
    rows = s.rows
    arr = lambda f: np.array([bool(f(r)) for r in rows])
    hf = lambda r, k, d=None: r["hf"].get(k, d)
    hd = lambda r, k, d=None: r["hd"].get(k, d)
    every = np.ones(len(rows), bool)
    main = arr(lambda r: r["stage"] is not None and not r["qual"])
    known = arr(lambda r: r["stage"] is not None)
    seasoned = arr(lambda r: r["settle"] >= HISTORY)
    covered = arr(lambda r: r["settle"] >= COVERAGE+45*DAY)
    began = arr(lambda r: r.get("start_ts") is not None)
    out = {}

    def add(section, name, flag, where=every, who="f"):
        out[name] = (flag & where, where, who, section)

    # 1. the round and what might stand behind it
    add(1, "qualifying round (any)", arr(lambda r: r["qual"]), known)
    add(1, "qualifying, first round", arr(lambda r: r["stage"] in ("qual Q1", "qual Q")), known)
    add(1, "qualifying, final round", arr(lambda r: r["stage"] in ("qual QF", "qual Q2")), known)
    add(1, "main draw, first round", arr(lambda r: r["first_main"]), known)
    add(1, "main draw, quarterfinal on", arr(lambda r: r["stage"] in ("QF", "SF", "F")), known)
    add(1, "ITF round of 64", arr(lambda r: r["stage"] == "R64" and r["tour"].startswith("ITF")), arr(lambda r: r["tour"].startswith("ITF")))
    add(1, "first round, underdog came through qualifying", arr(lambda r: r["first_main"] and hd(r, "qualified_here")), main, "d")
    add(1, "first round, favourite came through qualifying", arr(lambda r: r["first_main"] and hf(r, "qualified_here")), main)
    add(1, "first round, underdog new to the books (wildcard-like)",
        arr(lambda r: r["first_main"] and r["hd"]["n"] <= 2 and not hd(r, "qualified_here")), main & seasoned, "d")
    add(1, "favourite has played a higher tour in the past year", arr(lambda r: hf(r, "matches_above", 0) >= 3), seasoned)
    add(1, "favourite higher-tour player, underdog never", arr(lambda r: hf(r, "matches_above", 0) >= 3 and hd(r, "matches_above", 0) == 0), seasoned)
    add(1, "Saturday to Monday", arr(lambda r: r.get("weekday") in (5, 6, 0)), began)
    add(1, "start before 09:00 UTC", arr(lambda r: r.get("hour_utc", 12) < 9), began)
    add(1, "start 18:00 UTC or later", arr(lambda r: r.get("hour_utc", 12) >= 18), began)
    # Attention: thirds of the open interest before play, cut within each tour on its holdout matches.
    oi = np.array([r.get("open_interest") if r.get("open_interest") is not None else np.nan for r in rows])
    low, high = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
    for tour in sorted(set(s.tour)):
        part = (s.tour == tour) & began & ~s.recent
        base = part if part.sum() >= 50 else (s.tour == tour) & began
        cuts = np.nanpercentile(oi[base], [100/3, 200/3])
        low |= (s.tour == tour) & (oi <= cuts[0])
        high |= (s.tour == tour) & (oi > cuts[1])
    add(1, "low open interest before play (tour's bottom third)", low, began)
    add(1, "high open interest before play (tour's top third)", high, began)
    # 2. the players, walk-forward
    for side, h, who, label in (("f", hf, "f", "favourite"), ("d", hd, "d", "underdog")):
        formed = arr(lambda r: h(r, "form") is not None)
        add(2, f"{label} beat its price lately (form > +0.15)", arr(lambda r: (h(r, "form") or 0) > .15), formed, who)
        add(2, f"{label} missed its price lately (form < -0.15)", arr(lambda r: (h(r, "form") or 0) < -.15), formed, who)
        add(2, f"{label} new to the books (0-2 prior markets)", arr(lambda r: r["h"+side]["n"] <= 2), seasoned, who)
        add(2, f"{label} established (30+ prior markets)", arr(lambda r: r["h"+side]["n"] >= 30), seasoned, who)
        add(2, f"{label} came through qualifying (any main-draw round)", arr(lambda r: h(r, "qualified_here")), main, who)
        idle = arr(lambda r: r["h"+side]["n"] >= 3 and h(r, "last_seen", 0) >= COVERAGE)
        add(2, f"{label} back after 30+ days", arr(lambda r: h(r, "days_idle", 0) >= 30), idle & covered, who)
        add(2, f"{label} back after 60+ days", arr(lambda r: h(r, "days_idle", 0) >= 60), idle & covered, who)
        rested = arr(lambda r: r["h"+side]["n"] >= 1)
        add(2, f"{label} played within 24 hours", arr(lambda r: h(r, "rest_hours", 1e9) < 24), rested, who)
        add(2, f"{label} played within 12 hours", arr(lambda r: h(r, "rest_hours", 1e9) < 12), rested, who)
        add(2, f"{label} lost as a favourite last time out", arr(lambda r: h(r, "lost_as_favourite_last")), rested, who)
    both = arr(lambda r: hf(r, "form") is not None and hd(r, "form") is not None)
    add(2, "favourite's form at least 0.2 above the underdog's", arr(lambda r: (hf(r, "form") or 0)-(hd(r, "form") or 0) > .2), both)
    add(2, "favourite's form at least 0.2 below the underdog's", arr(lambda r: (hf(r, "form") or 0)-(hd(r, "form") or 0) < -.2), both)
    # 3. tournaments
    for region in REGIONS:
        add(3, f"region {region}", arr(lambda r: r["region"] == region), arr(lambda r: r["region"] != "unmapped"))
    for code in ("USA", "CHN", "ITA", "FRA", "ESP", "TUN", "EGY", "TUR", "BRA"):
        add(3, f"country {code}", arr(lambda r: r["country"] == code), arr(lambda r: r["region"] != "unmapped"))
    add(3, "ITF resort circuit (Monastir, Sharm, Antalya...)", arr(lambda r: r["city"] in RESORTS), arr(lambda r: r["tour"].startswith("ITF")))
    month = lambda r: datetime.fromtimestamp(r["settle"], timezone.utc).month
    north = arr(lambda r: r["region"] in ("W Europe", "E Europe+Turkey", "N America"))
    add(3, "indoor season (Oct-Mar, Europe and N America)", arr(lambda r: month(r) in (10, 11, 12, 1, 2, 3)) & north, north)
    add(3, "November (end of season)", arr(lambda r: month(r) == 11), s.recent.copy())
    add(3, "weekend", arr(lambda r: r.get("weekday") in (5, 6)), began)
    add(3, "Grand Slam", arr(lambda r: r["tier"] == "Grand Slam"), arr(lambda r: LEVEL[r["tour"]] == 3))
    add(3, "ITF lowest tier (M15/W15)", arr(lambda r: r["tier"] in ("M15", "W15")), arr(lambda r: r["tour"].startswith("ITF")))
    # 4. underdogs
    add(4, "underdog the higher-profile name (more higher-tour matches)",
        arr(lambda r: hd(r, "matches_above", 0) >= 3 and hd(r, "matches_above", 0) > hf(r, "matches_above", 0)), seasoned, "d")
    add(4, "underdog has far more Kalshi history (3x and 15+)", arr(lambda r: r["hd"]["n"] >= 15 and r["hd"]["n"] >= 3*max(1, r["hf"]["n"])), seasoned, "d")
    add(4, "underdog has 2+ wins in this tournament", arr(lambda r: hd(r, "wins_here", 0) >= 2), known, "d")
    add(4, "underdog has 3+ wins in this tournament", arr(lambda r: hd(r, "wins_here", 0) >= 3), known, "d")
    add(4, "underdog has more wins here than the favourite", arr(lambda r: hd(r, "wins_here", 0) > hf(r, "wins_here", 0)), known, "d")
    add(4, "underdog is a lucky loser (lost in qualifying here)", arr(lambda r: hd(r, "lost_qualifying_here") and not r["qual"]), main, "d")
    add(4, "underdog won as a big underdog last time out", arr(lambda r: hd(r, "upset_last")), arr(lambda r: r["hd"]["n"] >= 1), "d")
    add(4, "underdog has beaten this favourite before", arr(lambda r: hd(r, "beat_opponent_before")), seasoned, "d")
    add(4, "favourite has beaten this underdog before", arr(lambda r: hf(r, "beat_opponent_before")), seasoned)
    return out


def screen(s, ctx):
    """Every context x band: gap inside, the contrast with the rest of the band, per period and pooled."""
    cells = []
    for name, (flag, where, who, section) in ctx.items():
        clusters = s.kf if who == "f" else s.kd
        for band, lo, hi in BANDS:
            base = s.band(lo, hi) & where
            cell = dict(context=name, band=band, section=section, who=who)
            for period, pm in (("holdout", ~s.recent), ("recent", s.recent), ("both", np.ones(len(s.gap), bool))):
                m = base & pm
                inside = m & flag
                diff, se = contrast(s.gap[m], flag[m], clusters[m], s.price[m]) if inside.sum() >= 30 and (m & ~flag).sum() >= 30 else (None, None)
                cell[period] = dict(n=int(inside.sum()), gap=float(s.gap[inside].mean()) if inside.sum() else None, diff=diff, se=se,
                                    z=None if not se else diff/se)
            cells.append(cell)
    return cells


def shrink(cells):
    """Normal-normal shrinkage of the pooled contrasts towards zero; the spread of true effects is estimated from all cells."""
    use = [c for c in cells if c["both"]["se"]]
    d = np.array([c["both"]["diff"] for c in use])
    se = np.array([c["both"]["se"] for c in use])
    w = 1/se**2
    tau2 = max(0.0, float((w*(d**2-se**2)).sum()/w.sum()))
    for c, di, si in zip(use, d, se):
        c["shrunk"] = float(di*tau2/(tau2+si**2))
        c["shrunk_se"] = float(np.sqrt(tau2*si**2/(tau2+si**2))) if tau2 else 0.0
    return tau2**.5


def cell_line(c):
    parts = []
    for period in ("holdout", "recent", "both"):
        p = c[period]
        parts.append(f'{period[:4]} n {p["n"]:5} gap {p["gap"] if p["gap"] is not None else float("nan"):+6.2f} '
                     f'vs rest {p["diff"] if p["diff"] is not None else float("nan"):+6.2f} z {p["z"] if p["z"] is not None else float("nan"):+5.2f}')
    return f'    {c["context"][:52]:52} {c["band"]:6} ' + " | ".join(parts) + (f' | shrunk {c["shrunk"]:+5.2f}' if "shrunk" in c else "")


def passes(c, z=1.96):
    a, b = c["holdout"]["z"], c["recent"]["z"]
    return a is not None and b is not None and a*b > 0 and abs(a) > z and abs(b) > z


# ---------------------------------------------------------------- report: retirements

def plain(text):
    return "".join(ch for ch in unicodedata.normalize("NFKD", (text or "").lower()) if ch.isalpha() or ch == " ")


def match_status(rows):
    """Finished / Retired / Walk Over from the tennis provider's fixtures on disk (ATP July-October, ATP and Challenger
    30 September-16 November), joined by the two surnames and the day."""
    truth = []
    for path in sorted((ROOT/"data/raw/break-hold-fixtures").glob("*.json")):
        truth += [(m["date"], m["players"][0], m["players"][1], m["match_status"]) for m in json.loads(path.read_text())]
    for path in sorted((ROOT/"data/raw/native-atp").glob("*.json")):
        data = json.loads(path.read_text())
        data = data.get("result", []) if isinstance(data, dict) else data
        truth += [(m["event_date"], m["event_first_player"], m["event_second_player"], m["event_status"]) for m in data]
    index = defaultdict(list)
    for day, a, b, status in truth:
        if a and b and plain(a).split() and plain(b).split():
            index[frozenset((plain(a).split()[-1], plain(b).split()[-1]))].append((datetime.strptime(day, "%Y-%m-%d").date(), status))
    found = 0
    for r in rows:
        if r["tour"] not in ("ATP", "Challenger") or not plain(r["p1"]).split() or not plain(r["p2"]).split():
            continue
        day = datetime.fromtimestamp(r["close"], timezone.utc).date()
        for when, status in index.get(frozenset((plain(r["p1"]).split()[-1], plain(r["p2"]).split()[-1])), ()):
            if abs((when-day).days) <= 1:
                r["status"] = status
                found += 1
                break
    return found


def retirements(s, all_rows, report):
    print("\n(5) RETIREMENTS AND WALKOVERS")
    matched = match_status(all_rows)
    out = report["5_retirements"] = {}
    print(f"  provider status joined to {matched} ATP and Challenger matches (the only tours with a status on disk)")
    # Walkovers and cancellations: a fractional settlement.
    print("  fractional settlements (the match never started: walkover or cancellation), all listed matches")
    for tour in TOURS.values():
        part = [r for r in all_rows if r["tour"] == tour]
        frac = [r for r in part if 0 < r["outcome"] < 1]
        half = sum(r["outcome"] == .5 for r in frac)
        priced = [r for r in frac if r.get("stable")]
        move = [abs(100*r["pay_fav"]-r["start"]) for r in priced]
        out[f"fractional | {tour}"] = dict(listed=len(part), fractional=len(frac), at_50=half, with_stable_price=len(priced),
                                           mean_abs_gap=float(np.mean(move)) if move else None)
        print(f"    {tour:15} listed {len(part):6} fractional {len(frac):4} ({100*len(frac)/len(part):.2f}%) settled at exactly 0.50: {half:4} "
              f"| with a stable pre-match price {len(priced):4}, mean |settlement - price| {np.mean(move) if move else float('nan'):5.1f}c")
    for label, lo, hi in BANDS[:-1]:
        for tours, name in ((("ITF men", "ITF women"), "ITF (walkover pays 0.50)"), (("ATP", "WTA", "Challenger", "WTA Challenger"), "other tours (fair price)")):
            m = s.band(lo, hi) & np.isin(s.tour, tours)
            frac = m & (s.pay > 0) & (s.pay < 1)
            if m.sum():
                cost = float((s.gap[frac]).sum()/m.sum())
                out[f"walkover cost | {name} | {label}"] = dict(matches=int(m.sum()), fractional=int(frac.sum()), gap_contribution=cost)
                print(f"    {name:26} favourite {label}: {int(frac.sum()):3} of {int(m.sum()):5} priced matches never started; "
                      f"they add {cost:+.2f}c to the favourite's gap")
    have = np.array(["status" in r for r in s.rows])
    status = np.array([r.get("status", "") for r in s.rows])
    retired, finished = have & (status == "Retired"), have & (status == "Finished")
    print("  retirements after play began (provider status), matches with a stable pre-match price")
    for label, mask in (("all", have), ("ATP", have & (s.tour == "ATP")), ("Challenger", have & (s.tour == "Challenger")),
                        ("qualifying", have & np.array([r["qual"] for r in s.rows])),
                        ("main draw first round", have & np.array([r["first_main"] for r in s.rows])),
                        ("main draw later", have & np.array([not r["qual"] and not r["first_main"] for r in s.rows])),
                        ("holdout", have & ~s.recent), ("recent", have & s.recent)):
        n, k = int((mask & (retired | finished)).sum()), int((mask & retired).sum())
        if not n:
            continue
        fav_won = float(s.pay[mask & retired].mean()) if k else float("nan")
        out[f"rate | {label}"] = dict(matches=n, retired=k, rate=k/n, favourite_won_share=fav_won,
                                      mean_price=float(s.price[mask & retired].mean()) if k else None)
        print(f"    {label:22} matches {n:5} retired {k:4} ({100*k/n:.1f}%) | favourite won {100*fav_won:5.1f}% of them at a mean price of "
              f"{s.price[mask & retired].mean() if k else float('nan'):.1f}c")
    print("  does the favourite's gap come from underdog retirements? gap with and without the retired matches")
    for label, lo, hi in BANDS:
        m = s.band(lo, hi) & (retired | finished)
        r = m & retired
        if m.sum() < 30:
            continue
        a, b, c = describe(s, m, s.kf), describe(s, m & finished, s.kf), describe(s, r, s.kf)
        contribution = float(s.gap[r].sum()/m.sum()) if r.sum() else 0.0
        out[f"gap | {label}"] = dict(all=a, finished_only=b, retired_only=c, contribution_of_retirements=contribution)
        print(f"    favourite {label:6} all n {a['n']:5} gap {a['gap']:+6.2f} {band_text(a['gap_ci'])} | finished only {b['gap']:+6.2f} {band_text(b['gap_ci'])} "
              f"| retired n {c.get('n', 0):3} gap {c.get('gap', float('nan')):+6.2f} {band_text(c.get('gap_ci'))} | retirements contribute {contribution:+.2f}c")
    print("  a pre-match tell? retired against finished matches (favourite's drift from the warm-up to the last quote before play, money in the book)")
    drift = np.array([(r.get("price_before_play") or np.nan)-r["start"] if r.get("price_before_play") is not None else np.nan for r in s.rows])
    money = np.array([r.get("open_interest") if r.get("open_interest") is not None else np.nan for r in s.rows])
    volume = np.array([r.get("prematch_volume") if r.get("prematch_volume") is not None else np.nan for r in s.rows])
    for name, values in (("drift (cents)", drift), ("abs drift (cents)", np.abs(drift)), ("open interest before play", money), ("pre-match volume", volume)):
        a, b = values[retired & ~np.isnan(values)], values[finished & ~np.isnan(values)]
        if len(a) < 10:
            continue
        rank = np.argsort(np.argsort(np.concatenate([a, b])))
        auc = (rank[:len(a)].sum()-len(a)*(len(a)-1)/2)/(len(a)*len(b))
        sd = np.sqrt((len(a)+len(b)+1)/(12*len(a)*len(b)))
        out[f"tell | {name}"] = dict(retired_mean=float(a.mean()), finished_mean=float(b.mean()), retired_median=float(np.median(a)),
                                     finished_median=float(np.median(b)), auc=float(auc), z=float((auc-.5)/sd))
        print(f"    {name:28} retired mean {a.mean():10.2f} median {np.median(a):9.2f} | finished mean {b.mean():10.2f} median {np.median(b):9.2f} "
              f"| AUC {auc:.3f} (z {(auc-.5)/sd:+.2f})")
    # Who retires: the pre-match underdog or the favourite, and does a drifting favourite retire more often?
    k = retired.sum()
    if k:
        dog_quit = float(s.pay[retired].mean())
        fair = float(s.price[retired].mean()/100)
        se = float(np.sqrt((s.price[retired]/100*(1-s.price[retired]/100)).sum())/k)
        out["who retires"] = dict(retired=int(k), underdog_retired_share=dog_quit, share_expected_from_price=fair, z=(dog_quit-fair)/se)
        print(f"    of {int(k)} retirements the underdog was the one to stop in {100*dog_quit:.1f}%; the favourites' prices implied {100*fair:.1f}% "
              f"(z {(dog_quit-fair)/se:+.2f})")
    print("  retirement rate by what was known before the match (either player)")
    either = lambda key, test: np.array([test(r["hf"].get(key)) or test(r["hd"].get(key)) for r in s.rows])
    known = {"a player back after 30+ days": either("days_idle", lambda v: v is not None and v >= 30),
             "a player on court within 24 hours": either("rest_hours", lambda v: v is not None and v < 24),
             "a player with 3+ wins in this tournament": either("wins_here", lambda v: v is not None and v >= 3),
             "a player who came through qualifying": either("qualified_here", bool),
             "favourite drifted down 2c+ before play": drift <= -2, "favourite 80c+": s.price >= 80}
    played = retired | finished
    for name, flag in known.items():
        a, b = played & flag, played & ~flag
        if a.sum() < 30:
            continue
        ra, rb = retired[a].mean(), retired[b].mean()
        pooled_rate = retired[played].mean()
        z = (ra-rb)/np.sqrt(pooled_rate*(1-pooled_rate)*(1/a.sum()+1/b.sum()))
        out[f"rate by | {name}"] = dict(matches=int(a.sum()), rate=float(ra), rate_otherwise=float(rb), z=float(z))
        print(f"    {name:42} matches {int(a.sum()):5} retired {100*ra:4.1f}% | otherwise {100*rb:4.1f}% (z {z:+.2f})")
    dipped = have & s.dip
    if dipped.sum():
        print(f"    80c dip fills with a status: {int(dipped.sum())}, of which retired {int((dipped & retired).sum())} "
              f"(favourite won {int(s.pay[dipped & retired].sum())}); net/fill all {(100*s.pay[dipped]-80).mean():+.2f}, "
              f"finished only {(100*s.pay[dipped & finished]-80).mean():+.2f}")
        out["dip fills"] = dict(fills=int(dipped.sum()), retired=int((dipped & retired).sum()), favourite_won_retired=float(s.pay[dipped & retired].sum()),
                                net_all=float((100*s.pay[dipped]-80).mean()), net_finished=float((100*s.pay[dipped & finished]-80).mean()))


# ---------------------------------------------------------------- report: follow-ups on what the screen turned up

def followups(s, out):
    """Looked at AFTER the screen, so these are descriptions of the screen's leads, not fresh tests (13 extra tables)."""
    rows = s.rows
    both = np.ones(len(rows), bool)
    periods = (("holdout", ~s.recent), ("recent", s.recent), ("both", both))
    itf = np.isin(s.tour, ("ITF men", "ITF women"))

    def rung(r):
        if not r["stage"]:
            return "unknown"
        if r["qual"]:
            return "1 qualifying, first round" if r["stage"] in ("qual Q1", "qual Q") else "2 qualifying, later round"
        if r["stage"] in ("QF", "SF", "F"):
            return "5 quarterfinal on"
        return "3 main draw, first round" if r["first_main"] else "4 main draw, before the quarterfinal"
    ladder = np.array([rung(r) for r in rows])
    print("\n(7) FOLLOW-UPS (descriptive; chosen after seeing the screen)")
    print("  (7a) the favourite's gap down the draw")
    for band, lo, hi in BANDS[3:5]:
        for scope, sm in (("all tours", both), ("ATP, WTA, Challengers", ~itf), ("ITF", itf)):
            for value in sorted(set(ladder)-{"unknown"}):
                for period, pm in periods:
                    d = describe(s, s.band(lo, hi) & sm & pm & (ladder == value), s.kf)
                    out[f"7a | {band} | {scope} | {value} | {period}"] = d
                    if d["n"] >= 25 and (period == "both" or scope == "all tours"):
                        print(show(f"{band} {scope[:12]} {value[2:]} {period}", d))
    print("  (7b) by how many matches the UNDERDOG has already won in this tournament, favourite 90-99c and all prices")
    wins = np.array([min(3, r["hd"].get("wins_here", 0)) for r in rows])
    for band, lo, hi in (BANDS[4], BANDS[5]):
        for k in range(4):
            for period, pm in periods:
                d = describe(s, s.band(lo, hi) & pm & (wins == k), s.kd, side="d" if k >= 2 else "f")
                out[f"7b | {band} | underdog wins {k} | {period}"] = d
                print(show(f"{band} underdog has won {k}{'+' if k == 3 else ''} here, {period}", d))
    print("  (7c) open interest before play, thirds within the tour, favourite 90-99c; and crossed with first-round qualifying")
    ctx = contexts(s)
    low, high = ctx["low open interest before play (tour's bottom third)"][0], ctx["high open interest before play (tour's top third)"][0]
    began = ctx["low open interest before play (tour's bottom third)"][1]
    third = np.where(low, "low", np.where(high, "high", "mid"))
    q1 = ladder == "1 qualifying, first round"
    for scope, sm in (("all tours", both), ("ATP, WTA, Challengers", ~itf), ("ITF", itf), ("first-round qualifying", q1), ("not first-round qualifying", ~q1),
                      ("not Q1, ATP/WTA/Challengers", ~q1 & ~itf)):
        for value in ("low", "mid", "high"):
            for period, pm in periods:
                d = describe(s, s.band(90, 99) & began & sm & pm & (third == value), s.kf)
                out[f"7c | {scope} | {value} | {period}"] = d
                if d["n"] >= 25:
                    print(show(f"{scope[:27]} OI {value} {period}", d))
    print("  (7d) first-round qualifying: the same trade at the last quote before play instead of the warm-up quote")
    ask = np.array([r.get("ask_before_play") if r.get("ask_before_play") is not None else np.nan for r in rows], float)
    mid_now = np.array([r.get("price_before_play") if r.get("price_before_play") is not None else np.nan for r in rows], float)
    net = 100*s.pay-ask-7*ask/100*(1-ask/100)
    for band, lo, hi in BANDS[3:5]:
        for period, pm in periods:
            m = q1 & pm & (mid_now >= lo) & (mid_now < hi) & ~np.isnan(ask)
            if m.sum() >= 20:
                out[f"7d | {band} | {period}"] = dict(n=int(m.sum()), ask=float(ask[m].mean()), won=float(s.pay[m].mean()), net=float(net[m].mean()), ci=boot(net[m], s.kf[m]))
                print(f"    favourite {band} before play, {period:8} n {int(m.sum()):4} ask {ask[m].mean():5.1f} won {100*s.pay[m].mean():5.1f}% "
                      f"net after fee {net[m].mean():+6.2f} {band_text(boot(net[m], s.kf[m]))}")
    print("  (7e) first-round qualifying by tour and by quarter, favourite 90-99c and 80-90c")
    quarter = np.array([f'Q{(datetime.fromtimestamp(r["settle"], timezone.utc).month-1)//3+1}' for r in rows])
    for band, lo, hi in BANDS[3:5]:
        for name, sm in [(t, s.tour == t) for t in ("ATP", "WTA", "Challenger", "WTA Challenger")]+[(q, quarter == q) for q in ("Q1", "Q2", "Q3", "Q4")]:
            d = describe(s, s.band(lo, hi) & q1 & sm, s.kf)
            out[f"7e | {band} | {name}"] = d
            print(show(f"{band} Q1 qualifying, {name}", d))
    print("  (7f) resting bids in play, by rung of the draw (strict fill: a later in-play minute prints below the bid; net per fill, no fee)")
    low_play = np.array([r["low_in_play"] if r["low_in_play"] is not None else 999 for r in rows], float)
    for band, lo, hi, bids in (("90-99", 90, 99, (85, 80, 75, 70)), ("80-90", 80, 90, (75, 70, 65))):
        for scope, sm in (("first-round qualifying", q1), ("every other match", ~q1 & ~itf), ("ITF (no qualifying listed)", itf)):
            for bid in bids:
                cells = []
                for period, pm in periods:
                    m = s.band(lo, hi) & sm & pm
                    fill = m & (low_play < bid)
                    nets = 100*s.pay[fill]-bid
                    cell = dict(orders=int(m.sum()), fills=int(fill.sum()), net=float(nets.mean()) if fill.sum() else None, ci=boot(nets, s.kf[fill]))
                    out[f"7f | {band} | {scope} | bid {bid} | {period}"] = cell
                    cells.append(f'{period[:4]} orders {cell["orders"]:4} fills {cell["fills"]:3} net {cell["net"] if cell["net"] is not None else float("nan"):+6.2f} {band_text(cell["ci"])}')
                print(f"    {band} {scope[:26]:26} bid {bid}: " + " | ".join(cells))
    print("  (7h) the 80c dip rule and the pre-match purchase split three ways, favourite 90-99c (the split was formed after the screen)")
    later_qual = ladder == "2 qualifying, later round"
    groups = (("first-round qualifying", q1), ("later qualifying rounds", later_qual), ("other, low open interest", ~q1 & ~later_qual & began & low),
              ("other, mid or high open interest", ~q1 & ~later_qual & began & ~low))
    for scope, sm in (("Challenger, WTA, WTA Challenger", s.edge), ("ATP", s.tour == "ATP"), ("ITF", itf), ("all tours", both)):
        for name, gm in groups:
            for period, pm in periods:
                d = describe(s, s.band(90, 99) & sm & gm & pm, s.kf)
                out[f"7h | {scope} | {name} | {period}"] = d
                if d["n"] >= 15:
                    print(show(f"{scope[:12]} {name[:22]} {period}", d))
    print("  (7g) 90c+ favourites from the quarterfinal on, by tour group; and the underdog trade")
    late = ladder == "5 quarterfinal on"
    for scope, sm in (("all tours", both), ("ATP, WTA, Challengers", ~itf), ("ITF men", s.tour == "ITF men"), ("ITF women", s.tour == "ITF women")):
        for period, pm in periods:
            d = describe(s, s.band(90, 99) & late & sm & pm, s.kd, side="d")
            out[f"7g | {scope} | {period}"] = d
            if d["n"] >= 10:
                print(show(f"quarterfinal on, {scope[:22]} {period}", d))


# ---------------------------------------------------------------- report

def report():
    all_rows = prepare()
    s = Sample(all_rows)
    out = {}
    both = np.ones(len(s.gap), bool)
    periods = (("holdout", ~s.recent), ("recent", s.recent), ("both", both))
    print(f"{len(all_rows)} listed matches, {len(s.rows)} with a stable pre-match price of 50-99c "
          f"(holdout {int((~s.recent).sum())}, recent {int(s.recent.sum())}); {len(set(s.kf) | set(s.kd))} players")

    print("\n(0) BASELINE: favourite's gap by price band and tour (cents; interval clustered by favourite)")
    for tour in ("all",)+tuple(TOURS.values()):
        for band, lo, hi in BANDS:
            for period, pm in periods:
                m = s.band(lo, hi) & pm & (both if tour == "all" else s.tour == tour)
                d = describe(s, m, s.kf)
                out[f"0 | {tour} | {band} | {period}"] = d
                if tour == "all" or band in ("80-90", "90-99"):
                    print(show(f"{tour} {band} {period}", d))

    print("\n(1) WHY QUALIFYING? stage of the draw, by band (all six tours unless stated)")
    stage = np.array([("qualifying" if r["qual"] else "first round" if r["first_main"] else "later main draw") if r["stage"] else "unknown" for r in s.rows])
    for band, lo, hi in BANDS[2:5]:
        for scope, sm in (("all tours", both), ("ATP+WTA", np.isin(s.tour, ("ATP", "WTA"))), ("Challenger+WTA Chall", np.isin(s.tour, ("Challenger", "WTA Challenger"))),
                          ("ITF", np.isin(s.tour, ("ITF men", "ITF women")))):
            for value in ("qualifying", "first round", "later main draw"):
                for period, pm in periods:
                    d = describe(s, s.band(lo, hi) & sm & pm & (stage == value), s.kf)
                    out[f"1 | stage | {scope} | {band} | {value} | {period}"] = d
                    if d["n"] >= 20:
                        print(show(f"{band} {scope[:20]} {value} {period}", d))
    print("  earlier work (favourite_gap.py) read 'Qualification Final' as a Final; the same split with that parser, 90-99c, traded tours:")
    old = np.array([("qualifying" if (r["round"] or "").startswith("Qualif") else "quarterfinal on" if r["round"] in ("Quarterfinal", "Quarterfinals", "Semifinal", "Semifinals", "Final", "Finals")
                     else "early main draw") for r in s.rows])
    for value in ("qualifying", "early main draw", "quarterfinal on"):
        for period, pm in periods:
            d = describe(s, s.band(90, 99) & s.edge & pm & (old == value), s.kf)
            out[f"1 | old parser | {value} | {period}"] = d
            print(show(f"old parser: {value} {period}", d))
    true_final = np.array([r["stage"] == "qual QF" for r in s.rows])
    print(f"    of the old parser's 'quarterfinal on' 90-99c matches on the traded tours, "
          f"{int((s.band(90, 99) & s.edge & (old == 'quarterfinal on') & true_final).sum())} of {int((s.band(90, 99) & s.edge & (old == 'quarterfinal on')).sum())} are qualifying finals")
    print("  what stands behind it? gap regressed on the round and its possible causes together (cents, cluster-robust by favourite)")
    ctx = contexts(s)
    names = ("qualifying round (any)", "main draw, first round", "low open interest before play (tour's bottom third)", "Saturday to Monday",
             "start before 09:00 UTC", "underdog new to the books (0-2 prior markets)", "favourite higher-tour player, underdog never")
    tours = sorted(set(s.tour))
    for band, lo, hi in BANDS[2:5]:
        for period, pm in periods:
            where = s.band(lo, hi) & pm
            for n in names:
                where &= ctx[n][1]
            if where.sum() < 200:
                continue
            X = np.column_stack([ctx[n][0][where].astype(float) for n in names]+[(s.tour[where] == t).astype(float) for t in tours])
            beta, se = ols(s.gap[where], X, s.kf[where])
            out[f"1 | regression | {band} | {period}"] = {n: dict(coef=float(b), se=float(e)) for n, b, e in zip(names, beta, se)}
            print(f"    {band} {period:8} n {int(where.sum()):5}: " + "; ".join(f"{n[:28]} {b:+.2f} (se {e:.2f})" for n, b, e in zip(names, beta, se)))
    print("  qualifying against the main draw inside strata, 90-99c, both periods (gap in qualifying, gap in main draw)")
    q, md = ctx["qualifying round (any)"][0], ctx["qualifying round (any)"][1] & ~ctx["qualifying round (any)"][0]
    strata = {f"tour {t}": s.tour == t for t in tours if not t.startswith("ITF")}
    for n in ("low open interest before play (tour's bottom third)", "high open interest before play (tour's top third)", "Saturday to Monday",
              "underdog new to the books (0-2 prior markets)", "favourite higher-tour player, underdog never"):
        strata[n] = ctx[n][0]
        strata["NOT "+n] = ctx[n][1] & ~ctx[n][0]
    for name, sm in strata.items():
        for period, pm in periods:
            m = s.band(90, 99) & sm & pm & ~np.isin(s.tour, ("ITF men", "ITF women"))
            a, b = describe(s, m & q, s.kf), describe(s, m & md, s.kf)
            out[f"1 | strata | {name} | {period}"] = dict(qualifying=a, main_draw=b)
            if a["n"] >= 15 and b["n"] >= 15:
                print(f"    {name[:50]:50} {period:8} qualifying n {a['n']:4} gap {a['gap']:+6.2f} {band_text(a['gap_ci'])} | main n {b['n']:4} gap {b['gap']:+6.2f} {band_text(b['gap_ci'])}")

    cells = screen(s, ctx)
    tau = shrink(cells)
    by = {(c["context"], c["band"]): c for c in cells}
    titles = {1: "(1b) ROUND, ATTENTION, SCHEDULING", 2: "(2) PLAYER EFFECTS, WALK-FORWARD", 3: "(3) TOURNAMENT EFFECTS", 4: "(4) UNDERDOGS"}
    for section in (1, 2, 3, 4):
        print(f"\n{titles[section]}: favourite's gap inside the context and against the rest of the same band (z cluster-robust)")
        for name, (flag, where, who, sec) in ctx.items():
            if sec != section:
                continue
            for band in ("all", "70-80", "80-90", "90-99") if section != 4 else ("all", "50-60", "60-70", "70-80", "80-90", "90-99"):
                c = by[name, band]
                if c["both"]["n"] >= 40:
                    print(cell_line(c))
    # Underdog trades in the underdog contexts and wherever the pooled contrast is negative.
    print("\n(4b) BUYING THE UNDERDOG at its ask plus the fee, in the contexts where the favourite's gap is most negative (pooled z < -1.5)")
    for c in sorted((c for c in cells if c["both"]["z"] is not None and c["both"]["z"] < -1.5), key=lambda c: c["both"]["z"]):
        flag, where, who, _ = ctx[c["context"]]
        lo, hi = next((lo, hi) for b, lo, hi in BANDS if b == c["band"])
        for period, pm in periods:
            d = describe(s, s.band(lo, hi) & where & flag & pm, s.kf if who == "f" else s.kd, side="d")
            out[f"4b | {c['context']} | {c['band']} | {period}"] = d
            print(show(f"{c['context'][:30]} {c['band']} {period}", d))

    retirements(s, all_rows, out)

    print("\n(6) MULTIPLICITY")
    testable = [c for c in cells if c["holdout"]["z"] is not None and c["recent"]["z"] is not None]
    hits = [c for c in testable if passes(c)]
    loose = [c for c in testable if passes(c, 1.0) and abs(c["both"]["z"]) > 1.96]
    pooled = [c for c in cells if c["both"]["z"] is not None]
    pv = sorted((2*stats.norm.sf(abs(c["both"]["z"])), i) for i, c in enumerate(pooled))
    last = max((rank for rank, (p, i) in enumerate(pv, 1) if p*len(pv)/rank < .10), default=0)
    bh = [pooled[i] for p, i in pv[:last]]
    big = [c for c in pooled if abs(c["both"]["z"]) > 1.96]
    out["6_multiplicity"] = dict(
        contexts=len(ctx), cells=len(cells), cells_testable_in_both_periods=len(testable), cells_testable_pooled=len(pooled),
        strict_pass=[(c["context"], c["band"]) for c in hits], strict_expected_by_chance=len(testable)*2*.025**2,
        loose_pass=[(c["context"], c["band"]) for c in loose],
        pooled_abs_z_over_1_96=len(big), pooled_expected_by_chance=.05*len(pooled), benjamini_hochberg_10pct=[(c["context"], c["band"]) for c in bh],
        sd_of_true_effects_cents=tau)
    print(f"  {len(ctx)} contexts x {len(BANDS)} bands = {len(cells)} cells; {len(pooled)} large enough to test pooled, {len(testable)} in both periods")
    print(f"  pooled |z| > 1.96: {len(big)} cells against {.05*len(pooled):.1f} expected by chance (bands overlap with 'all', so cells are not independent)")
    print(f"  same sign and |z| > 1.96 in BOTH periods: {len(hits)} cells against {len(testable)*2*.025**2:.2f} expected by chance")
    print(f"  Benjamini-Hochberg at 10% on the pooled z: {len(bh)} cells")
    print(f"  estimated spread (sd) of true context effects across all cells: {tau:.2f}c")
    print("  the 25 largest pooled contrasts, with the shrunk estimate:")
    for c in sorted(pooled, key=lambda c: -abs(c["both"]["z"]))[:25]:
        print(cell_line(c)+("  PASS both periods" if passes(c) else "")+("  BH" if c in bh else ""))
    print("  tradeable version of every cell that passes both periods, or BH, or (same sign, |z|>1 in both and pooled |z|>1.96):")
    keep = {id(c): c for c in hits+bh+loose}
    out["6_survivors"] = {}
    for c in sorted(keep.values(), key=lambda c: -abs(c["both"]["z"])):
        flag, where, who, _ = ctx[c["context"]]
        lo, hi = next((lo, hi) for b, lo, hi in BANDS if b == c["band"])
        side = "f" if c["both"]["diff"] > 0 else "d"
        print(cell_line(c))
        for period, pm in periods:
            m = s.band(lo, hi) & where & flag & pm
            d = describe(s, m, s.kf if who == "f" else s.kd, side=side)
            rest = s.band(lo, hi) & where & ~flag & pm & s.dip
            if rest.sum():
                d["dip_rest_fills"], d["dip_rest_net"] = int(rest.sum()), round(float((100*s.pay[rest]-80).mean()), 2)
            out["6_survivors"][f"{c['context']} | {c['band']} | {period}"] = d
            print(show(f"  {period}", d)+(f' (rest: {d["dip_rest_fills"]} fills {d["dip_rest_net"]:+.2f})' if "dip_rest_fills" in d and d.get("dip_fills") else ""))
    out["cells"] = cells

    print("\n(6b) THE 80c DIP RULE BY CONTEXT (favourite 90c+, traded tours Challenger, WTA, WTA Challenger; net per fill, cents)")
    dip_tests = []
    for name, (flag, where, who, section) in ctx.items():
        line = []
        for period, pm in periods:
            m = s.dip & s.edge & where & pm
            a, b = 100*s.pay[m & flag]-80, 100*s.pay[m & ~flag]-80
            cell = dict(fills_in=len(a), net_in=float(a.mean()) if len(a) else None, ci_in=boot(a, s.kf[m & flag]), fills_out=len(b), net_out=float(b.mean()) if len(b) else None)
            if len(a) >= 15 and len(b) >= 15:
                diff, se = contrast(100*s.pay[m]-80, flag[m], s.kf[m])
                cell.update(diff=diff, z=diff/se if se else None)
            out[f"6b | {name} | {period}"] = cell
            line.append((period, cell))
        if line[2][1]["fills_in"] >= 25 and line[2][1].get("z") is not None:
            dip_tests.append((name, line))
    for name, line in sorted(dip_tests, key=lambda t: -abs(t[1][2][1]["z"])):
        print(f"    {name[:52]:52} " + " | ".join(
            f'{p[:4]} in {c["fills_in"]:3} {c["net_in"] if c["net_in"] is not None else float("nan"):+6.2f} out {c["fills_out"]:3} '
            f'{c["net_out"] if c["net_out"] is not None else float("nan"):+6.2f} z {c.get("z") if c.get("z") is not None else float("nan"):+5.2f}' for p, c in line))
    n = len(dip_tests)
    strong = sum(abs(l[2][1]["z"]) > 1.96 for _, l in dip_tests)
    print(f"  {n} contexts had 25+ dip fills; pooled |z| > 1.96 in {strong}, against {.05*n:.1f} expected by chance")
    out["6b_summary"] = dict(contexts_tested=n, pooled_abs_z_over_1_96=strong, expected_by_chance=.05*n)
    followups(s, out)
    save(OUT/"summary.json", json.loads(json.dumps(out, default=float).replace("NaN", "null")))
    print(f"\nwrote {OUT/'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=("build", "report"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for step in args.steps:
        if step == "build":
            build(args.workers)
        else:
            report()
