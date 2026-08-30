"""League strength reference data.

Three things live here:

* NATION -- UEFA country coefficients for all 54 member associations, taken from
  the CoeffY2 export ("Coef. 24/25", the column the original 34-league table
  matched exactly on 31 of 34 nations). One source, one season, one scale.

* NON_UEFA -- leagues outside UEFA, which by definition have no coefficient.
  Their strengths are ESTIMATES, each placed alongside the UEFA nation it most
  resembles. These are the least trustworthy numbers in this file. Run
  "python main.py audit <Position>" to see what they are doing to your data.

* DIVISIONS -- maps FM division names to (nation, tier). A second tier is not a
  separate country, it is a fraction of its parent's strength, so mapping by
  tier covers many divisions without inventing a coefficient for each.

avg_ca reports how a player compares with their league and is never a model
feature. Values for the original 34 leagues are as supplied; the rest are
derived from the fit those 34 establish (R2 0.912).
"""
import math
import unicodedata
from typing import NamedTuple

# Floor for the log: a coefficient can reach 0, and log(0) is undefined.
MIN_COEFFICIENT = 0.5


class LeagueStrength(NamedTuple):
    coefficient: float
    avg_ca: float

    @property
    def log_coefficient(self) -> float:
        """League strength on a log scale.

        Reputation relates to ability logarithmically, not linearly. Fitting
        this table's own avg_ca against each form gives R2 0.834 for the raw
        coefficient and 0.912 for its log (RMSE 6.89 -> 5.02 CA points), so the
        log is what the model should see. An equal-sized raw gap matters far
        more at the bottom of the table than at the top.
        """
        return math.log(max(self.coefficient, MIN_COEFFICIENT))


# UEFA country coefficients, Coef. 24/25.
NATION: dict[str, LeagueStrength] = {
    'England': LeagueStrength(111.428, 146.59),
    'Spain': LeagueStrength(89.677, 140.64),
    'Italy': LeagueStrength(86.855, 139.33),
    'Germany': LeagueStrength(81.624, 136.24),
    'France': LeagueStrength(67.748, 133.08),
    'Netherlands': LeagueStrength(62.100, 119.88),
    'Portugal': LeagueStrength(59.149, 125.35),
    'Belgium': LeagueStrength(46.300, 117.79),
    'Turkey': LeagueStrength(40.850, 118.83),
    'Scotland': LeagueStrength(39.450, 111.39),
    'Switzerland': LeagueStrength(34.575, 116.98),
    'Ukraine': LeagueStrength(31.900, 105.95),
    'Serbia': LeagueStrength(31.775, 105.79),
    'Austria': LeagueStrength(31.300, 117.27),
    'Norway': LeagueStrength(29.250, 111.79),
    'Greece': LeagueStrength(28.825, 113.83),
    'Denmark': LeagueStrength(28.200, 115.32),
    'Czechia': LeagueStrength(27.800, 111.19),
    'Israel': LeagueStrength(26.125, 104.59),
    'Sweden': LeagueStrength(24.375, 109.19),
    'Croatia': LeagueStrength(23.025, 112.75),
    'Poland': LeagueStrength(23.000, 108.41),
    'Cyprus': LeagueStrength(22.725, 106.36),  # avg_ca derived
    'Hungary': LeagueStrength(21.625, 106.75),
    'Bulgaria': LeagueStrength(21.250, 101.98),
    'Slovakia': LeagueStrength(20.625, 100.01),
    'Romania': LeagueStrength(19.875, 107.11),
    'Azerbaijan': LeagueStrength(17.250, 100.34),  # avg_ca derived
    'Moldova': LeagueStrength(13.875, 95.59),  # avg_ca derived
    'Ireland': LeagueStrength(13.750, 91.04),
    'Slovenia': LeagueStrength(13.750, 105.43),
    'Kazakhstan': LeagueStrength(13.000, 94.17),  # avg_ca derived
    'Finland': LeagueStrength(11.750, 87.80),
    'Georgia': LeagueStrength(10.875, 90.28),  # avg_ca derived
    'Latvia': LeagueStrength(10.500, 83.83),
    'Liechtenstein': LeagueStrength(10.500, 89.51),  # avg_ca derived
    'Armenia': LeagueStrength(10.000, 88.44),  # avg_ca derived
    'Faroe Islands': LeagueStrength(9.875, 88.17),  # avg_ca derived
    'Northern Ireland': LeagueStrength(9.208, 77.94),
    'Bosnia and Herzegovina': LeagueStrength(9.125, 86.45),  # avg_ca derived
    'Lithuania': LeagueStrength(8.875, 85.84),  # avg_ca derived
    'Kosovo': LeagueStrength(8.541, 85.00),  # avg_ca derived
    'Iceland': LeagueStrength(8.417, 93.64),
    'Malta': LeagueStrength(7.875, 83.23),  # avg_ca derived
    'Luxembourg': LeagueStrength(7.625, 82.53),  # avg_ca derived
    'Estonia': LeagueStrength(7.582, 82.40),  # avg_ca derived
    'Belarus': LeagueStrength(7.250, 87.64),
    'North Macedonia': LeagueStrength(7.000, 80.66),  # avg_ca derived
    'Albania': LeagueStrength(6.375, 78.62),  # avg_ca derived
    'Montenegro': LeagueStrength(5.875, 76.84),  # avg_ca derived
    'Wales': LeagueStrength(5.416, 75.06),  # avg_ca derived
    'Gibraltar': LeagueStrength(5.291, 76.21),
    'Andorra': LeagueStrength(4.499, 71.01),  # avg_ca derived
    'San Marino': LeagueStrength(1.499, 47.03),  # avg_ca derived
}

# Leagues outside UEFA. ESTIMATED -- no coefficient exists for these, so each is
# placed beside the UEFA nation it most resembles. Tune against your own save.
NON_UEFA: dict[str, LeagueStrength] = {
    'Saudi Arabia': LeagueStrength(55.0, 124.0),   # ~ Portugal
    'Brazil': LeagueStrength(52.0, 123.0),         # ~ Portugal / Netherlands
    'Argentina': LeagueStrength(40.0, 118.0),      # ~ Turkey
    'USA': LeagueStrength(36.0, 115.0),            # ~ Switzerland
    'Mexico': LeagueStrength(34.0, 114.0),         # ~ Switzerland
    'Russia': LeagueStrength(33.0, 115.0),         # suspended from UEFA
    'Japan': LeagueStrength(30.0, 112.0),          # ~ Norway
    'South Korea': LeagueStrength(26.0, 109.0),    # ~ Israel
    'China': LeagueStrength(20.0, 104.0),          # ~ Slovakia
    'Australia': LeagueStrength(18.0, 101.0),      # ~ Romania
}

LEAGUE: dict[str, LeagueStrength] = {**NATION, **NON_UEFA}

# How much of its parent nation's coefficient a division at each tier carries.
# Calibrated by inverting the avg_ca ~ log(coefficient) fit against the observed
# average CA of 15 second-to-fourth tier divisions in a real export, then taking
# the median per tier. Tier 4 rests on one division, so trust it least.
TIER_FACTOR: dict[int, float] = {1: 1.0, 2: 0.245, 3: 0.142, 4: 0.065}

# FM division name -> (nation, tier). Tier 1 is a nation's top flight.
DIVISIONS: dict[str, tuple[str, int]] = {
    # --- UEFA top flights --------------------------------------------------
    'Premier League': ('England', 1),
    'LALIGA EA Sports': ('Spain', 1),
    'Serie A Enilive': ('Italy', 1),
    'Bundesliga': ('Germany', 1),
    "Ligue 1 McDonald's": ('France', 1),
    'Eredivisie': ('Netherlands', 1),
    'Liga Portugal Betclic': ('Portugal', 1),
    'Jupiler Pro League': ('Belgium', 1),
    'Trendyol Süper Lig': ('Turkey', 1),
    'William Hill Premiership': ('Scotland', 1),
    'Credit Suisse Super League': ('Switzerland', 1),
    'VBet Liha': ('Ukraine', 1),
    'Mozzart Bet SuperLiga': ('Serbia', 1),
    'Admiral Bundesliga': ('Austria', 1),
    'Eliteserien': ('Norway', 1),
    'Stoiximan Super League': ('Greece', 1),
    '3F Superliga': ('Denmark', 1),
    'Chance Liga': ('Czechia', 1),
    'Ligat TOTO Winner': ('Israel', 1),
    'Allsvenskan': ('Sweden', 1),
    'SuperSport Hrvatska nogometna liga': ('Croatia', 1),
    'PKO Bank Polski Ekstraklasa': ('Poland', 1),
    'OTP Bank Liga': ('Hungary', 1),
    'efbet Liga': ('Bulgaria', 1),
    'Niké Liga': ('Slovakia', 1),
    'Superliga': ('Romania', 1),
    'SSE Airtricity Premier Division': ('Ireland', 1),
    'Prva liga Telemach': ('Slovenia', 1),
    'Veikkausliiga': ('Finland', 1),
    'TonyBet Virslīga': ('Latvia', 1),
    'Sports Direct Premiership': ('Northern Ireland', 1),
    'Besta deild karla': ('Iceland', 1),
    'Vyšejšaja Liha': ('Belarus', 1),
    'Gibraltar Football League': ('Gibraltar', 1),

    # --- non-UEFA top flights (strengths in NON_UEFA are estimates) ---------
    'Roshn Saudi League': ('Saudi Arabia', 1),
    'Brasileirao Serie A Betano': ('Brazil', 1),
    'Brasileirão Série A Betano': ('Brazil', 1),
    'Liga Profesional de Fútbol': ('Argentina', 1),
    'Major League Soccer': ('USA', 1),
    'Liga BBVA MX': ('Mexico', 1),
    'Mir Rossiyskaya Premyer Liga': ('Russia', 1),
    'Meiji Yasuda J1 League': ('Japan', 1),
    'Hana One Q K League 1': ('South Korea', 1),
    'Chinese Super League': ('China', 1),
    'Isuzu UTE A-League': ('Australia', 1),

    # --- second tiers ------------------------------------------------------
    'Sky Bet Championship': ('England', 2),
    'LALIGA HYPERMOTION': ('Spain', 2),
    'Serie BKT': ('Italy', 2),
    '2. Bundesliga': ('Germany', 2),
    'Ligue 2 BKT': ('France', 2),
    'Keuken Kampioen Divisie': ('Netherlands', 2),
    'Betnation Divisie': ('Netherlands', 2),
    'Liga Portugal 2 Meu Super': ('Portugal', 2),
    'Challenger Pro League': ('Belgium', 2),
    'Trendyol 1. Lig': ('Turkey', 2),
    'William Hill Championship': ('Scotland', 2),
    'Hoval Promotion League': ('Switzerland', 2),
    'Persha Liha': ('Ukraine', 2),
    'Admiral 2. Liga': ('Austria', 2),
    'OBOS-ligaen': ('Norway', 2),
    'Super League 2 Betsson': ('Greece', 2),
    'NordicBet Liga': ('Denmark', 2),
    'Superettan': ('Sweden', 2),
    'Betclic I liga': ('Poland', 2),
    'Merkantil Bank Liga': ('Hungary', 2),
    'Liga 2': ('Romania', 2),
    'SSE Airtricity First Division': ('Ireland', 2),
    'Ykkosliiga': ('Finland', 2),
    'Ykkösliiga': ('Finland', 2),
    'Prva liga': ('Slovenia', 2),
    'Druha Liha': ('Ukraine', 3),
    'Druhi Liha': ('Ukraine', 3),

    # --- third tiers -------------------------------------------------------
    'Sky Bet League One': ('England', 3),
    '3. Liga': ('Germany', 3),
    'Championnat de France National': ('France', 3),
    'Primera Federación Grupo 1': ('Spain', 3),
    'Primera Federación Grupo 2': ('Spain', 3),
    'Liga 3 Placard': ('Portugal', 3),
    'Serie C NOW Girone A': ('Italy', 3),
    'Serie C NOW Girone B': ('Italy', 3),
    'Serie C NOW Girone C': ('Italy', 3),
    'Betclic II liga': ('Poland', 3),
    'Tweede Divisie': ('Netherlands', 3),
    'PostFinance 1. Liga Classic': ('Switzerland', 3),

    # --- fourth tiers ------------------------------------------------------
    'Sky Bet League Two': ('England', 4),
    'Campeonato de Portugal Prio': ('Portugal', 4),
}


def normalise_division(name) -> str:
    """Fold a division name to a stable lookup key.

    FM's sponsor names carry typographic apostrophes and accents that vary
    between exports and locales -- "Ligue 1 McDonald's" ships with U+2019, and a
    straight-quote copy would otherwise silently drop the entire French league.
    Compare case- and punctuation-insensitively instead.
    """
    text = unicodedata.normalize('NFKC', str(name))
    for curly, straight in (('’', "'"), ('‘', "'"),
                            ('“', '"'), ('”', '"')):
        text = text.replace(curly, straight)
    return ' '.join(text.split()).casefold()


_BY_DIVISION = {normalise_division(d): v for d, v in DIVISIONS.items()}


def lookup_division(division):
    """Return (nation, tier) for an FM division name, or None if unmapped."""
    return _BY_DIVISION.get(normalise_division(division))


def nation_for_division(division):
    """Nation for an FM division name, or None if it is not mapped."""
    found = lookup_division(division)
    return found[0] if found else None


def tier_for_division(division):
    """Tier for an FM division name, or None if it is not mapped."""
    found = lookup_division(division)
    return found[1] if found else None


def strength_for_division(division):
    """LeagueStrength for a division, scaled down for tiers below the top.

    A second tier is not its own country -- it is a fraction of its parent's
    strength, so one entry per nation covers every division beneath it.
    """
    found = lookup_division(division)
    if found is None:
        return None
    nation, tier = found
    parent = LEAGUE[nation]
    factor = TIER_FACTOR.get(tier, TIER_FACTOR[max(TIER_FACTOR)])
    if factor == 1.0:
        return parent
    coefficient = max(parent.coefficient * factor, MIN_COEFFICIENT)
    # Move avg_ca along the same fit the coefficients were calibrated against.
    avg_ca = parent.avg_ca + AVG_CA_PER_LOG_COEF * math.log(factor)
    return LeagueStrength(coefficient, avg_ca)


# Slope of avg_ca against log(coefficient) across the original 34 leagues.
AVG_CA_PER_LOG_COEF = 21.82


# A fixed centring point for league-interaction terms, derived from this
# reference table rather than from whatever dataset is loaded -- so the feature
# means the same thing at training time and at scoring time.
LOG_COEF_REFERENCE = math.log(
    sum(v.coefficient for v in NATION.values()) / len(NATION))
