"""Column names and human-readable stat labels.

`stats_dict` doubles as the list of columns preprocessing coerces to numeric,
so a stat only needs adding here once.
"""

# Columns the pipeline depends on by name.
VALUE_COL = 'Transfer Value'
NAME_COL = 'Name'
POINTS_COL = 'Pts/Gm'
LEAGUE_COL = 'League'
SCORE_COL = 'score'

# Sentinel used by Football Manager exports for players who cannot be bought.
NOT_FOR_SALE = 'Not for Sale'

stats_dict = {'xGP/90': 'Expected Goals Prevented per 90',
              'Hdrs L/90': 'Headers Lost per 90',
              'Sv %': 'Save Percentage',
              'xSv %': 'Expected Save Percentage',
              'Aer A/90': 'Aerial Actions Attempted per 90',
              'OP-Crs C/90': 'Open PLay Crosses Completed per 90',
              'OP-Crs A/90': 'Open Play Crosses Attempted per 90',
              'Conv %': 'Conversion Rate',
              'Mins/Gl': 'Minutes per Goal',
              'xA/90': 'Expected Assists per 90',
              'Saves/90': 'Saves per 90',
              'xG/90': 'Expected Goals per 90',
              'Gls/90': 'Goals per 90',
              'NP-xG/90': 'Non-Penalty Expected Goals per 90',
              'xG/shot': 'Expected Goals per Shot',
              'Shot %': 'Shot Percentage',
              'Hdrs W/90': 'Headers Won per 90',
              'K Hdrs/90': 'Key Headers per 90',
              'Cr C/A': 'Crosses Completed/Attempted',
              'Crs A/90': 'Crosses Attempted per 90',
              'Cr C/90': 'Crosses Completed per 90',
              'OP-Cr %': 'Open Play Cross Percentage',
              'Dist/90': 'Distance per 90',
              'Drb/90': 'Dribbles per 90',
              'Ps A/90': 'Passes Attempted per 90',
              'Ps C/90': 'Passes Completed per 90',
              'Sprints/90': 'Sprints per 90',
              'Poss Lost/90': 'Possession Lost per 90',
              'Pr passes/90': 'Progressive Passes per 90',
              'Shots Outside Box/90': 'Shots Outside Box per 90',
              'Shot/90': 'Shots per 90',
              'ShT/90': 'Shots on Target per 90',
              'Blk/90': 'Blocks per 90',
              'Clr/90': 'Clearances per 90',
              'Int/90': 'Interceptions per 90',
              'Poss Won/90': 'Possession Won per 90',
              'OP-KP/90': 'OP Key Passes per 90',
              'Pres C/90': 'Pressures Completed per 90',
              'Shts Blckd/90': 'Shots Blocked per 90',
              'Hdr %': 'Header Percentage',
              'Tck/90': 'Tackles per 90',
              'Tck R': 'Tackle Won Ratio',
              'Pas %': 'Pass Percentage',
              'Pres A/90': 'Pressures Attempted per 90',
              'K Tck/90': 'Key Tackles per 90',
              'Ch C/90': 'Chances Created per 90',
              'K Ps/90': 'Key Passes per 90',
              'Asts/90': 'Assists per 90',
              'Pens Saved Ratio': 'Penalties Saved Ratio',
              'SvRatio': 'Saves vs Expected Saves',
              }


# Derived by preprocessing rather than read from the CSV.
DERIVED_COLS = ('SvRatio',)

# Everything preprocessing should coerce to numeric.
NUMERIC_COLUMNS = tuple(c for c in stats_dict if c not in DERIVED_COLS) + (POINTS_COL,)
