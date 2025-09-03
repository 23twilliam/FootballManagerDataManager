import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefer
import matplotlib.pyplot as plt
from scipy import stats
import math

def plot_stat_percentile_bar(df, df2, stat_col, label, display_name):
    # Extract the specific value from df2 for the player
    player_value = df2.loc[df2['Name'] == label, stat_col].values[0]

    # Calculate percentile
    percentile = stats.percentileofscore(df[stat_col].dropna(), player_value, kind='rank')

    # Plot bar
    plt.bar(display_name, percentile, color='lightblue')

    # Annotate with truncated percentile
    plt.annotate(
        math.trunc(percentile),
        (display_name, percentile),
        ha='center',
        color='white',
        xytext=(display_name, percentile + 1),
        fontsize=12
    )